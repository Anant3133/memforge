#pragma once

#include "memforge/types.hpp"
#include "memforge/sequence.hpp"
#include <vector>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>
#include <stdexcept>

namespace memforge {

/**
 * @brief Thread-safe physical block allocator for Paged KV-Cache.
 * Manages fixed-size memory blocks (e.g. 16 tokens/block) and provides
 * OS-style virtual memory paging over GPU/Host memory buffers.
 */
class BlockAllocator {
public:
    explicit BlockAllocator(size_t total_blocks, size_t block_size = DEFAULT_BLOCK_SIZE);
    ~BlockAllocator() = default;

    // Non-copyable
    BlockAllocator(const BlockAllocator&) = delete;
    BlockAllocator& operator=(const BlockAllocator&) = delete;

    /**
     * @brief Checks if enough free physical blocks exist to satisfy the requested count.
     */
    bool can_allocate(size_t num_blocks) const;

    /**
     * @brief Allocates a single physical block.
     * @return BlockId Allocated block ID. Returns INVALID_BLOCK_ID if out of memory.
     */
    BlockId allocate();

    /**
     * @brief Allocates a batch of physical blocks.
     * @param num_blocks Number of blocks required.
     * @return std::vector<BlockId> List of allocated block IDs. Throws if not enough blocks.
     */
    std::vector<BlockId> allocate_blocks(size_t num_blocks);

    /**
     * @brief Frees a physical block, returning it to the pool.
     */
    void free(BlockId block_id);

    /**
     * @brief Frees a list of physical blocks.
     */
    void free_blocks(const std::vector<BlockId>& block_ids);

    /**
     * @brief Frees all blocks associated with a sequence.
     */
    void free_sequence(Sequence& seq);

    /**
     * @brief Ensures a sequence has enough physical blocks for its current token length.
     * If more tokens were added and crossed the block boundary, allocates a new block.
     */
    bool ensure_capacity(Sequence& seq);

    // ------------------------------------------------------- prefix sharing

    /**
     * @brief Takes an additional reference on an already-allocated block.
     *
     * This is the one operation that makes a block *shared*: a prefix-cache hit
     * points a second sequence's block table at a physical block that another
     * sequence is already using. Returns false if the block is out of range or
     * not currently allocated (a stale registry entry), in which case the caller
     * must treat the lookup as a miss rather than proceeding.
     */
    bool add_ref(BlockId block_id);

    /**
     * @brief Frees blocks, reporting only those whose ref_count actually hit 0.
     *
     * The returned ids are the blocks that genuinely returned to the free list
     * and so must be dropped from the PrefixCache registry. The caller forwards
     * them to `PrefixCache::evict()` *after* this call returns -- never from
     * inside the allocator, because that would invert the documented lock order
     * (PrefixCache::mutex_ -> BlockAllocator::mutex_) and deadlock.
     */
    std::vector<BlockId> release(const std::vector<BlockId>& block_ids);

    /**
     * @brief Copy-on-write guard. Call before writing to a sequence's logical block.
     *
     * If that block is shared (ref_count > 1), a private replacement is
     * allocated, the sequence's block table is rewritten to point at it, and the
     * shared reference is dropped. Returns {old_physical, new_physical} so the
     * caller can copy the old block's KV contents into the new one before
     * writing; the copy itself lives in Python because the pool tensors do.
     *
     * Returns std::nullopt when the block is already private -- the common case,
     * since only full blocks are ever shared and decode writes the trailing
     * (private) block. Also returns nullopt when no block is free to copy into,
     * which the caller must treat as "cannot write now", not as "write anyway".
     */
    std::optional<std::pair<BlockId, BlockId>> prepare_write(Sequence& seq, size_t logical_index);

    /// Reference count of a physical block; 0 when free or out of range.
    int get_ref_count(BlockId block_id) const;

    /// Allocated blocks carrying more than one reference, i.e. actually shared.
    size_t get_num_shared_blocks() const;

    /**
     * @brief Blocks saved by sharing: sum(ref_count) - count(allocated blocks).
     *
     * The honest savings figure -- how many extra block-allocations the pool
     * would have had to serve without sharing.
     */
    size_t get_num_blocks_saved() const;

    // Telemetry and statistics
    size_t get_num_free_blocks() const;
    size_t get_num_allocated_blocks() const;
    size_t get_total_blocks() const { return total_blocks_; }
    size_t get_block_size() const { return block_size_; }
    double get_utilization_ratio() const;

    // Reset allocator (for testing/benchmarks)
    void reset();

private:
    /// Drop one reference to `block_id`, returning true if it reached 0 and went
    /// back on the free list. Caller must already hold `mutex_`; this is the one
    /// place block release is implemented, so `free()` and `release()` cannot
    /// drift apart.
    bool release_locked(BlockId block_id);

    size_t total_blocks_;
    size_t block_size_;
    std::vector<PhysicalBlock> blocks_;
    std::deque<BlockId> free_list_;
    mutable std::mutex mutex_;
};

} // namespace memforge
