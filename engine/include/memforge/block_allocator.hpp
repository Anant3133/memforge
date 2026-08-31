#pragma once

#include "memforge/types.hpp"
#include "memforge/sequence.hpp"
#include <vector>
#include <deque>
#include <mutex>
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

    // Telemetry and statistics
    size_t get_num_free_blocks() const;
    size_t get_num_allocated_blocks() const;
    size_t get_total_blocks() const { return total_blocks_; }
    size_t get_block_size() const { return block_size_; }
    double get_utilization_ratio() const;

    // Reset allocator (for testing/benchmarks)
    void reset();

private:
    size_t total_blocks_;
    size_t block_size_;
    std::vector<PhysicalBlock> blocks_;
    std::deque<BlockId> free_list_;
    mutable std::mutex mutex_;
};

} // namespace memforge
