#pragma once

#include "memforge/types.hpp"
#include "memforge/block_allocator.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace memforge {

/**
 * @brief 64-bit mixing function (splitmix64 finalizer).
 *
 * Used to avalanche the chained hash so that two prefixes differing in one
 * token id do not land in adjacent hash buckets.
 */
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/**
 * @brief Chained prefix hash for one block of tokens.
 *
 * `parent_hash` is the hash of the *preceding* block (0 for the first block),
 * which is what makes this a prefix hash rather than a block hash. That chaining
 * is load-bearing for correctness: a block's Key/Value tensors are a function of
 * every token before it, because attention is causal and cumulative. Hashing
 * only a block's own tokens would match two sequences that happen to share a
 * middle fragment but diverge earlier, and their KV would be completely
 * different -- yielding fluent, confident, wrong output rather than a crash.
 *
 * Never returns 0, so `PhysicalBlock::block_hash == 0` unambiguously means
 * "not registered".
 */
inline uint64_t hash_block(uint64_t parent_hash, const TokenId* tokens, size_t n) {
    // FNV-1a over the token ids, seeded with the mixed parent hash.
    uint64_t h = splitmix64(parent_hash ^ 0xCBF29CE484222325ULL);
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(static_cast<uint32_t>(tokens[i]));
        h *= 0x100000001B3ULL;  // FNV prime
    }
    uint64_t out = splitmix64(h);
    return out == 0 ? 1 : out;
}

/**
 * @brief One cached, shareable physical block.
 *
 * `tokens` is kept so lookups can *verify* rather than trust the hash. A 64-bit
 * collision is vanishingly unlikely, but "vanishingly unlikely" combined with
 * "silently serves the wrong KV cache" is a bad trade. 16 int32 is 64 bytes and
 * the memcmp is free next to a model forward pass, so the entire collision
 * failure mode is removed rather than merely made improbable.
 */
struct CachedBlock {
    BlockId block_id{INVALID_BLOCK_ID};
    uint64_t block_hash{0};
    std::vector<TokenId> tokens;
    uint64_t last_hit_timestamp{0};
};

/**
 * @brief Registry mapping chained prefix hashes to shareable physical blocks.
 *
 * Only *full* blocks are ever registered. A partially filled block is still
 * being written to, so its contents are not final and its hash is not yet
 * meaningful. A prompt's trailing block therefore always stays private, which is
 * what keeps the decode path copy-on-write-free: decode only ever writes the
 * trailing block.
 *
 * Lifetime policy (stage 1): an entry lives exactly as long as some sequence
 * references its block. When the last reference drops, `evict()` removes the
 * entry so a recycled physical block can never be served for stale content.
 * Retaining zero-reference entries with LRU eviction -- which is what keeps a
 * system prompt warm *between* requests -- is a deliberate follow-up, not part
 * of this stage.
 *
 * LOCK ORDER: PrefixCache::mutex_ -> BlockAllocator::mutex_, never the reverse.
 * Methods here may call into the allocator while holding `mutex_`; the allocator
 * must never call back into this class while holding its own. That is why
 * `BlockAllocator::release()` *returns* the ids that hit zero instead of
 * evicting them itself.
 */
class PrefixCache {
public:
    PrefixCache(std::shared_ptr<BlockAllocator> allocator, size_t block_size);

    // Non-copyable: it owns reference counts held in the allocator.
    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    /**
     * @brief Find the longest cached prefix of `tokens` and take a reference on it.
     *
     * Walks block by block, stopping at the first miss -- block i+1 cannot be
     * shared if block i differs, even if its tokens coincide. Bumps `ref_count`
     * on every hit, so the returned blocks are owned by the caller and must
     * either be installed into a sequence's block table or handed back via
     * `release_acquired()`.
     *
     * @return Physical block ids for logical blocks 0..k-1, possibly empty.
     */
    std::vector<BlockId> acquire_shared_prefix(const std::vector<TokenId>& tokens);

    /**
     * @brief Hand back references taken by `acquire_shared_prefix()`.
     *
     * Required on the admission failure path: a prefix hit followed by "not
     * enough blocks for the private tail" must not strand the references, or
     * those blocks can never return to the pool.
     */
    void release_acquired(const std::vector<BlockId>& block_ids);

    /**
     * @brief Register every full block of a sequence whose KV is now resident.
     *
     * Call after a prefill has actually written the KV, never before -- an entry
     * registered ahead of its data would serve zeros to the next hit. Blocks
     * already covered by a cache hit are skipped, as is the trailing block when
     * the token count lands exactly on a block boundary (it must stay private so
     * the next write has somewhere to go).
     *
     * @return Number of newly registered blocks.
     */
    size_t register_full_blocks(const std::vector<TokenId>& tokens,
                                const std::vector<BlockId>& block_table);

    /**
     * @brief Drop registry entries for blocks that have returned to the free list.
     *
     * Fed from `BlockAllocator::release()`'s return value. This is the mechanism
     * that makes stale reuse impossible: without it a finished sequence's block
     * goes back to the pool, gets handed to an unrelated request, and the next
     * hash hit returns a block full of someone else's conversation.
     */
    void evict(const std::vector<BlockId>& released);

    /// Forget everything. Does not touch reference counts (for tests/benchmarks).
    void clear();

    /**
     * @brief Chained hash of the first block of `tokens`, or 0 if none is shareable.
     *
     * Lets the scheduler recognise two waiting requests that begin with the same
     * prefix *before* either has been prefilled. Nothing can be shared until a
     * block's KV is actually written, so without this the first burst of
     * identical prompts all miss and each computes its own copy.
     */
    uint64_t first_block_hash(const std::vector<TokenId>& tokens) const;

    // ----------------------------------------------------------- telemetry

    size_t num_cached_blocks() const;
    uint64_t num_hits() const;
    uint64_t num_misses() const;
    double hit_rate() const;
    size_t get_block_size() const { return block_size_; }

private:
    /// How many leading blocks of `num_tokens` are eligible for sharing.
    size_t shareable_block_count(size_t num_tokens) const;

    std::shared_ptr<BlockAllocator> allocator_;
    size_t block_size_;

    std::unordered_map<uint64_t, CachedBlock> by_hash_;
    /// Reverse index, so `evict()` can find an entry from its block id. Without
    /// this the registry would outlive the blocks it points at.
    std::unordered_map<BlockId, uint64_t> by_block_;

    uint64_t hits_{0};
    uint64_t misses_{0};
    uint64_t clock_{0};  // monotonic counter for LRU ordering (stage 2)

    mutable std::mutex mutex_;
};

}  // namespace memforge
