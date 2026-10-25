#include "memforge/prefix_cache.hpp"

namespace memforge {

PrefixCache::PrefixCache(std::shared_ptr<BlockAllocator> allocator, size_t block_size)
    : allocator_(std::move(allocator))
    , block_size_(block_size ? block_size : DEFAULT_BLOCK_SIZE) {
}

size_t PrefixCache::shareable_block_count(size_t num_tokens) const {
    size_t full = num_tokens / block_size_;
    // When the token count lands exactly on a block boundary, the final full
    // block must stay private: the next token written has to go somewhere, and a
    // fully cached sequence would otherwise leave the prefill with an empty
    // suffix -- a zero-length forward pass.
    if (full > 0 && num_tokens % block_size_ == 0) {
        full -= 1;
    }
    return full;
}

std::vector<BlockId> PrefixCache::acquire_shared_prefix(const std::vector<TokenId>& tokens) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<BlockId> acquired;
    const size_t candidates = shareable_block_count(tokens.size());
    if (candidates == 0) {
        return acquired;
    }
    acquired.reserve(candidates);

    uint64_t parent = 0;
    for (size_t i = 0; i < candidates; ++i) {
        const TokenId* begin = tokens.data() + i * block_size_;
        const uint64_t h = hash_block(parent, begin, block_size_);

        auto it = by_hash_.find(h);
        if (it == by_hash_.end()) {
            misses_++;
            break;
        }

        // Verify the tokens rather than trusting the hash (see CachedBlock).
        const CachedBlock& cached = it->second;
        if (cached.tokens.size() != block_size_ ||
            !std::equal(cached.tokens.begin(), cached.tokens.end(), begin)) {
            misses_++;
            break;
        }

        // A failed add_ref means the entry is stale -- the block was freed and
        // possibly reallocated. Treat it as a miss and drop the bad entry.
        if (!allocator_->add_ref(cached.block_id)) {
            misses_++;
            by_block_.erase(cached.block_id);
            by_hash_.erase(it);
            break;
        }

        acquired.push_back(cached.block_id);
        hits_++;
        parent = h;
    }

    return acquired;
}

void PrefixCache::release_acquired(const std::vector<BlockId>& block_ids) {
    if (block_ids.empty()) {
        return;
    }
    // Deliberately not holding mutex_ here: release() may drive blocks to zero
    // references, and evict() takes mutex_ itself.
    std::vector<BlockId> freed = allocator_->release(block_ids);
    if (!freed.empty()) {
        evict(freed);
    }
}

size_t PrefixCache::register_full_blocks(const std::vector<TokenId>& tokens,
                                         const std::vector<BlockId>& block_table) {
    std::lock_guard<std::mutex> lock(mutex_);

    const size_t candidates = std::min(shareable_block_count(tokens.size()), block_table.size());
    size_t registered = 0;
    uint64_t parent = 0;

    for (size_t i = 0; i < candidates; ++i) {
        const TokenId* begin = tokens.data() + i * block_size_;
        const uint64_t h = hash_block(parent, begin, block_size_);
        parent = h;

        const BlockId id = block_table[i];
        if (id < 0) {
            break;
        }
        // Already cached under this hash (the common case for a shared prefix),
        // or this block is already registered under a different hash. Either
        // way, leave the existing entry alone: first registration wins, which
        // keeps `by_block_` a genuine one-to-one reverse index.
        if (by_hash_.count(h) || by_block_.count(id)) {
            continue;
        }

        CachedBlock entry;
        entry.block_id = id;
        entry.block_hash = h;
        entry.tokens.assign(begin, begin + block_size_);

        by_hash_.emplace(h, std::move(entry));
        by_block_.emplace(id, h);
        registered++;
    }

    return registered;
}

void PrefixCache::evict(const std::vector<BlockId>& released) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (BlockId id : released) {
        auto it = by_block_.find(id);
        if (it == by_block_.end()) {
            continue;
        }
        by_hash_.erase(it->second);
        by_block_.erase(it);
    }
}

uint64_t PrefixCache::first_block_hash(const std::vector<TokenId>& tokens) const {
    if (shareable_block_count(tokens.size()) == 0) {
        return 0;
    }
    return hash_block(0, tokens.data(), block_size_);
}

void PrefixCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    by_hash_.clear();
    by_block_.clear();
    hits_ = 0;
    misses_ = 0;
}

size_t PrefixCache::num_cached_blocks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return by_hash_.size();
}

uint64_t PrefixCache::num_hits() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return hits_;
}

uint64_t PrefixCache::num_misses() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return misses_;
}

double PrefixCache::hit_rate() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t total = hits_ + misses_;
    return total ? static_cast<double>(hits_) / static_cast<double>(total) : 0.0;
}

}  // namespace memforge
