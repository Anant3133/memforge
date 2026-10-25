#include "memforge/block_allocator.hpp"
#include <algorithm>

namespace memforge {

BlockAllocator::BlockAllocator(size_t total_blocks, size_t block_size)
    : total_blocks_(total_blocks)
    , block_size_(block_size) {
    blocks_.resize(total_blocks_);
    for (size_t i = 0; i < total_blocks_; ++i) {
        blocks_[i].block_id = static_cast<BlockId>(i);
        blocks_[i].ref_count = 0;
        blocks_[i].is_allocated = false;
        free_list_.push_back(static_cast<BlockId>(i));
    }
}

bool BlockAllocator::can_allocate(size_t num_blocks) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_list_.size() >= num_blocks;
}

BlockId BlockAllocator::allocate() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_list_.empty()) {
        return INVALID_BLOCK_ID;
    }
    BlockId block_id = free_list_.front();
    free_list_.pop_front();

    blocks_[block_id].is_allocated = true;
    blocks_[block_id].ref_count = 1;
    blocks_[block_id].block_hash = 0;
    return block_id;
}

std::vector<BlockId> BlockAllocator::allocate_blocks(size_t num_blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_list_.size() < num_blocks) {
        throw std::runtime_error("Out of memory in BlockAllocator: cannot allocate " + 
                                 std::to_string(num_blocks) + " blocks. Free: " + 
                                 std::to_string(free_list_.size()));
    }

    std::vector<BlockId> allocated;
    allocated.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        BlockId block_id = free_list_.front();
        free_list_.pop_front();

        blocks_[block_id].is_allocated = true;
        blocks_[block_id].ref_count = 1;
        blocks_[block_id].block_hash = 0;
        allocated.push_back(block_id);
    }
    return allocated;
}

bool BlockAllocator::release_locked(BlockId block_id) {
    if (block_id < 0 || static_cast<size_t>(block_id) >= total_blocks_) {
        return false;
    }
    if (!blocks_[block_id].is_allocated) {
        return false;
    }

    blocks_[block_id].ref_count--;
    if (blocks_[block_id].ref_count > 0) {
        // Still shared by another sequence -- not actually free.
        return false;
    }

    blocks_[block_id].is_allocated = false;
    blocks_[block_id].ref_count = 0;
    // The block is about to be handed to an unrelated sequence, so it must not
    // stay discoverable under its old prefix hash.
    blocks_[block_id].block_hash = 0;
    free_list_.push_back(block_id);
    return true;
}

void BlockAllocator::free(BlockId block_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    release_locked(block_id);
}

void BlockAllocator::free_blocks(const std::vector<BlockId>& block_ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (BlockId id : block_ids) {
        release_locked(id);
    }
}

std::vector<BlockId> BlockAllocator::release(const std::vector<BlockId>& block_ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<BlockId> freed;
    freed.reserve(block_ids.size());
    for (BlockId id : block_ids) {
        if (release_locked(id)) {
            freed.push_back(id);
        }
    }
    return freed;
}

bool BlockAllocator::add_ref(BlockId block_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (block_id < 0 || static_cast<size_t>(block_id) >= total_blocks_) {
        return false;
    }
    // Refusing to reference a free block is what turns a stale PrefixCache entry
    // into a clean miss instead of a block full of someone else's KV.
    if (!blocks_[block_id].is_allocated || blocks_[block_id].ref_count <= 0) {
        return false;
    }
    blocks_[block_id].ref_count++;
    return true;
}

std::optional<std::pair<BlockId, BlockId>> BlockAllocator::prepare_write(Sequence& seq, size_t logical_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::vector<BlockId>& table = seq.get_block_table();
    if (logical_index >= table.size()) {
        return std::nullopt;
    }

    BlockId old_id = table[logical_index];
    if (old_id < 0 || static_cast<size_t>(old_id) >= total_blocks_) {
        return std::nullopt;
    }
    if (!blocks_[old_id].is_allocated || blocks_[old_id].ref_count <= 1) {
        return std::nullopt;  // already private: write straight through
    }
    if (free_list_.empty()) {
        return std::nullopt;  // nowhere to copy to; caller must not write
    }

    BlockId new_id = free_list_.front();
    free_list_.pop_front();
    blocks_[new_id].is_allocated = true;
    blocks_[new_id].ref_count = 1;
    blocks_[new_id].block_hash = 0;  // a private copy is never a cache entry

    seq.set_block(logical_index, new_id);
    // Safe to decrement directly: ref_count was > 1, so this cannot reach 0 and
    // the old block stays allocated for whoever else references it.
    blocks_[old_id].ref_count--;

    return std::make_pair(old_id, new_id);
}

int BlockAllocator::get_ref_count(BlockId block_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (block_id < 0 || static_cast<size_t>(block_id) >= total_blocks_) {
        return 0;
    }
    return blocks_[block_id].is_allocated ? blocks_[block_id].ref_count : 0;
}

size_t BlockAllocator::get_num_shared_blocks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const PhysicalBlock& block : blocks_) {
        if (block.is_allocated && block.ref_count > 1) {
            count++;
        }
    }
    return count;
}

size_t BlockAllocator::get_num_blocks_saved() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t refs = 0;
    size_t allocated = 0;
    for (const PhysicalBlock& block : blocks_) {
        if (block.is_allocated) {
            refs += static_cast<size_t>(block.ref_count);
            allocated++;
        }
    }
    return refs - allocated;
}

std::vector<BlockId> BlockAllocator::free_sequence(Sequence& seq) {
    std::vector<BlockId> freed = release(seq.get_block_table());
    seq.clear_blocks();
    return freed;
}

bool BlockAllocator::ensure_capacity(Sequence& seq) {
    size_t needed_blocks = seq.get_num_required_blocks();
    size_t current_blocks = seq.get_num_allocated_blocks();

    if (needed_blocks <= current_blocks) {
        return true;
    }

    size_t additional = needed_blocks - current_blocks;
    if (!can_allocate(additional)) {
        return false;
    }

    std::vector<BlockId> new_blocks = allocate_blocks(additional);
    for (BlockId id : new_blocks) {
        seq.add_block(id);
    }
    return true;
}

size_t BlockAllocator::get_num_free_blocks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_list_.size();
}

size_t BlockAllocator::get_num_allocated_blocks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_blocks_ - free_list_.size();
}

double BlockAllocator::get_utilization_ratio() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (total_blocks_ == 0) return 0.0;
    return static_cast<double>(total_blocks_ - free_list_.size()) / static_cast<double>(total_blocks_);
}

void BlockAllocator::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    free_list_.clear();
    for (size_t i = 0; i < total_blocks_; ++i) {
        blocks_[i].is_allocated = false;
        blocks_[i].ref_count = 0;
        blocks_[i].block_hash = 0;
        free_list_.push_back(static_cast<BlockId>(i));
    }
}

} // namespace memforge
