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
        allocated.push_back(block_id);
    }
    return allocated;
}

void BlockAllocator::free(BlockId block_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (block_id < 0 || static_cast<size_t>(block_id) >= total_blocks_) {
        return;
    }

    if (blocks_[block_id].is_allocated) {
        blocks_[block_id].ref_count--;
        if (blocks_[block_id].ref_count <= 0) {
            blocks_[block_id].is_allocated = false;
            blocks_[block_id].ref_count = 0;
            free_list_.push_back(block_id);
        }
    }
}

void BlockAllocator::free_blocks(const std::vector<BlockId>& block_ids) {
    for (BlockId id : block_ids) {
        free(id);
    }
}

void BlockAllocator::free_sequence(Sequence& seq) {
    free_blocks(seq.get_block_table());
    seq.clear_blocks();
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
        free_list_.push_back(static_cast<BlockId>(i));
    }
}

} // namespace memforge
