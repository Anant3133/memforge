#include "memforge/scheduler.hpp"
#include <algorithm>

namespace memforge {

Scheduler::Scheduler(std::shared_ptr<BlockAllocator> allocator, size_t max_num_seqs)
    : allocator_(std::move(allocator))
    , max_num_seqs_(max_num_seqs) {
}

SequencePtr Scheduler::add_sequence(SequenceId id, const std::vector<TokenId>& prompt_tokens, size_t max_new_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto seq = std::make_shared<Sequence>(id, prompt_tokens, max_new_tokens, allocator_->get_block_size());
    waiting_queue_.push_back(seq);
    all_sequences_[id] = seq;
    return seq;
}

SchedulerStepResult Scheduler::step() {
    std::lock_guard<std::mutex> lock(mutex_);
    SchedulerStepResult result;

    // 1. Check running sequences: allocate blocks if needed for decoding
    std::vector<SequencePtr> still_running;
    for (auto& seq : running_queue_) {
        if (seq->is_finished()) {
            allocator_->free_sequence(*seq);
            result.finished_sequence_ids.push_back(seq->get_id());
            continue;
        }

        // Ensure sequence has enough capacity for next token
        if (!allocator_->ensure_capacity(*seq)) {
            // Memory pressure: preemption or wait
            seq->set_status(SequenceStatus::SWAPPED);
            continue;
        }

        still_running.push_back(seq);
    }
    running_queue_ = std::move(still_running);

    // 2. Promote waiting sequences to running if slots and memory are available
    while (!waiting_queue_.empty() && running_queue_.size() < max_num_seqs_) {
        auto seq = waiting_queue_.front();
        size_t needed_blocks = seq->get_num_required_blocks();

        if (!allocator_->can_allocate(needed_blocks)) {
            // Cannot schedule more sequences in this iteration
            break;
        }

        waiting_queue_.pop_front();
        std::vector<BlockId> blocks = allocator_->allocate_blocks(needed_blocks);
        for (BlockId id : blocks) {
            seq->add_block(id);
        }

        seq->set_status(SequenceStatus::RUNNING);
        running_queue_.push_back(seq);
    }

    // 3. Build step result metadata for active sequences
    result.running_sequences.reserve(running_queue_.size());
    for (auto& seq : running_queue_) {
        SequenceStepInfo info;
        info.seq_id = seq->get_id();
        info.block_table = seq->get_block_table();
        info.prompt_length = seq->get_prompt_len();
        info.output_length = seq->get_output_len();
        info.total_length = seq->get_total_len();
        info.status = seq->get_status();
        info.is_prefill = !seq->has_completed_prefill();

        if (info.is_prefill) {
            seq->mark_prefill_completed();
        }

        result.running_sequences.push_back(info);
    }

    result.num_free_blocks = allocator_->get_num_free_blocks();
    result.total_blocks = allocator_->get_total_blocks();
    result.memory_utilization_ratio = allocator_->get_utilization_ratio();

    return result;
}

void Scheduler::update_sequence(SequenceId seq_id, TokenId new_token, bool is_finished) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = all_sequences_.find(seq_id);
    if (it != all_sequences_.end()) {
        auto& seq = it->second;
        seq->append_token(new_token);
        if (is_finished) {
            seq->finish();
        }
    }
}

bool Scheduler::abort_sequence(SequenceId seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = all_sequences_.find(seq_id);
    if (it == all_sequences_.end()) {
        return false;
    }

    auto& seq = it->second;
    seq->set_status(SequenceStatus::ABORTED);
    allocator_->free_sequence(*seq);

    // Remove from waiting if present
    for (auto wit = waiting_queue_.begin(); wit != waiting_queue_.end(); ++wit) {
        if ((*wit)->get_id() == seq_id) {
            waiting_queue_.erase(wit);
            break;
        }
    }

    return true;
}

size_t Scheduler::get_num_waiting() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return waiting_queue_.size();
}

size_t Scheduler::get_num_running() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return running_queue_.size();
}

size_t Scheduler::get_num_finished() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& pair : all_sequences_) {
        if (pair.second->is_finished()) {
            count++;
        }
    }
    return count;
}

bool Scheduler::has_unfinished_sequences() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !waiting_queue_.empty() || !running_queue_.empty();
}

SequencePtr Scheduler::get_sequence(SequenceId seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = all_sequences_.find(seq_id);
    if (it != all_sequences_.end()) {
        return it->second;
    }
    return nullptr;
}

} // namespace memforge
