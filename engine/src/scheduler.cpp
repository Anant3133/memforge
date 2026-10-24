#include "memforge/scheduler.hpp"
#include <algorithm>
#include <unordered_set>

namespace memforge {

Scheduler::Scheduler(std::shared_ptr<BlockAllocator> allocator, size_t max_num_seqs,
                     std::shared_ptr<PrefixCache> prefix_cache)
    : allocator_(std::move(allocator))
    , prefix_cache_(std::move(prefix_cache))
    , max_num_seqs_(max_num_seqs) {
}

void Scheduler::release_sequence_blocks(Sequence& seq) {
    // release() reports only the blocks whose ref_count actually reached zero.
    // A shared block released here stays allocated for whoever else references
    // it, which is why preemption no longer evicts a prefix another sequence is
    // still reading.
    std::vector<BlockId> freed = allocator_->release(seq.get_block_table());
    seq.clear_blocks();
    if (prefix_cache_ && !freed.empty()) {
        // Mandatory: a block back on the free list must stop being discoverable
        // under its old prefix hash, or it will later be served to a sequence
        // whose tokens it no longer holds.
        prefix_cache_->evict(freed);
    }
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
    std::vector<SequencePtr> preempted;
    for (auto& seq : running_queue_) {
        if (seq->is_finished()) {
            release_sequence_blocks(*seq);
            result.finished_sequence_ids.push_back(seq->get_id());
            continue;
        }

        // Ensure sequence has enough capacity for next token
        if (!allocator_->ensure_capacity(*seq)) {
            // Memory pressure: preempt by recomputation. Release the physical
            // blocks back to the pool so shorter sequences can make progress,
            // and re-queue at the head so this sequence is admitted first once
            // memory frees up. Generated tokens are retained in the Sequence, so
            // the ML layer re-runs prefill over prompt + output on readmission.
            release_sequence_blocks(*seq);
            seq->reset_prefill();
            seq->set_status(SequenceStatus::WAITING);
            preempted.push_back(seq);
            result.preempted_sequence_ids.push_back(seq->get_id());
            continue;
        }

        still_running.push_back(seq);
    }
    running_queue_ = std::move(still_running);

    // Preempted sequences regain priority ahead of never-started requests.
    for (auto it = preempted.rbegin(); it != preempted.rend(); ++it) {
        waiting_queue_.push_front(*it);
    }

    // 2. Promote waiting sequences to running if slots and memory are available
    //
    // First-block hashes of the fresh (cache-missing) prefills admitted during
    // THIS step. Nothing is shareable until its KV has actually been written, so
    // a burst of identical prompts would otherwise all miss together and each
    // compute and store its own copy of the same prefix. Holding the duplicates
    // back for one step lets the first one prefill and register, after which the
    // rest share it. Requests with *different* prefixes are unaffected and still
    // all admit in a single step.
    std::unordered_set<uint64_t> fresh_prefixes;
    while (!waiting_queue_.empty() && running_queue_.size() < max_num_seqs_) {
        auto seq = waiting_queue_.front();
        size_t needed_blocks = seq->get_num_required_blocks();

        if (needed_blocks > allocator_->get_total_blocks()) {
            // Unsatisfiable at any point in time: the request alone exceeds the
            // entire pool. Abort it instead of head-of-line blocking the queue.
            waiting_queue_.pop_front();
            seq->set_status(SequenceStatus::ABORTED);
            result.aborted_sequence_ids.push_back(seq->get_id());
            continue;
        }

        // Reuse any cached prefix first, so only the uncached remainder has to
        // come out of the pool. A preempted sequence arrives here with
        // prompt + already-generated tokens, which is exactly what it will
        // recompute, so hashing all of them is correct.
        std::vector<BlockId> shared;
        if (prefix_cache_) {
            const std::vector<TokenId> tokens = seq->get_all_tokens();
            shared = prefix_cache_->acquire_shared_prefix(tokens);
            if (shared.empty()) {
                const uint64_t first = prefix_cache_->first_block_hash(tokens);
                if (first != 0) {
                    if (fresh_prefixes.count(first)) {
                        // A sequence admitted this step is about to prefill this
                        // exact prefix. Wait one step and share its blocks rather
                        // than storing a second copy. Breaking (not skipping)
                        // keeps the queue FIFO, so nothing can starve.
                        break;
                    }
                    fresh_prefixes.insert(first);
                }
            }
        }
        const size_t fresh_blocks =
            needed_blocks > shared.size() ? needed_blocks - shared.size() : 0;

        if (!allocator_->can_allocate(fresh_blocks)) {
            // Cannot schedule more sequences in this iteration. Hand back the
            // references acquire_shared_prefix() just took -- holding them would
            // pin those blocks forever, since nothing else would ever release
            // them.
            if (prefix_cache_ && !shared.empty()) {
                prefix_cache_->release_acquired(shared);
            }
            break;
        }

        waiting_queue_.pop_front();
        for (BlockId id : shared) {
            seq->add_block(id);
        }
        if (fresh_blocks) {
            std::vector<BlockId> blocks = allocator_->allocate_blocks(fresh_blocks);
            for (BlockId id : blocks) {
                seq->add_block(id);
            }
        }
        // Tells the ML layer how many leading tokens are already resident, so
        // prefill neither recomputes them nor overwrites the shared blocks.
        seq->set_num_cached_tokens(shared.size() * allocator_->get_block_size());

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
        info.num_cached_tokens = seq->get_num_cached_tokens();

        if (info.is_prefill) {
            seq->mark_prefill_completed();
        }

        result.running_sequences.push_back(info);
    }

    result.num_free_blocks = allocator_->get_num_free_blocks();
    result.total_blocks = allocator_->get_total_blocks();
    result.num_waiting = waiting_queue_.size();
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
    release_sequence_blocks(*seq);

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
