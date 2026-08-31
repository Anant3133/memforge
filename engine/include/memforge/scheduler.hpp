#pragma once

#include "memforge/types.hpp"
#include "memforge/sequence.hpp"
#include "memforge/block_allocator.hpp"
#include <vector>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace memforge {

/**
 * @brief Continuous Batching Scheduler.
 * Dynamically schedules LLM generation requests step-by-step, eliminating
 * static batching bubbles and dynamically injecting/evicting sequences.
 */
class Scheduler {
public:
    Scheduler(std::shared_ptr<BlockAllocator> allocator, size_t max_num_seqs = 256);
    ~Scheduler() = default;

    /**
     * @brief Adds a new incoming prompt request to the waiting queue.
     */
    SequencePtr add_sequence(SequenceId id, const std::vector<TokenId>& prompt_tokens, size_t max_new_tokens);

    /**
     * @brief Executes one iteration-level scheduling cycle.
     * Selects sequences to execute, allocates required blocks, and produces step metadata.
     */
    SchedulerStepResult step();

    /**
     * @brief Appends generated tokens to active sequences and marks completed sequences.
     */
    void update_sequence(SequenceId seq_id, TokenId new_token, bool is_finished);

    /**
     * @brief Force aborts a sequence and frees its allocated resources.
     */
    bool abort_sequence(SequenceId seq_id);

    // Queue query methods
    size_t get_num_waiting() const;
    size_t get_num_running() const;
    size_t get_num_finished() const;
    bool has_unfinished_sequences() const;

    SequencePtr get_sequence(SequenceId seq_id);
    std::shared_ptr<BlockAllocator> get_allocator() const { return allocator_; }

private:
    std::shared_ptr<BlockAllocator> allocator_;
    size_t max_num_seqs_;

    std::deque<SequencePtr> waiting_queue_;
    std::vector<SequencePtr> running_queue_;
    std::unordered_map<SequenceId, SequencePtr> all_sequences_;

    mutable std::mutex mutex_;
};

} // namespace memforge
