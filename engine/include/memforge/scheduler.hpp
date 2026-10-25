#pragma once

#include "memforge/types.hpp"
#include "memforge/sequence.hpp"
#include "memforge/block_allocator.hpp"
#include "memforge/prefix_cache.hpp"
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
    /**
     * @param prefix_cache Optional. When null, admission allocates every block
     *        fresh -- exactly the pre-prefix-sharing behaviour. Passing a cache
     *        makes identical prompt prefixes share physical blocks. Kept
     *        optional so the benchmark can measure both paths in one process and
     *        so sharing can be switched off if it ever misbehaves in the field.
     */
    Scheduler(std::shared_ptr<BlockAllocator> allocator, size_t max_num_seqs = 256,
              std::shared_ptr<PrefixCache> prefix_cache = nullptr);
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
    bool has_unfinished_sequences() const;

    SequencePtr get_sequence(SequenceId seq_id);
    std::shared_ptr<BlockAllocator> get_allocator() const { return allocator_; }
    std::shared_ptr<PrefixCache> get_prefix_cache() const { return prefix_cache_; }
    bool prefix_sharing_enabled() const { return prefix_cache_ != nullptr; }

private:
    /**
     * @brief Release a sequence's blocks and keep the prefix registry honest.
     *
     * Caller must hold `mutex_`. Lock order throughout this class is
     * Scheduler -> PrefixCache -> BlockAllocator; nothing below ever reaches
     * back up, so this is safe to call under the scheduler lock.
     */
    void release_sequence_blocks(Sequence& seq);

    std::shared_ptr<BlockAllocator> allocator_;
    std::shared_ptr<PrefixCache> prefix_cache_;  // null => sharing disabled
    size_t max_num_seqs_;

    std::deque<SequencePtr> waiting_queue_;
    std::vector<SequencePtr> running_queue_;
    std::unordered_map<SequenceId, SequencePtr> all_sequences_;

    mutable std::mutex mutex_;
};

} // namespace memforge
