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
    /**
     * @param max_prefill_tokens_per_step Chunked prefill budget (Phase C).
     *        0 means unlimited -- a prompt is prefilled in one step, which is
     *        the original behaviour. Any positive value caps how many prompt
     *        tokens are processed per step, so a 2,000-token prompt arriving
     *        cannot stall the decode steps of requests already in flight.
     *        Decode is never charged against this budget: a running sequence
     *        emits its token every step regardless, which is the whole point.
     */
    Scheduler(std::shared_ptr<BlockAllocator> allocator, size_t max_num_seqs = 256,
              std::shared_ptr<PrefixCache> prefix_cache = nullptr,
              size_t max_prefill_tokens_per_step = 0);
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
    size_t get_max_prefill_tokens_per_step() const { return max_prefill_tokens_per_step_; }
    void set_max_prefill_tokens_per_step(size_t n) { max_prefill_tokens_per_step_ = n; }
    bool chunked_prefill_enabled() const { return max_prefill_tokens_per_step_ > 0; }

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
    size_t max_prefill_tokens_per_step_{0};  // 0 => unlimited

    std::deque<SequencePtr> waiting_queue_;
    std::vector<SequencePtr> running_queue_;
    std::unordered_map<SequenceId, SequencePtr> all_sequences_;

    mutable std::mutex mutex_;
};

} // namespace memforge
