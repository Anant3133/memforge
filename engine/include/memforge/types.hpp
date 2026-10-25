#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace memforge {

using SequenceId = uint64_t;
using TokenId = int32_t;
using BlockId = int32_t;

constexpr BlockId INVALID_BLOCK_ID = -1;
constexpr size_t DEFAULT_BLOCK_SIZE = 16;

/**
 * @brief Represents the execution state of a sequence in the inference engine.
 */
enum class SequenceStatus {
    WAITING,    // Queued, awaiting block allocation and scheduler slot
    RUNNING,    // Active in continuous batching iteration pool
    SWAPPED,    // Reserved. Preemption is by recomputation, not host swapping,
                // so nothing currently sets this; see PROGRESS_AND_NEXT_PHASES.md §4.
    FINISHED,   // Generation reached EOS or max_tokens limit
    ABORTED     // Cancelled by client or errored
};

/**
 * @brief Represents a physical memory block metadata for PagedAttention.
 */
struct PhysicalBlock {
    BlockId block_id{INVALID_BLOCK_ID};
    int ref_count{0};
    bool is_allocated{false};
    /// Chained prefix hash this block is registered under in the PrefixCache,
    /// or 0 when it is not shareable. `hash_block()` never returns 0, so the
    /// sentinel is unambiguous.
    uint64_t block_hash{0};
};

/**
 * @brief Engine execution step metadata returned to Python runtime per iteration.
 */
struct SequenceStepInfo {
    SequenceId seq_id{0};
    std::vector<BlockId> block_table;
    size_t prompt_length{0};
    size_t output_length{0};
    size_t total_length{0};
    SequenceStatus status{SequenceStatus::WAITING};
    bool is_prefill{false}; // True if running initial prompt prefill, false if decoding single token
    /// Leading tokens whose KV is already resident in the pool, so the prefill
    /// forward pass must skip them and start at this offset. Resident either
    /// because a prefix-cache hit supplied them (Phase B) or because an earlier
    /// chunk of this sequence's own prefill wrote them (Phase C). 0 means start
    /// from the beginning.
    size_t num_cached_tokens{0};
    /// Exclusive end of the prompt range this step should prefill, so a long
    /// prompt is spread across steps instead of monopolising one. Equal to
    /// `num_cached_tokens` when no budget was left this step, which means "do
    /// nothing for this sequence now"; equal to `total_length` on the final
    /// chunk, which is the only chunk that produces a token.
    size_t prefill_chunk_end{0};
};

/**
 * @brief Result of an iteration-level scheduler step.
 */
struct SchedulerStepResult {
    std::vector<SequenceStepInfo> running_sequences;
    std::vector<SequenceId> finished_sequence_ids;
    std::vector<SequenceId> preempted_sequence_ids; // Evicted under memory pressure, will be recomputed
    std::vector<SequenceId> aborted_sequence_ids;   // Unschedulable (request larger than the whole pool)
    size_t num_free_blocks{0};
    size_t total_blocks{0};
    size_t num_waiting{0};
    double memory_utilization_ratio{0.0};
};

} // namespace memforge
