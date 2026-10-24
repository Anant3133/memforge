#pragma once

#include "memforge/types.hpp"
#include <vector>
#include <string>
#include <memory>

namespace memforge {

class Sequence {
public:
    Sequence(SequenceId id, std::vector<TokenId> prompt_tokens, size_t max_new_tokens, size_t block_size = DEFAULT_BLOCK_SIZE);

    SequenceId get_id() const { return id_; }
    SequenceStatus get_status() const { return status_; }
    void set_status(SequenceStatus status) { status_ = status; }

    const std::vector<TokenId>& get_prompt_tokens() const { return prompt_tokens_; }
    const std::vector<TokenId>& get_output_tokens() const { return output_tokens_; }
    std::vector<TokenId> get_all_tokens() const;

    size_t get_prompt_len() const { return prompt_tokens_.size(); }
    size_t get_output_len() const { return output_tokens_.size(); }
    size_t get_total_len() const { return prompt_tokens_.size() + output_tokens_.size(); }
    size_t get_max_new_tokens() const { return max_new_tokens_; }
    size_t get_block_size() const { return block_size_; }

    // Logical-to-Physical Block Table
    const std::vector<BlockId>& get_block_table() const { return block_table_; }
    void add_block(BlockId physical_block_id);
    /// Repoint one logical block at a different physical block. Used by
    /// copy-on-write, which swaps a shared block for a private copy.
    void set_block(size_t logical_index, BlockId physical_block_id);
    void clear_blocks();
    size_t get_num_required_blocks() const;
    size_t get_num_allocated_blocks() const { return block_table_.size(); }

    // Token management
    void append_token(TokenId token);
    bool is_finished() const;
    void finish();

    // Prefill tracking
    bool has_completed_prefill() const { return has_completed_prefill_; }
    void mark_prefill_completed() { has_completed_prefill_ = true; }
    /// Reset for preempt-by-recompute. Also clears the cached-prefix count: a
    /// preempted sequence released its blocks, so nothing of it is resident any
    /// more and the next admission must re-run the prefix lookup from scratch.
    void reset_prefill() { has_completed_prefill_ = false; num_cached_tokens_ = 0; }

    // Prefix sharing: leading tokens whose KV came from a prefix-cache hit and
    // must therefore NOT be recomputed (or rewritten) by the prefill pass.
    size_t get_num_cached_tokens() const { return num_cached_tokens_; }
    void set_num_cached_tokens(size_t n) { num_cached_tokens_ = n; }

private:
    SequenceId id_;
    std::vector<TokenId> prompt_tokens_;
    std::vector<TokenId> output_tokens_;
    std::vector<BlockId> block_table_;
    size_t max_new_tokens_{128};
    size_t block_size_{DEFAULT_BLOCK_SIZE};
    SequenceStatus status_{SequenceStatus::WAITING};
    bool has_completed_prefill_{false};
    size_t num_cached_tokens_{0};
};

using SequencePtr = std::shared_ptr<Sequence>;

} // namespace memforge
