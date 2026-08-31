#include "memforge/sequence.hpp"
#include <cmath>

namespace memforge {

Sequence::Sequence(SequenceId id, std::vector<TokenId> prompt_tokens, size_t max_new_tokens, size_t block_size)
    : id_(id)
    , prompt_tokens_(std::move(prompt_tokens))
    , max_new_tokens_(max_new_tokens)
    , block_size_(block_size)
    , status_(SequenceStatus::WAITING)
    , has_completed_prefill_(false) {
    output_tokens_.reserve(max_new_tokens_);
}

std::vector<TokenId> Sequence::get_all_tokens() const {
    std::vector<TokenId> all_tokens;
    all_tokens.reserve(prompt_tokens_.size() + output_tokens_.size());
    all_tokens.insert(all_tokens.end(), prompt_tokens_.begin(), prompt_tokens_.end());
    all_tokens.insert(all_tokens.end(), output_tokens_.begin(), output_tokens_.end());
    return all_tokens;
}

void Sequence::add_block(BlockId physical_block_id) {
    block_table_.push_back(physical_block_id);
}

void Sequence::clear_blocks() {
    block_table_.clear();
}

size_t Sequence::get_num_required_blocks() const {
    size_t total_tokens = get_total_len();
    if (total_tokens == 0) return 0;
    return (total_tokens + block_size_ - 1) / block_size_;
}

void Sequence::append_token(TokenId token) {
    output_tokens_.push_back(token);
    if (output_tokens_.size() >= max_new_tokens_) {
        status_ = SequenceStatus::FINISHED;
    }
}

bool Sequence::is_finished() const {
    return status_ == SequenceStatus::FINISHED || status_ == SequenceStatus::ABORTED;
}

void Sequence::finish() {
    status_ = SequenceStatus::FINISHED;
}

} // namespace memforge
