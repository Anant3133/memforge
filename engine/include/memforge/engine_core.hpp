#pragma once

#include "memforge/types.hpp"
#include "memforge/block_allocator.hpp"
#include "memforge/scheduler.hpp"
#include <memory>
#include <atomic>

namespace memforge {

class EngineCore {
public:
    explicit EngineCore(const EngineConfig& config = EngineConfig{});
    ~EngineCore() = default;

    SequenceId submit_request(const std::vector<TokenId>& prompt_tokens, size_t max_new_tokens);
    SchedulerStepResult step();
    void update_sequence(SequenceId seq_id, TokenId new_token, bool is_finished);
    bool abort_request(SequenceId seq_id);

    std::vector<TokenId> get_generated_tokens(SequenceId seq_id);
    SequenceStatus get_sequence_status(SequenceId seq_id);

    std::shared_ptr<BlockAllocator> get_allocator() const { return allocator_; }
    std::shared_ptr<Scheduler> get_scheduler() const { return scheduler_; }
    const EngineConfig& get_config() const { return config_; }

private:
    EngineConfig config_;
    std::shared_ptr<BlockAllocator> allocator_;
    std::shared_ptr<Scheduler> scheduler_;
    std::atomic<SequenceId> next_seq_id_{1};
};

} // namespace memforge
