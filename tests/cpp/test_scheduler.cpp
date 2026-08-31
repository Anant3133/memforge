#include "memforge/scheduler.hpp"
#include <iostream>
#include <cassert>

using namespace memforge;

void test_continuous_scheduling() {
    std::cout << "Running test_continuous_scheduling..." << std::endl;
    auto allocator = std::make_shared<BlockAllocator>(20, 16);
    Scheduler scheduler(allocator, 4);

    // Add 2 requests
    std::vector<TokenId> prompt1 = {1, 2, 3};
    std::vector<TokenId> prompt2 = {4, 5, 6, 7};
    scheduler.add_sequence(101, prompt1, 5);
    scheduler.add_sequence(102, prompt2, 5);

    assert(scheduler.get_num_waiting() == 2);
    assert(scheduler.get_num_running() == 0);

    // Step 1: Should promote both to running
    SchedulerStepResult step1 = scheduler.step();
    assert(step1.running_sequences.size() == 2);
    assert(scheduler.get_num_running() == 2);
    assert(scheduler.get_num_waiting() == 0);

    // Update seq 101 with tokens until finish
    for (int i = 0; i < 5; ++i) {
        scheduler.update_sequence(101, 1000 + i, (i == 4));
    }

    // Step 2: Seq 101 finishes and is cleaned up, seq 102 continues
    SchedulerStepResult step2 = scheduler.step();
    assert(step2.finished_sequence_ids.size() == 1);
    assert(step2.finished_sequence_ids[0] == 101);

    std::cout << "test_continuous_scheduling PASSED." << std::endl;
}

int main() {
    std::cout << "=== MemForge Scheduler Unit Tests ===" << std::endl;
    test_continuous_scheduling();
    std::cout << "All Scheduler tests passed successfully!" << std::endl;
    return 0;
}
