#include "memforge/block_allocator.hpp"
#include <iostream>
#include <cassert>

using namespace memforge;

void test_basic_allocation() {
    std::cout << "Running test_basic_allocation..." << std::endl;
    BlockAllocator allocator(10, 16);

    assert(allocator.get_total_blocks() == 10);
    assert(allocator.get_num_free_blocks() == 10);
    assert(allocator.get_num_allocated_blocks() == 0);

    BlockId b0 = allocator.allocate();
    assert(b0 == 0);
    assert(allocator.get_num_free_blocks() == 9);
    assert(allocator.get_num_allocated_blocks() == 1);

    auto blocks = allocator.allocate_blocks(3);
    assert(blocks.size() == 3);
    assert(allocator.get_num_free_blocks() == 6);

    allocator.free(b0);
    assert(allocator.get_num_free_blocks() == 7);

    allocator.free_blocks(blocks);
    assert(allocator.get_num_free_blocks() == 10);
    std::cout << "test_basic_allocation PASSED." << std::endl;
}

void test_sequence_capacity() {
    std::cout << "Running test_sequence_capacity..." << std::endl;
    BlockAllocator allocator(10, 16);
    std::vector<TokenId> prompt = {1, 2, 3, 4, 5};
    Sequence seq(1, prompt, 32, 16);

    assert(seq.get_num_required_blocks() == 1);
    bool ok = allocator.ensure_capacity(seq);
    assert(ok);
    assert(seq.get_num_allocated_blocks() == 1);

    // Append 12 tokens to cross boundary (5 + 12 = 17 tokens -> requires 2 blocks)
    for (int i = 0; i < 12; ++i) {
        seq.append_token(100 + i);
    }
    assert(seq.get_total_len() == 17);
    assert(seq.get_num_required_blocks() == 2);

    ok = allocator.ensure_capacity(seq);
    assert(ok);
    assert(seq.get_num_allocated_blocks() == 2);

    allocator.free_sequence(seq);
    assert(allocator.get_num_free_blocks() == 10);
    std::cout << "test_sequence_capacity PASSED." << std::endl;
}

int main() {
    std::cout << "=== MemForge BlockAllocator Unit Tests ===" << std::endl;
    test_basic_allocation();
    test_sequence_capacity();
    std::cout << "All BlockAllocator tests passed successfully!" << std::endl;
    return 0;
}
