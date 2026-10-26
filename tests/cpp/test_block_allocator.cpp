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

    // free_sequence reports which blocks actually reached ref_count 0, so a
    // caller cannot forget to evict them from the PrefixCache. Both of this
    // sequence's blocks are private, so both come back.
    std::vector<BlockId> freed = allocator.free_sequence(seq);
    assert(freed.size() == 2);
    assert(seq.get_num_allocated_blocks() == 0);
    assert(allocator.get_num_free_blocks() == 10);
    std::cout << "test_sequence_capacity PASSED." << std::endl;
}

void test_lookahead_reservation() {
    std::cout << "Running test_lookahead_reservation..." << std::endl;
    // Speculative decoding writes K draft tokens plus a bonus token before it
    // knows how many survive, so the blocks for all of them have to exist
    // up front: a rejected candidate still had to be written to be rejected.
    BlockAllocator allocator(10, 4);
    Sequence seq(1, {1, 2, 3}, 32, 4);

    // 3 tokens fit in one block, but 3 + 5 lookahead needs two.
    assert(seq.get_num_required_blocks() == 1);
    assert(seq.get_num_required_blocks_with_lookahead(5) == 2);
    // A lookahead that stays inside the current block adds nothing.
    assert(seq.get_num_required_blocks_with_lookahead(1) == 1);

    bool ok = allocator.ensure_capacity_with_lookahead(seq, 5);
    assert(ok);
    assert(seq.get_num_allocated_blocks() == 2);

    // Idempotent: the reservation already covers the next step.
    ok = allocator.ensure_capacity_with_lookahead(seq, 5);
    assert(ok);
    assert(seq.get_num_allocated_blocks() == 2);

    // Tokens landing in reserved space need no further allocation. This is the
    // point of reserving: the write cannot fail part-way through a round.
    for (int i = 0; i < 5; ++i) {
        seq.append_token(50 + i);
    }
    assert(seq.get_total_len() == 8);
    assert(seq.get_num_allocated_blocks() == 2);
    assert(allocator.ensure_capacity(seq));
    assert(seq.get_num_allocated_blocks() == 2);

    allocator.free_sequence(seq);
    assert(allocator.get_num_free_blocks() == 10);
    std::cout << "test_lookahead_reservation PASSED." << std::endl;
}

void test_lookahead_reservation_is_all_or_nothing() {
    std::cout << "Running test_lookahead_reservation_is_all_or_nothing..." << std::endl;
    // A partially reserved sequence is the dangerous state: it would run its
    // draft and then discover there is nowhere to put the last candidate. So a
    // reservation that cannot be met in full must allocate nothing at all.
    BlockAllocator allocator(3, 4);
    Sequence seq(1, {1, 2, 3, 4}, 64, 4);

    assert(allocator.ensure_capacity(seq));
    assert(seq.get_num_allocated_blocks() == 1);
    assert(allocator.get_num_free_blocks() == 2);

    // Needs 4 + 40 tokens = 11 blocks, but only 2 are free.
    bool ok = allocator.ensure_capacity_with_lookahead(seq, 40);
    assert(!ok);
    // Nothing was taken, and nothing was stranded.
    assert(seq.get_num_allocated_blocks() == 1);
    assert(allocator.get_num_free_blocks() == 2);

    // A reservation that does fit still succeeds afterwards, so the failure
    // left the allocator in a usable state rather than a wedged one.
    ok = allocator.ensure_capacity_with_lookahead(seq, 4);
    assert(ok);
    assert(seq.get_num_allocated_blocks() == 2);

    allocator.free_sequence(seq);
    assert(allocator.get_num_free_blocks() == 3);
    std::cout << "test_lookahead_reservation_is_all_or_nothing PASSED." << std::endl;
}

void test_zero_lookahead_matches_plain_capacity() {
    std::cout << "Running test_zero_lookahead_matches_plain_capacity..." << std::endl;
    // Plain decoding passes lookahead 0, and must behave exactly as it did
    // before the feature existed.
    BlockAllocator a(10, 4), b(10, 4);
    Sequence plain(1, {1, 2, 3, 4, 5}, 32, 4);
    Sequence zero(2, {1, 2, 3, 4, 5}, 32, 4);

    assert(a.ensure_capacity(plain));
    assert(b.ensure_capacity_with_lookahead(zero, 0));
    assert(plain.get_num_allocated_blocks() == zero.get_num_allocated_blocks());
    assert(a.get_num_free_blocks() == b.get_num_free_blocks());
    std::cout << "test_zero_lookahead_matches_plain_capacity PASSED." << std::endl;
}

int main() {
    std::cout << "=== MemForge BlockAllocator Unit Tests ===" << std::endl;
    test_basic_allocation();
    test_sequence_capacity();
    test_lookahead_reservation();
    test_lookahead_reservation_is_all_or_nothing();
    test_zero_lookahead_matches_plain_capacity();
    std::cout << "All BlockAllocator tests passed successfully!" << std::endl;
    return 0;
}
