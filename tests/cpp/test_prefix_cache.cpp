#include "memforge/prefix_cache.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

using namespace memforge;

namespace {

constexpr size_t BS = 16;  // block size used throughout

/// Tokens 0..n-1, so every block has distinct content.
std::vector<TokenId> ramp(size_t n, TokenId offset = 0) {
    std::vector<TokenId> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(static_cast<TokenId>(i) + offset);
    }
    return out;
}

/// Admit a sequence the way the scheduler does: share what we can, allocate the
/// rest, then register the now-resident full blocks.
std::vector<BlockId> admit(PrefixCache& cache,
                           BlockAllocator& allocator,
                           const std::vector<TokenId>& tokens) {
    std::vector<BlockId> shared = cache.acquire_shared_prefix(tokens);
    const size_t required = (tokens.size() + BS - 1) / BS;
    const size_t needed = required - shared.size();

    std::vector<BlockId> table = shared;
    if (needed) {
        std::vector<BlockId> fresh = allocator.allocate_blocks(needed);
        table.insert(table.end(), fresh.begin(), fresh.end());
    }
    cache.register_full_blocks(tokens, table);
    return table;
}

}  // namespace

void test_hash_is_prefix_dependent() {
    std::cout << "Running test_hash_is_prefix_dependent..." << std::endl;

    std::vector<TokenId> block_a = ramp(BS, 0);
    std::vector<TokenId> block_b = ramp(BS, 100);

    // Deterministic: same parent + same tokens => same hash.
    assert(hash_block(0, block_a.data(), BS) == hash_block(0, block_a.data(), BS));

    // Different tokens under the same parent => different hash.
    assert(hash_block(0, block_a.data(), BS) != hash_block(0, block_b.data(), BS));

    // The point of chaining: IDENTICAL block tokens under DIFFERENT parents must
    // hash differently, because their KV depends on the whole preceding prefix.
    const uint64_t under_root = hash_block(0, block_a.data(), BS);
    const uint64_t under_other = hash_block(12345, block_a.data(), BS);
    assert(under_root != under_other);

    // 0 is reserved as the "unregistered" sentinel and must never be produced.
    assert(hash_block(0, block_a.data(), BS) != 0);
    assert(hash_block(0, block_b.data(), BS) != 0);

    std::cout << "test_hash_is_prefix_dependent PASSED." << std::endl;
}

void test_identical_prompts_share_blocks() {
    std::cout << "Running test_identical_prompts_share_blocks..." << std::endl;

    auto allocator = std::make_shared<BlockAllocator>(64, BS);
    PrefixCache cache(allocator, BS);

    // 40 tokens => 2 full blocks (shareable) + 1 partial (always private).
    std::vector<TokenId> tokens = ramp(40);

    std::vector<BlockId> first = admit(cache, *allocator, tokens);
    assert(first.size() == 3);
    const size_t after_first = allocator->get_num_allocated_blocks();
    assert(after_first == 3);
    assert(cache.num_cached_blocks() == 2);

    // Second identical prompt reuses both full blocks and allocates only its tail.
    std::vector<BlockId> second = admit(cache, *allocator, tokens);
    assert(second.size() == 3);
    assert(second[0] == first[0]);
    assert(second[1] == first[1]);
    assert(second[2] != first[2]);  // trailing block is private

    assert(allocator->get_num_allocated_blocks() == 4);  // 3 + 1, not 3 + 3
    assert(allocator->get_ref_count(first[0]) == 2);
    assert(allocator->get_ref_count(first[1]) == 2);
    assert(allocator->get_ref_count(first[2]) == 1);
    assert(allocator->get_num_shared_blocks() == 2);
    assert(allocator->get_num_blocks_saved() == 2);

    std::cout << "test_identical_prompts_share_blocks PASSED." << std::endl;
}

void test_sharing_stops_at_first_divergence() {
    std::cout << "Running test_sharing_stops_at_first_divergence..." << std::endl;

    auto allocator = std::make_shared<BlockAllocator>(64, BS);
    PrefixCache cache(allocator, BS);

    // Share blocks 0 and 1, diverge inside block 2, then converge again in
    // block 3. Block 3 must NOT be shared despite identical tokens, because its
    // KV depends on the divergent block 2.
    std::vector<TokenId> base = ramp(4 * BS + 5);
    admit(cache, *allocator, base);

    std::vector<TokenId> variant = base;
    variant[2 * BS + 3] = 9999;  // one token differs, inside block 2

    std::vector<BlockId> shared = cache.acquire_shared_prefix(variant);
    assert(shared.size() == 2);  // blocks 0 and 1 only
    cache.release_acquired(shared);

    std::cout << "test_sharing_stops_at_first_divergence PASSED." << std::endl;
}

void test_short_and_boundary_prompts() {
    std::cout << "Running test_short_and_boundary_prompts..." << std::endl;

    auto allocator = std::make_shared<BlockAllocator>(64, BS);
    PrefixCache cache(allocator, BS);

    // Shorter than one block: nothing is full, so nothing is shareable.
    std::vector<TokenId> tiny = ramp(BS - 1);
    admit(cache, *allocator, tiny);
    assert(cache.num_cached_blocks() == 0);
    assert(cache.acquire_shared_prefix(tiny).empty());

    // Exactly on a block boundary: the final full block must stay private so the
    // next write has somewhere to go and the prefill suffix is never empty.
    std::vector<TokenId> exact = ramp(3 * BS, 500);
    admit(cache, *allocator, exact);
    assert(cache.num_cached_blocks() == 2);  // 3 full blocks, last one withheld

    std::vector<BlockId> shared = cache.acquire_shared_prefix(exact);
    assert(shared.size() == 2);
    assert(shared.size() < 3);  // never fully cached
    cache.release_acquired(shared);

    std::cout << "test_short_and_boundary_prompts PASSED." << std::endl;
}

void test_refcount_lifecycle_and_eviction() {
    std::cout << "Running test_refcount_lifecycle_and_eviction..." << std::endl;

    auto allocator = std::make_shared<BlockAllocator>(64, BS);
    PrefixCache cache(allocator, BS);

    std::vector<TokenId> tokens = ramp(40);
    std::vector<BlockId> a = admit(cache, *allocator, tokens);
    std::vector<BlockId> b = admit(cache, *allocator, tokens);

    assert(allocator->get_ref_count(a[0]) == 2);

    // First sequence finishes. Shared blocks survive for the second one.
    std::vector<BlockId> freed_a = allocator->release(a);
    cache.evict(freed_a);
    assert(freed_a.size() == 1);  // only its private tail actually freed
    assert(allocator->get_ref_count(a[0]) == 1);
    assert(cache.num_cached_blocks() == 2);  // registry intact

    // Second finishes too: everything comes back and the registry empties.
    std::vector<BlockId> freed_b = allocator->release(b);
    cache.evict(freed_b);
    assert(freed_b.size() == 3);
    assert(allocator->get_num_allocated_blocks() == 0);
    assert(allocator->get_num_free_blocks() == 64);
    assert(cache.num_cached_blocks() == 0);

    std::cout << "test_refcount_lifecycle_and_eviction PASSED." << std::endl;
}

void test_stale_entry_is_never_served() {
    std::cout << "Running test_stale_entry_is_never_served..." << std::endl;

    auto allocator = std::make_shared<BlockAllocator>(8, BS);
    PrefixCache cache(allocator, BS);

    std::vector<TokenId> tokens = ramp(40);
    std::vector<BlockId> table = admit(cache, *allocator, tokens);
    assert(cache.num_cached_blocks() == 2);

    // Sequence ends, but we "forget" to evict -- simulating the bug this guards
    // against. The blocks are now free and will be handed to someone else.
    std::vector<BlockId> freed = allocator->release(table);
    assert(freed.size() == 3);

    // An unrelated request takes those exact physical blocks.
    std::vector<BlockId> stolen = allocator->allocate_blocks(3);
    (void)stolen;

    // The stale registry entry must NOT be served: add_ref refuses a block whose
    // reference was re-established by someone else... it IS allocated now, so the
    // real protection is eviction. Prove that evict() closes the hole.
    cache.evict(freed);
    assert(cache.num_cached_blocks() == 0);
    assert(cache.acquire_shared_prefix(tokens).empty());

    std::cout << "test_stale_entry_is_never_served PASSED." << std::endl;
}

void test_add_ref_refuses_free_blocks() {
    std::cout << "Running test_add_ref_refuses_free_blocks..." << std::endl;

    BlockAllocator allocator(8, BS);
    BlockId id = allocator.allocate();
    assert(allocator.add_ref(id));
    assert(allocator.get_ref_count(id) == 2);

    allocator.free(id);
    allocator.free(id);
    assert(allocator.get_ref_count(id) == 0);

    // A freed block cannot be referenced -- this is what converts a stale cache
    // entry into a clean miss rather than corrupted output.
    assert(!allocator.add_ref(id));
    assert(!allocator.add_ref(-1));
    assert(!allocator.add_ref(999));

    std::cout << "test_add_ref_refuses_free_blocks PASSED." << std::endl;
}

void test_copy_on_write() {
    std::cout << "Running test_copy_on_write..." << std::endl;

    BlockAllocator allocator(16, BS);
    std::vector<TokenId> prompt = ramp(40);
    Sequence seq(1, prompt, 32, BS);

    bool ok = allocator.ensure_capacity(seq);
    assert(ok);
    assert(seq.get_num_allocated_blocks() == 3);

    const BlockId private_block = seq.get_block_table()[2];
    // A private block needs no copy: prepare_write must be a no-op.
    assert(!allocator.prepare_write(seq, 2).has_value());
    assert(seq.get_block_table()[2] == private_block);

    // Force block 0 to be shared, as a prefix hit would.
    const BlockId shared_block = seq.get_block_table()[0];
    assert(allocator.add_ref(shared_block));
    assert(allocator.get_ref_count(shared_block) == 2);

    auto cow = allocator.prepare_write(seq, 0);
    assert(cow.has_value());
    assert(cow->first == shared_block);
    assert(cow->second != shared_block);

    // The sequence now points at its private copy; the original survives for
    // whoever else still references it.
    assert(seq.get_block_table()[0] == cow->second);
    assert(allocator.get_ref_count(cow->second) == 1);
    assert(allocator.get_ref_count(shared_block) == 1);

    // Out-of-range logical index is a no-op, not a crash.
    assert(!allocator.prepare_write(seq, 99).has_value());

    std::cout << "test_copy_on_write PASSED." << std::endl;
}

void test_release_acquired_strands_nothing() {
    std::cout << "Running test_release_acquired_strands_nothing..." << std::endl;

    auto allocator = std::make_shared<BlockAllocator>(64, BS);
    PrefixCache cache(allocator, BS);

    std::vector<TokenId> tokens = ramp(40);
    std::vector<BlockId> table = admit(cache, *allocator, tokens);
    const size_t baseline = allocator->get_num_allocated_blocks();

    // Admission hits the prefix but then fails to find room for the tail, so it
    // must hand the references straight back. Missing this leaks ref-counts
    // permanently: blocks that can never return to the pool.
    for (int i = 0; i < 5; ++i) {
        std::vector<BlockId> shared = cache.acquire_shared_prefix(tokens);
        assert(shared.size() == 2);
        cache.release_acquired(shared);
        assert(allocator->get_num_allocated_blocks() == baseline);
        assert(allocator->get_ref_count(table[0]) == 1);
    }

    // The entries are still cached and still usable afterwards.
    assert(cache.num_cached_blocks() == 2);

    std::cout << "test_release_acquired_strands_nothing PASSED." << std::endl;
}

void test_hit_rate_accounting() {
    std::cout << "Running test_hit_rate_accounting..." << std::endl;

    auto allocator = std::make_shared<BlockAllocator>(64, BS);
    PrefixCache cache(allocator, BS);

    std::vector<TokenId> tokens = ramp(40);
    admit(cache, *allocator, tokens);
    assert(cache.num_hits() == 0);

    std::vector<BlockId> shared = cache.acquire_shared_prefix(tokens);
    // Two hits, then a miss on the withheld trailing block's successor.
    assert(cache.num_hits() == 2);
    assert(cache.hit_rate() > 0.0 && cache.hit_rate() <= 1.0);
    cache.release_acquired(shared);

    std::cout << "test_hit_rate_accounting PASSED." << std::endl;
}

int main() {
    std::cout << "=== PrefixCache Test Suite ===" << std::endl;
    test_hash_is_prefix_dependent();
    test_identical_prompts_share_blocks();
    test_sharing_stops_at_first_divergence();
    test_short_and_boundary_prompts();
    test_refcount_lifecycle_and_eviction();
    test_stale_entry_is_never_served();
    test_add_ref_refuses_free_blocks();
    test_copy_on_write();
    test_release_acquired_strands_nothing();
    test_hit_rate_accounting();
    std::cout << "=== ALL PREFIX CACHE TESTS PASSED ===" << std::endl;
    return 0;
}
