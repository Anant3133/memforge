/**
 * Sustained-load driver for the C++ core, built to be run under a sanitizer.
 *
 * `benchmarks/run_stress_test.py` already covers sustained load at the *Python*
 * level and reports flat RSS and zero leaked blocks. What it cannot see is
 * memory misuse inside the C++ allocator and prefix cache: a use-after-free on
 * a recycled block, a read past the end of a block table, or heap blocks the
 * allocator's own bookkeeping never returns. Those are invisible from Python
 * because they corrupt silently -- the symptom is wrong KV, which reads as
 * fluent, confident, wrong text rather than a crash.
 *
 * So this driver hammers the subsystems in the pattern a real server produces
 * -- admit, prefill, share a prefix, copy on write, decode, preempt, finish,
 * recycle -- and leaves the diagnosis to the tool it is built with:
 *
 *   * `/fsanitize=address` catches use-after-free, buffer overflow and
 *     double-free. On Windows it does **not** catch leaks; MSVC's ASan ships no
 *     LeakSanitizer, which `scripts/cpp_asan.py` verifies rather than assumes.
 *   * `_CRTDBG_LEAK_CHECK_DF` (debug CRT, `MEMFORGE_CRT_LEAK_CHECK`) covers that
 *     other half and reports unfreed heap allocations at exit.
 *
 * The driver also asserts the *logical* invariant on every cycle, which no
 * sanitizer can check: every block handed out must come back, so the free list
 * must return to its starting size. A logical block leak and a heap leak are
 * different failures and both matter.
 */

#include "memforge/block_allocator.hpp"
#include "memforge/prefix_cache.hpp"
#include "memforge/scheduler.hpp"
#include "memforge/sequence.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#ifdef MEMFORGE_CRT_LEAK_CHECK
#include <crtdbg.h>
#endif

using namespace memforge;

namespace {

constexpr size_t kBlockSize = 16;
constexpr size_t kTotalBlocks = 256;

std::vector<TokenId> make_prompt(std::mt19937& rng, size_t length, int shared_prefix) {
    std::vector<TokenId> tokens;
    tokens.reserve(length);
    // A deterministic shared head, so prefix sharing actually engages, followed
    // by a random tail so sharing stops at a realistic divergence point rather
    // than always at a block boundary.
    for (int i = 0; i < shared_prefix; ++i) {
        tokens.push_back(static_cast<TokenId>(1000 + i));
    }
    std::uniform_int_distribution<int> dist(1, 30000);
    while (tokens.size() < length) {
        tokens.push_back(static_cast<TokenId>(dist(rng)));
    }
    return tokens;
}

/**
 * One full request lifecycle, driven through the scheduler the way the engine
 * drives it. Returns the number of tokens generated.
 */
size_t run_cycle(const std::shared_ptr<BlockAllocator>& allocator,
                 const std::shared_ptr<PrefixCache>& prefix_cache,
                 Scheduler& scheduler,
                 std::mt19937& rng,
                 size_t num_requests,
                 SequenceId& next_seq_id) {
    std::uniform_int_distribution<int> prompt_len(8, 96);
    std::uniform_int_distribution<int> new_tokens(4, 24);
    std::uniform_int_distribution<int> shared(0, 48);

    std::vector<SequenceId> ids;
    ids.reserve(num_requests);
    for (size_t i = 0; i < num_requests; ++i) {
        const size_t length = static_cast<size_t>(prompt_len(rng));
        std::vector<TokenId> tokens = make_prompt(rng, length, shared(rng));
        const SequenceId id = next_seq_id++;
        scheduler.add_sequence(id, tokens, static_cast<size_t>(new_tokens(rng)));
        ids.push_back(id);
    }

    size_t generated = 0;
    // Bounded: a scheduler that stops retiring sequences must fail the test
    // rather than spin here forever.
    const size_t max_steps = 20000;
    for (size_t step = 0; step < max_steps; ++step) {
        SchedulerStepResult out = scheduler.step();
        if (out.running_sequences.empty() && out.num_waiting == 0) {
            break;
        }

        for (const SequenceStepInfo& info : out.running_sequences) {
            // Touch the block table the way a forward pass would: every entry
            // has to be a live, allocated block. Reading a recycled block here
            // is exactly the use-after-free ASan is being asked about.
            for (BlockId block : info.block_table) {
                assert(block != INVALID_BLOCK_ID);
                const int refs = allocator->get_ref_count(block);
                assert(refs > 0 && "block table points at a free block");
                (void)refs;
            }

            if (info.is_prefill) {
                SequencePtr seq = scheduler.get_sequence(info.seq_id);
                if (seq && !info.block_table.empty()) {
                    // Exercise copy-on-write on the block a write would land
                    // in. Done before registration, which is the order the
                    // engine uses: a sequence must privatise its write target
                    // before advertising anything.
                    const size_t logical = info.block_table.size() - 1;
                    auto cow = allocator->prepare_write(*seq, logical);
                    if (cow.has_value()) {
                        assert(allocator->get_ref_count(cow->second) == 1);
                    }

                    // Advertise the prefilled blocks for reuse, exactly as
                    // ContinuousBatchingEngine._run_prefill does. Without this
                    // nothing is ever shared and the sharing paths -- the most
                    // memory-unsafe part of the C++ -- go unexercised.
                    prefix_cache->register_full_blocks(
                        seq->get_all_tokens(),
                        std::vector<BlockId>(seq->get_block_table()));
                }
                continue;
            }

            scheduler.update_sequence(info.seq_id, static_cast<TokenId>(7), false);
            ++generated;
        }
    }

    // Abort whatever is still alive, so each cycle ends with the pool empty and
    // a leak cannot hide behind a sequence that simply never finished.
    for (SequenceId id : ids) {
        scheduler.abort_sequence(id);
    }
    scheduler.step();
    return generated;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef MEMFORGE_CRT_LEAK_CHECK
    // Report unfreed heap allocations at exit, with the allocation that leaked.
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
#endif

    const size_t cycles = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 40;
    const size_t requests_per_cycle = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 12;

    std::printf("sustained load: %zu cycles x %zu requests, %zu blocks of %zu\n",
                cycles, requests_per_cycle, kTotalBlocks, kBlockSize);

    auto allocator = std::make_shared<BlockAllocator>(kTotalBlocks, kBlockSize);
    auto prefix_cache = std::make_shared<PrefixCache>(allocator, kBlockSize);
    Scheduler scheduler(allocator, 8, prefix_cache, 0, 0);

    const size_t free_at_start = allocator->get_num_free_blocks();
    assert(free_at_start == kTotalBlocks);

    std::mt19937 rng(12345);
    size_t total_generated = 0;
    SequenceId next_seq_id = 1;

    for (size_t cycle = 0; cycle < cycles; ++cycle) {
        total_generated += run_cycle(allocator, prefix_cache, scheduler, rng,
                                     requests_per_cycle, next_seq_id);

        // The logical invariant, checked every cycle rather than once at the
        // end: a leak that recurs per cycle and one that happens once look the
        // same in a final total, and they are very different bugs.
        const size_t free_now = allocator->get_num_free_blocks();
        if (free_now != free_at_start) {
            std::fprintf(stderr,
                         "FAIL: cycle %zu leaked %zu blocks (%zu free, expected %zu)\n",
                         cycle, free_at_start - free_now, free_now, free_at_start);
            return 1;
        }
    }

    // Half the cycles again, but dropping the prefix cache partway through, so
    // eviction and the allocator's reverse index are exercised while sequences
    // still hold references.
    for (size_t cycle = 0; cycle < cycles / 2; ++cycle) {
        total_generated += run_cycle(allocator, prefix_cache, scheduler, rng,
                                     requests_per_cycle, next_seq_id);
        if (cycle % 4 == 0) {
            prefix_cache->clear();
        }
        const size_t free_now = allocator->get_num_free_blocks();
        if (free_now != free_at_start) {
            std::fprintf(stderr,
                         "FAIL: post-clear cycle %zu leaked %zu blocks\n",
                         cycle, free_at_start - free_now);
            return 1;
        }
    }

    std::printf("generated %zu tokens; free blocks %zu/%zu\n",
                total_generated, allocator->get_num_free_blocks(), kTotalBlocks);
    std::printf("prefix cache: %zu hits, %zu misses, hit rate %.3f\n",
                prefix_cache->num_hits(), prefix_cache->num_misses(),
                prefix_cache->hit_rate());

    // Non-vacuity. A clean sanitizer result over code that never ran is not
    // evidence, and a hit rate of zero means every sharing path -- ref counts
    // above one, copy-on-write, eviction of a shared block -- was skipped. An
    // earlier version of this driver reported exactly that while the runner
    // claimed sharing had been exercised.
    if (prefix_cache->num_hits() == 0) {
        std::fprintf(stderr,
                     "FAIL: prefix cache served no hits, so the sharing paths "
                     "were never exercised and a clean result proves nothing "
                     "about them\n");
        return 1;
    }
    if (allocator->get_num_blocks_saved() != 0) {
        std::fprintf(stderr, "FAIL: %zu blocks still shared at exit\n",
                     allocator->get_num_blocks_saved());
        return 1;
    }

    std::printf("no logical block leak across %zu cycles\n", cycles + cycles / 2);
    std::printf("SUSTAINED LOAD PASSED\n");
    return 0;
}
