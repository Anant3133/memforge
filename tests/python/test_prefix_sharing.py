"""
Prefix sharing (Phase B): block reuse, copy-on-write, and byte-identical output.

Every test here asserts **identical token ids**, never "similar text". The
failure modes this feature introduces -- a non-absolute `position_ids`, a suffix
written over its own cached prefix, a stale registry entry serving a recycled
block -- all produce fluent, confident, *wrong* output rather than a crash.
Anything weaker than exact equality does not detect them.
"""

from __future__ import annotations

import pytest
import torch

from memforge_ml.runtime.continuous_engine import ContinuousBatchingEngine
from memforge_ml.runtime.prefix_sharing import block_owners, copy_block
from memforge_ml.runtime.sampling import SamplingParams

BLOCK = 16


def make_engine(tiny_model, stub_tokenizer, *, sharing: bool, num_blocks: int = 128,
                max_num_seqs: int = 8, max_model_len: int = 512):
    return ContinuousBatchingEngine(
        tiny_model,
        stub_tokenizer,
        block_size=BLOCK,
        num_blocks=num_blocks,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        enable_prefix_sharing=sharing,
    )


def greedy(max_new_tokens: int = 6) -> SamplingParams:
    # Temperature 0 so the comparison is deterministic and exact.
    return SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0, stop_on_eos=False)


def ids_for(prompt_len: int, salt: int = 0) -> list:
    """Token ids inside the stub tokenizer's vocab, distinct per salt."""
    return [2 + ((i * 7 + salt * 101) % 200) for i in range(prompt_len)]


# ------------------------------------------------------------------ identity

def test_shared_prefix_output_is_byte_identical(tiny_model, stub_tokenizer):
    """Sharing must not change a single generated token."""
    prompt = ids_for(40)
    prompts = [prompt + [50 + i] for i in range(4)]

    off = make_engine(tiny_model, stub_tokenizer, sharing=False)
    on = make_engine(tiny_model, stub_tokenizer, sharing=True)

    want = [r["token_ids"] for r in off.generate(prompts, greedy())]
    got = [r["token_ids"] for r in on.generate(prompts, greedy())]

    assert got == want, "prefix sharing changed the output"
    assert on.prefix_cache is not None
    assert on.telemetry()["cached_prefix_tokens"] > 0, "no prefix was actually reused"


def test_partial_prefill_logits_match_full_prefill(tiny_model, stub_tokenizer):
    """
    The mechanism behind partial prefill, isolated from sampling.

    Prefilling 48 tokens in one pass and prefilling 32 then 16 against the same
    pool must leave the model in the same state, to floating-point exactness.
    This is what makes skipping a cached prefix safe; the end-to-end token
    identity tests above would also catch a break here, but this says *where*.
    """
    from memforge_ml.runtime.paged_cache import PagedCache

    engine = make_engine(tiny_model, stub_tokenizer, sharing=True)
    tokens = ids_for(48)

    full_cache = PagedCache(engine.pool, [0, 1, 2, 3], device=engine.device)
    with torch.no_grad():
        full = tiny_model(
            input_ids=torch.tensor([tokens], dtype=torch.long),
            past_key_values=full_cache,
            use_cache=True,
        )

    split_cache = PagedCache(engine.pool, [4, 5, 6, 7], device=engine.device)
    with torch.no_grad():
        tiny_model(
            input_ids=torch.tensor([tokens[:32]], dtype=torch.long),
            past_key_values=split_cache,
            use_cache=True,
        )
    assert split_cache.position == 32
    with torch.no_grad():
        resumed = tiny_model(
            input_ids=torch.tensor([tokens[32:]], dtype=torch.long),
            past_key_values=split_cache,
            use_cache=True,
        )

    assert split_cache.position == 48
    assert torch.allclose(full.logits[0, -1], resumed.logits[0, -1], atol=1e-5), (
        "resuming a prefill from a seeded position diverged from one-shot prefill"
    )


def test_single_request_is_unaffected(tiny_model, stub_tokenizer):
    """With nothing to share, output and prefill work are unchanged."""
    prompt = ids_for(40)
    off = make_engine(tiny_model, stub_tokenizer, sharing=False)
    on = make_engine(tiny_model, stub_tokenizer, sharing=True)

    a = off.generate([prompt], greedy())[0]
    b = on.generate([prompt], greedy())[0]

    assert a["token_ids"] == b["token_ids"]
    assert on.telemetry()["cached_prefix_tokens"] == 0
    assert off.total_prefill_tokens == on.total_prefill_tokens


def test_divergent_suffix_outputs_match_solo_runs(tiny_model, stub_tokenizer):
    """Sharing a prefix must not leak one sequence's suffix into another's."""
    base = ids_for(48)
    a_prompt = base + ids_for(5, salt=1)
    b_prompt = base + ids_for(5, salt=2)

    solo = make_engine(tiny_model, stub_tokenizer, sharing=False)
    want_a = solo.generate([a_prompt], greedy())[0]["token_ids"]
    solo_b = make_engine(tiny_model, stub_tokenizer, sharing=False)
    want_b = solo_b.generate([b_prompt], greedy())[0]["token_ids"]

    shared = make_engine(tiny_model, stub_tokenizer, sharing=True)
    results = shared.generate([a_prompt, b_prompt], greedy())

    assert results[0]["token_ids"] == want_a
    assert results[1]["token_ids"] == want_b


# ------------------------------------------------------------ memory savings

def test_concurrent_identical_prompts_save_blocks(tiny_model, stub_tokenizer):
    """The Phase B headline: N copies of one prefix cost roughly one copy."""
    prompt = ids_for(8 * BLOCK)  # 128 tokens => 8 full blocks
    prompts = [prompt + [60 + i] for i in range(4)]

    peaks = {}
    for sharing in (False, True):
        engine = make_engine(tiny_model, stub_tokenizer, sharing=sharing)
        ids = [engine.submit(p, greedy(4)) for p in prompts]
        peak = 0
        while engine.has_work():
            engine.step()
            peak = max(peak, engine.allocator.get_num_allocated_blocks())
        peaks[sharing] = peak
        for i in ids:  # every request completed
            assert engine.result_for(i)["finish_reason"] is not None

    assert peaks[True] < peaks[False], (
        f"sharing did not reduce peak blocks: {peaks[True]} vs {peaks[False]}"
    )
    # 4 sequences x 9 blocks unshared vs 8 shared + 4 private tails.
    assert peaks[True] <= peaks[False] * 0.6


def test_sharing_telemetry_is_reported(tiny_model, stub_tokenizer):
    prompt = ids_for(40)
    engine = make_engine(tiny_model, stub_tokenizer, sharing=True)
    ids = [engine.submit(prompt + [70 + i], greedy(4)) for i in range(3)]

    saw_shared = False
    while engine.has_work():
        engine.step()
        if engine.telemetry()["shared_blocks"] > 0:
            saw_shared = True

    assert saw_shared, "no block was ever reported as shared"
    t = engine.telemetry()
    assert t["prefix_sharing_enabled"] is True
    assert t["prefix_hit_rate"] > 0.0
    assert t["memory_saved_mb"] >= 0.0
    assert t["cached_prefix_blocks"] >= 0
    assert len(ids) == 3


def test_disabled_sharing_reports_zeroed_keys(tiny_model, stub_tokenizer):
    """The dashboard binds to a stable shape, so keys exist even when off."""
    engine = make_engine(tiny_model, stub_tokenizer, sharing=False)
    engine.generate([ids_for(40)], greedy(3))
    t = engine.telemetry()

    assert t["prefix_sharing_enabled"] is False
    for key in ("shared_blocks", "blocks_saved", "cached_prefix_blocks",
                "prefix_cache_hits", "prefix_cache_misses"):
        assert t[key] == 0, key
    assert t["memory_saved_mb"] == 0.0
    assert t["prefix_hit_rate"] == 0.0


# ------------------------------------------------------------------ lifetime

def test_all_blocks_reclaimed_and_registry_emptied(tiny_model, stub_tokenizer):
    """No leaks: free + allocated == total, and nothing stays registered."""
    prompt = ids_for(40)
    engine = make_engine(tiny_model, stub_tokenizer, sharing=True)
    total = engine.allocator.get_total_blocks()

    engine.generate([prompt + [80 + i] for i in range(4)], greedy(4))
    # One more step lets the scheduler reap the final finished sequences.
    engine.step()

    assert engine.allocator.get_num_allocated_blocks() == 0
    assert engine.allocator.get_num_free_blocks() == total
    assert engine.prefix_cache.num_cached_blocks() == 0
    assert engine.allocator.get_num_shared_blocks() == 0


def test_abort_releases_shared_blocks(tiny_model, stub_tokenizer):
    prompt = ids_for(40)
    engine = make_engine(tiny_model, stub_tokenizer, sharing=True)
    total = engine.allocator.get_total_blocks()

    a = engine.submit(prompt + [91], greedy(8))
    b = engine.submit(prompt + [92], greedy(8))
    for _ in range(4):
        engine.step()

    engine.abort(a)
    engine.abort(b)
    while engine.has_work():
        engine.step()
    engine.step()

    assert engine.allocator.get_num_allocated_blocks() == 0
    assert engine.allocator.get_num_free_blocks() == total
    assert engine.prefix_cache.num_cached_blocks() == 0


def test_preemption_with_sharing_matches_unpressured_run(tiny_model, stub_tokenizer):
    """Prefix sharing and preempt-by-recompute must compose without drift."""
    prompt = ids_for(40)
    prompts = [prompt + [100 + i] for i in range(3)]

    roomy = make_engine(tiny_model, stub_tokenizer, sharing=True, num_blocks=128)
    want = [r["token_ids"] for r in roomy.generate(prompts, greedy(8))]

    # Small enough to force eviction while sharing is active.
    tight = make_engine(tiny_model, stub_tokenizer, sharing=True, num_blocks=12)
    got = [r["token_ids"] for r in tight.generate(prompts, greedy(8))]

    assert got == want
    assert tight.allocator.get_num_allocated_blocks() >= 0  # no accounting underflow


# ----------------------------------------------------------- copy-on-write

def test_prepare_write_is_noop_for_private_blocks(tiny_model, stub_tokenizer):
    """The common path: decode writes a private tail, so CoW never fires."""
    engine = make_engine(tiny_model, stub_tokenizer, sharing=True)
    engine.generate([ids_for(40) + [110 + i] for i in range(3)], greedy(6))

    assert engine.total_cow_copies == 0, (
        "CoW fired on the hot path; only full blocks should ever be shared"
    )


def test_copy_on_write_copies_then_privatises(tiny_model, stub_tokenizer):
    """
    Force the condition the guard exists for.

    Only full blocks are shared today, so a decode write never lands on a shared
    block and this path is unreachable in normal operation. An unexercised safety
    mechanism is worthless, so the sharing is induced deliberately here.
    """
    import memforge_c

    engine = make_engine(tiny_model, stub_tokenizer, sharing=True)
    seq_id = engine.submit(ids_for(40), greedy(4))
    engine.step()  # admit + prefill, so the blocks hold real KV

    seq = engine.scheduler.get_sequence(seq_id)
    table = list(seq.get_block_table())
    target = table[0]

    # Pretend another sequence hit this block in the prefix cache.
    assert engine.allocator.add_ref(target)
    assert engine.allocator.get_ref_count(target) == 2

    before_k = engine.pool.key_pool[:, target].clone()
    before_v = engine.pool.value_pool[:, target].clone()

    cow = engine.allocator.prepare_write(seq, 0)
    assert cow is not None, "a shared block must trigger copy-on-write"
    old_block, new_block = cow
    assert old_block == target and new_block != target

    # The sequence now points at the private copy; the original is untouched.
    assert seq.get_block_table()[0] == new_block
    assert engine.allocator.get_ref_count(old_block) == 1
    assert engine.allocator.get_ref_count(new_block) == 1

    copy_block(engine.pool, old_block, new_block)
    assert torch.equal(engine.pool.key_pool[:, new_block], before_k)
    assert torch.equal(engine.pool.value_pool[:, new_block], before_v)
    assert torch.equal(engine.pool.key_pool[:, old_block], before_k)


def test_copy_block_is_a_noop_on_itself(tiny_model, stub_tokenizer):
    engine = make_engine(tiny_model, stub_tokenizer, sharing=True)
    engine.pool.key_pool[:, 3].fill_(1.5)
    copy_block(engine.pool, 3, 3)
    assert torch.allclose(engine.pool.key_pool[:, 3], torch.full_like(engine.pool.key_pool[:, 3], 1.5))


# --------------------------------------------------------- hashing contract

def test_hash_is_chained_over_the_whole_prefix():
    """
    The correctness argument for prefix sharing in one assertion.

    Identical block tokens under different predecessors must hash differently,
    because a block's KV is a function of everything before it.
    """
    import memforge_c

    block = list(range(BLOCK))
    assert memforge_c.hash_block(0, block) == memforge_c.hash_block(0, block)
    assert memforge_c.hash_block(0, block) != memforge_c.hash_block(7, block)
    assert memforge_c.hash_block(0, block) != memforge_c.hash_block(0, list(range(1, BLOCK + 1)))
    # 0 is the "unregistered" sentinel and must never be produced.
    assert memforge_c.hash_block(0, block) != 0


def test_stale_registry_entry_is_never_served():
    """
    The hazard that makes the reverse block->hash index load-bearing.

    A finished sequence's block returns to the pool and is handed to an unrelated
    request. If the registry still pointed at it, the next hash hit would serve a
    block holding someone else's KV.
    """
    import memforge_c

    allocator = memforge_c.BlockAllocator(8, BLOCK)
    cache = memforge_c.PrefixCache(allocator, BLOCK)
    tokens = list(range(40))

    table = list(allocator.allocate_blocks(3))
    cache.register_full_blocks(tokens, table)
    assert cache.num_cached_blocks() == 2

    freed = allocator.release(table)
    cache.evict(freed)

    # Those physical blocks now belong to someone else.
    allocator.allocate_blocks(3)
    assert cache.acquire_shared_prefix(tokens) == []
    assert cache.num_cached_blocks() == 0


def test_partial_blocks_are_never_shared():
    """A block still being written to has no meaningful hash yet."""
    import memforge_c

    allocator = memforge_c.BlockAllocator(16, BLOCK)
    cache = memforge_c.PrefixCache(allocator, BLOCK)

    short = list(range(BLOCK - 1))  # not even one full block
    cache.register_full_blocks(short, list(allocator.allocate_blocks(1)))
    assert cache.num_cached_blocks() == 0
    assert cache.acquire_shared_prefix(short) == []


def test_final_block_withheld_on_exact_boundary():
    """
    A prompt that is an exact multiple of the block size must not be fully
    cached, or partial prefill would be handed an empty suffix -- a zero-length
    forward pass.
    """
    import memforge_c

    allocator = memforge_c.BlockAllocator(16, BLOCK)
    cache = memforge_c.PrefixCache(allocator, BLOCK)
    tokens = list(range(3 * BLOCK))

    cache.register_full_blocks(tokens, list(allocator.allocate_blocks(3)))
    assert cache.num_cached_blocks() == 2  # 3 full blocks, last withheld

    shared = cache.acquire_shared_prefix(tokens)
    assert len(shared) == 2 < 3
    cache.release_acquired(shared)


def test_release_acquired_strands_no_references():
    """
    Admission can hit the prefix and then fail to find room for the tail. The
    references must come straight back, or those blocks are pinned forever.
    """
    import memforge_c

    allocator = memforge_c.BlockAllocator(32, BLOCK)
    cache = memforge_c.PrefixCache(allocator, BLOCK)
    tokens = list(range(40))

    table = list(allocator.allocate_blocks(3))
    cache.register_full_blocks(tokens, table)
    baseline = allocator.get_num_allocated_blocks()

    for _ in range(5):
        shared = cache.acquire_shared_prefix(tokens)
        assert len(shared) == 2
        cache.release_acquired(shared)
        assert allocator.get_num_allocated_blocks() == baseline
        assert allocator.get_ref_count(table[0]) == 1

    assert cache.num_cached_blocks() == 2


# ------------------------------------------------------------- block owners

def test_block_owners_exposes_multiple_owners():
    owners = block_owners([(1, [0, 1, 2]), (2, [0, 1, 5])], total_blocks=8)
    assert owners["0"] == [1, 2]
    assert owners["1"] == [1, 2]
    assert owners["2"] == [1]
    assert owners["5"] == [2]
    assert "7" not in owners


def test_block_owners_ignores_out_of_range():
    owners = block_owners([(1, [0, 99, -1])], total_blocks=4)
    assert owners == {"0": [1]}


def test_engine_block_owners_matches_occupancy(tiny_model, stub_tokenizer):
    prompt = ids_for(40)
    engine = make_engine(tiny_model, stub_tokenizer, sharing=True)
    for i in range(3):
        engine.submit(prompt + [120 + i], greedy(6))
    for _ in range(5):
        engine.step()

    owners = engine.block_owners()
    occupancy = engine.block_occupancy()
    assert owners, "expected live blocks"
    for block_str, seq_ids in owners.items():
        assert occupancy[int(block_str)] in seq_ids
    # At least one block is held by more than one sequence.
    assert any(len(v) > 1 for v in owners.values())


# ----------------------------------------------------------- seed_position

def test_seed_position_moves_every_layer(tiny_model, stub_tokenizer):
    engine = make_engine(tiny_model, stub_tokenizer, sharing=True)
    from memforge_ml.runtime.paged_cache import PagedCache

    cache = PagedCache(engine.pool, [0, 1, 2], device=engine.device)
    cache.seed_position(32)
    assert cache.position == 32
    assert all(layer.position == 32 for layer in cache.layers)
    # The causal mask span must follow, or the prefix would be masked out.
    assert cache.layers[0].get_mask_sizes(1)[0] == 33

    with pytest.raises(ValueError):
        cache.seed_position(-1)
