"""
Tests for the native C++ engine through its pybind11 bindings.

These exercise the allocator and the continuous-batching scheduler directly,
without any model, so failures point squarely at the C++ layer.
"""

import pytest

memforge_c = pytest.importorskip(
    "memforge_c", reason="native memforge_c extension is not built"
)


# ------------------------------------------------------------------- allocator

def test_allocator_starts_fully_free():
    allocator = memforge_c.BlockAllocator(64, 16)

    assert allocator.get_total_blocks() == 64
    assert allocator.get_num_free_blocks() == 64
    assert allocator.get_num_allocated_blocks() == 0
    assert allocator.get_block_size() == 16
    assert allocator.get_utilization_ratio() == 0.0


def test_allocate_and_free_round_trip():
    allocator = memforge_c.BlockAllocator(32, 16)

    blocks = allocator.allocate_blocks(5)

    assert len(blocks) == 5
    assert len(set(blocks)) == 5, "the allocator handed out a duplicate block"
    assert allocator.get_num_free_blocks() == 27

    allocator.free_blocks(blocks)
    assert allocator.get_num_free_blocks() == 32


def test_allocator_refuses_to_oversubscribe():
    allocator = memforge_c.BlockAllocator(4, 16)

    assert allocator.can_allocate(4) is True
    assert allocator.can_allocate(5) is False
    with pytest.raises(Exception):
        allocator.allocate_blocks(5)
    # The failed request must not have consumed anything.
    assert allocator.get_num_free_blocks() == 4


def test_blocks_are_never_handed_out_twice():
    allocator = memforge_c.BlockAllocator(16, 16)

    first = allocator.allocate_blocks(8)
    second = allocator.allocate_blocks(8)

    assert not set(first) & set(second)
    assert allocator.get_num_free_blocks() == 0


# ------------------------------------------------------------------- scheduler

def test_scheduler_admits_waiting_sequences_and_allocates_blocks():
    allocator = memforge_c.BlockAllocator(64, 16)
    scheduler = memforge_c.Scheduler(allocator, 8)
    scheduler.add_sequence(1, list(range(32)), 16)

    assert scheduler.get_num_waiting() == 1

    result = scheduler.step()

    assert len(result.running_sequences) == 1
    info = result.running_sequences[0]
    assert info.seq_id == 1
    assert info.is_prefill is True
    assert len(info.block_table) == 2  # 32 prompt tokens / 16 per block
    assert result.num_waiting == 0


def test_prefill_flag_is_only_set_on_the_first_step():
    allocator = memforge_c.BlockAllocator(64, 16)
    scheduler = memforge_c.Scheduler(allocator, 8)
    scheduler.add_sequence(1, [1, 2, 3], 16)

    first = scheduler.step()
    scheduler.update_sequence(1, 99, False)
    second = scheduler.step()

    assert first.running_sequences[0].is_prefill is True
    assert second.running_sequences[0].is_prefill is False


def test_block_table_grows_one_block_at_a_time():
    """Paging means a sequence never needs a contiguous up-front reservation."""
    allocator = memforge_c.BlockAllocator(64, 4)
    scheduler = memforge_c.Scheduler(allocator, 4)
    scheduler.add_sequence(1, [1, 2, 3, 4], 32)  # exactly one block of prompt

    result = scheduler.step()
    assert len(result.running_sequences[0].block_table) == 1

    sizes = []
    for _ in range(9):
        scheduler.update_sequence(1, 7, False)
        result = scheduler.step()
        sizes.append(len(result.running_sequences[0].block_table))

    # 4 tokens per block: the table grows by one block every 4 tokens, never more.
    assert sizes == [2, 2, 2, 2, 3, 3, 3, 3, 4]


def test_finished_sequences_are_reported_and_their_blocks_returned():
    allocator = memforge_c.BlockAllocator(64, 16)
    scheduler = memforge_c.Scheduler(allocator, 8)
    scheduler.add_sequence(1, [1, 2, 3], 4)
    scheduler.step()

    for _ in range(4):
        scheduler.update_sequence(1, 9, False)
    result = scheduler.step()

    assert 1 in result.finished_sequence_ids
    assert allocator.get_num_free_blocks() == allocator.get_total_blocks()
    assert scheduler.has_unfinished_sequences() is False


def test_memory_pressure_preempts_instead_of_stranding_a_sequence():
    """
    When the pool runs dry the scheduler must free the victim's blocks and
    re-queue it, not drop it on the floor. Blocks must stay conserved.
    """
    allocator = memforge_c.BlockAllocator(6, 16)  # 96 tokens total
    scheduler = memforge_c.Scheduler(allocator, 8)
    for seq_id in (1, 2, 3):
        scheduler.add_sequence(seq_id, list(range(32)), 200)  # 2 blocks each; pool full

    result = scheduler.step()
    assert len(result.running_sequences) == 3

    preempted = []
    for _ in range(40):
        for info in result.running_sequences:
            scheduler.update_sequence(info.seq_id, 7, False)
        result = scheduler.step()
        preempted.extend(result.preempted_sequence_ids)
        if preempted:
            break

    assert preempted, "expected a preemption once the pool was exhausted"
    # The victim went back to the waiting queue rather than vanishing.
    assert result.num_waiting >= 1
    total = allocator.get_total_blocks()
    assert allocator.get_num_free_blocks() + allocator.get_num_allocated_blocks() == total


def test_preempted_sequence_is_rescheduled_for_recomputation():
    allocator = memforge_c.BlockAllocator(6, 16)
    scheduler = memforge_c.Scheduler(allocator, 8)
    for seq_id in (1, 2, 3):
        scheduler.add_sequence(seq_id, list(range(32)), 200)

    result = scheduler.step()
    victim = None
    for _ in range(40):
        for info in result.running_sequences:
            scheduler.update_sequence(info.seq_id, 7, False)
        result = scheduler.step()
        if result.preempted_sequence_ids:
            victim = result.preempted_sequence_ids[0]
            break

    assert victim is not None
    sequence = scheduler.get_sequence(victim)
    # Its tokens are retained so the ML layer can recompute prompt + output,
    # and it needs a fresh prefill.
    assert sequence.get_output_len() > 0
    assert sequence.has_completed_prefill() is False
    assert len(sequence.get_block_table()) == 0


def test_request_larger_than_the_whole_pool_is_aborted_not_blocked():
    """One impossible request must not head-of-line block everything behind it."""
    allocator = memforge_c.BlockAllocator(4, 16)  # 64 tokens total
    scheduler = memforge_c.Scheduler(allocator, 8)
    scheduler.add_sequence(99, list(range(500)), 10)  # needs 32 blocks
    scheduler.add_sequence(100, list(range(16)), 10)  # needs 1

    result = scheduler.step()

    assert 99 in result.aborted_sequence_ids
    assert [info.seq_id for info in result.running_sequences] == [100]


def test_abort_releases_a_running_sequence():
    allocator = memforge_c.BlockAllocator(32, 16)
    scheduler = memforge_c.Scheduler(allocator, 8)
    scheduler.add_sequence(1, list(range(32)), 16)
    scheduler.step()
    assert allocator.get_num_allocated_blocks() == 2

    assert scheduler.abort_sequence(1) is True

    assert allocator.get_num_free_blocks() == 32
    assert scheduler.abort_sequence(12345) is False


def test_max_num_seqs_caps_the_running_set():
    allocator = memforge_c.BlockAllocator(64, 16)
    scheduler = memforge_c.Scheduler(allocator, 2)
    for seq_id in range(1, 6):
        scheduler.add_sequence(seq_id, [1, 2, 3], 16)

    result = scheduler.step()

    assert len(result.running_sequences) == 2
    assert result.num_waiting == 3


def test_step_result_reports_pool_utilization():
    allocator = memforge_c.BlockAllocator(10, 16)
    scheduler = memforge_c.Scheduler(allocator, 4)
    scheduler.add_sequence(1, list(range(32)), 16)

    result = scheduler.step()

    assert result.total_blocks == 10
    assert result.num_free_blocks == 8
    assert result.memory_utilization_ratio == pytest.approx(0.2)
