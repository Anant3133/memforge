"""A target with a wider padded vocabulary than its draft must still resolve a round."""
import torch
from memforge_ml.runtime.batched_speculative import resolve_round


def test_wider_target_vocab_does_not_crash_and_stays_greedy_exact():
    narrow, wide = 10, 14
    cand = [3, 7]
    q = [torch.nn.functional.one_hot(torch.tensor(t), narrow).float() for t in cand]
    logits = []
    for want in (3, 7, 2):                       # K+1 positions; last is the bonus
        row = torch.full((wide,), -5.0)
        row[want] = 5.0
        row[12] = 4.9                            # padding row, must never win or leak in
        logits.append(row)
    rnd = resolve_round(0, cand, q, logits, temperature=0.0)
    assert rnd.accepted[:2] == [3, 7] and rnd.num_accepted_draft == 2
    assert all(t < narrow for t in rnd.accepted)


def test_rejection_path_with_wider_target_vocab():
    # Draft proposes 3 then 5, target wants 3 then 7: second candidate is rejected,
    # which forces the residual (target - draft) computation across the width gap.
    narrow, wide = 10, 14
    cand = [3, 5]
    q = [torch.nn.functional.one_hot(torch.tensor(t), narrow).float() for t in cand]
    logits = []
    for want in (3, 7, 1):
        row = torch.full((wide,), -5.0)
        row[want] = 5.0
        row[12] = 4.9
        logits.append(row)
    rnd = resolve_round(0, cand, q, logits, temperature=0.0)
    assert rnd.accepted == [3, 7]                # greedy-exact: the target's own choice
    assert rnd.num_accepted_draft == 1
