import torch
import pytest
from memforge_ml.speculative.rejection_sampler import RejectionSampler


def test_rejection_sampler_perfect_match():
    # If target and draft distributions are identical, acceptance should be 100%
    vocab_size = 100
    draft_probs = [torch.zeros(vocab_size) for _ in range(4)]
    target_probs = [torch.zeros(vocab_size) for _ in range(5)]

    draft_tokens = [10, 20, 30, 40]
    for i, tok in enumerate(draft_tokens):
        draft_probs[i][tok] = 1.0
        target_probs[i][tok] = 1.0
    target_probs[4][50] = 1.0  # bonus token

    accepted, is_draft, num_accepted = RejectionSampler.verify_tokens(
        draft_tokens,
        draft_probs,
        target_probs,
    )

    assert num_accepted == 4
    assert len(accepted) == 5  # 4 draft tokens + 1 bonus token
    assert accepted[:4] == draft_tokens
    assert accepted[4] == 50
    assert is_draft == [True, True, True, True, False]


def test_rejection_sampler_rejection():
    # If target probability is 0 for draft token, it MUST be rejected
    vocab_size = 50
    draft_probs = [torch.zeros(vocab_size)]
    target_probs = [torch.zeros(vocab_size), torch.zeros(vocab_size)]

    draft_tokens = [15]
    draft_probs[0][15] = 1.0
    target_probs[0][15] = 0.0
    target_probs[0][25] = 1.0  # target exclusively wants token 25

    accepted, is_draft, num_accepted = RejectionSampler.verify_tokens(
        draft_tokens,
        draft_probs,
        target_probs,
    )

    assert num_accepted == 0
    assert len(accepted) == 1
    assert accepted[0] == 25  # Correctly resampled token 25
    assert is_draft[0] is False
