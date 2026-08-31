"""
Statistical Rejection Sampling for Speculative Decoding
Ensures that the output token distribution strictly matches the Target model distribution P(x)
while accepting tokens proposed by the Draft model Q(x).
"""

import torch
from typing import List, Tuple, Optional


class RejectionSampler:
    @staticmethod
    def sample_from_probs(probs: torch.Tensor) -> int:
        """
        Samples a single token from a 1D probability distribution.
        """
        # Ensure valid non-negative probabilities summing to 1
        probs = torch.clamp(probs, min=0.0)
        prob_sum = probs.sum()
        if prob_sum == 0:
            probs = torch.ones_like(probs) / probs.shape[0]
        else:
            probs = probs / prob_sum

        sample = torch.multinomial(probs, num_samples=1)
        return int(sample.item())

    @staticmethod
    def verify_tokens(
        draft_tokens: List[int],
        draft_probs: List[torch.Tensor],    # List of (vocab_size,) tensors
        target_probs: List[torch.Tensor],   # List of (vocab_size,) tensors for positions 1..K+1
    ) -> Tuple[List[int], List[bool], int]:
        """
        Executes speculative rejection sampling over K draft tokens.

        Returns:
            accepted_tokens: List of tokens generated in this iteration.
            token_is_draft: Boolean mask indicating whether each token was drafted or target-generated.
            num_accepted_draft: Number of draft tokens successfully accepted.
        """
        accepted_tokens: List[int] = []
        token_is_draft: List[bool] = []
        num_accepted = 0

        k = len(draft_tokens)

        for i in range(k):
            token = draft_tokens[i]
            q_token = float(draft_probs[i][token].item())
            p_token = float(target_probs[i][token].item())

            # Acceptance probability: min(1, p(x) / q(x))
            acceptance_prob = min(1.0, (p_token / (q_token + 1e-9)))
            r = float(torch.rand(1).item())

            if r <= acceptance_prob:
                # Token accepted!
                accepted_tokens.append(token)
                token_is_draft.append(True)
                num_accepted += 1
            else:
                # Token rejected! Resample from max(0, p(x) - q(x))
                diff_probs = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
                diff_sum = diff_probs.sum()

                if diff_sum > 0:
                    resampled_token = RejectionSampler.sample_from_probs(diff_probs / diff_sum)
                else:
                    resampled_token = RejectionSampler.sample_from_probs(target_probs[i])

                accepted_tokens.append(resampled_token)
                token_is_draft.append(False)
                # Discard all remaining draft tokens
                return accepted_tokens, token_is_draft, num_accepted

        # If all K tokens were accepted, sample one bonus token from target_probs[K]
        if num_accepted == k and len(target_probs) > k:
            bonus_token = RejectionSampler.sample_from_probs(target_probs[k])
            accepted_tokens.append(bonus_token)
            token_is_draft.append(False)

        return accepted_tokens, token_is_draft, num_accepted
