"""
Draft Model Runner: Generates candidate sequence tokens using the lightweight draft model.
"""

import torch
from typing import List, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer


class DraftRunner:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_draft_tokens(
        self,
        prefix_ids: torch.Tensor,   # (1, seq_len)
        lookahead_k: int = 4,
        temperature: float = 0.7,
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Generates K candidate draft tokens autoregressively.
        Returns:
            draft_tokens: List of token IDs
            draft_probs: List of probability distributions for each step
        """
        curr_ids = prefix_ids.clone()
        draft_tokens: List[int] = []
        draft_probs: List[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(lookahead_k):
                outputs = self.model(curr_ids)
                next_logits = outputs.logits[:, -1, :]  # (1, vocab_size)

                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1).squeeze(0)
                    next_token = torch.multinomial(probs, num_samples=1).view(1, 1)
                else:
                    probs = torch.softmax(next_logits, dim=-1).squeeze(0)
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

                token_id = int(next_token.item())
                draft_tokens.append(token_id)
                draft_probs.append(probs)

                if token_id == self.tokenizer.eos_token_id:
                    break

                curr_ids = torch.cat([curr_ids, next_token], dim=1)

        return draft_tokens, draft_probs
