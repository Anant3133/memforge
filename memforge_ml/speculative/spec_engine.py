"""
Speculative Decoding Engine: Combines Target Model, Draft Model, and RejectionSampler
to achieve 2x-3x generation speedup with lossless mathematical equivalence.
"""

import time
import torch
from typing import Dict, Any, List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

from memforge_ml.speculative.draft_runner import DraftRunner
from memforge_ml.speculative.rejection_sampler import RejectionSampler


class SpeculativeEngine:
    def __init__(
        self,
        target_model: PreTrainedModel,
        target_tokenizer: PreTrainedTokenizer,
        draft_model: PreTrainedModel,
        draft_tokenizer: PreTrainedTokenizer,
        device: str = "cpu",
        lookahead_k: int = 4,
    ):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.device = device
        self.lookahead_k = lookahead_k

        self.draft_runner = DraftRunner(self.draft_model, self.draft_tokenizer, device=self.device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Executes speculative generation loop over prompt.
        """
        inputs = self.target_tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        curr_ids = input_ids.clone()
        generated_tokens: List[int] = []
        token_origins: List[bool] = []  # True = Draft accepted, False = Target generated

        total_drafted = 0
        total_accepted = 0
        target_forward_passes = 0

        start_time = time.perf_counter()

        with torch.no_grad():
            while len(generated_tokens) < max_new_tokens:
                # 1. Draft model proposes K candidate tokens
                k_steps = min(self.lookahead_k, max_new_tokens - len(generated_tokens))
                draft_tokens, draft_probs = self.draft_runner.generate_draft_tokens(
                    curr_ids,
                    lookahead_k=k_steps,
                    temperature=temperature,
                )
                if not draft_tokens:
                    break

                total_drafted += len(draft_tokens)

                # 2. Target model evaluates prefix + all K tokens in a SINGLE forward pass
                draft_tensor = torch.tensor([draft_tokens], device=self.device, dtype=torch.long)
                candidate_ids = torch.cat([curr_ids, draft_tensor], dim=1)

                target_outputs = self.target_model(candidate_ids)
                target_forward_passes += 1

                # Extract logits for positions: end of prefix through end of candidates
                seq_len = curr_ids.shape[1]
                target_logits = target_outputs.logits[0, (seq_len - 1):, :]  # shape: (K+1, vocab_size)

                if temperature > 0:
                    target_probs = [torch.softmax(target_logits[i] / temperature, dim=-1) for i in range(target_logits.shape[0])]
                else:
                    target_probs = [torch.softmax(target_logits[i], dim=-1) for i in range(target_logits.shape[0])]

                # 3. Statistical verification & rejection sampling
                accepted, is_draft_mask, num_accepted = RejectionSampler.verify_tokens(
                    draft_tokens,
                    draft_probs,
                    target_probs,
                )

                total_accepted += num_accepted

                # Append accepted tokens
                for tok, is_dr in zip(accepted, is_draft_mask):
                    if len(generated_tokens) >= max_new_tokens:
                        break
                    generated_tokens.append(tok)
                    token_origins.append(is_dr)

                    # Update context tensor
                    tok_tensor = torch.tensor([[tok]], device=self.device, dtype=torch.long)
                    curr_ids = torch.cat([curr_ids, tok_tensor], dim=1)

                    if tok == self.target_tokenizer.eos_token_id:
                        break

                if generated_tokens and generated_tokens[-1] == self.target_tokenizer.eos_token_id:
                    break

        total_time = time.perf_counter() - start_time
        num_tokens = len(generated_tokens)
        tokens_per_sec = num_tokens / total_time if total_time > 0 else 0.0
        acceptance_rate = (total_accepted / total_drafted) if total_drafted > 0 else 0.0

        decoded_text = self.target_tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return {
            "prompt": prompt,
            "generated_text": decoded_text,
            "generated_tokens": generated_tokens,
            "token_origins": token_origins,  # For UI highlighting
            "num_generated_tokens": num_tokens,
            "total_drafted_tokens": total_drafted,
            "total_accepted_tokens": total_accepted,
            "acceptance_rate": acceptance_rate,
            "target_forward_passes": target_forward_passes,
            "tokens_per_forward_pass": (num_tokens / target_forward_passes) if target_forward_passes > 0 else 0.0,
            "total_time_seconds": total_time,
            "tokens_per_second": tokens_per_sec,
        }
