"""
Phase 1: Naive Baseline LLM Generation Runner
Runs standard unoptimized Hugging Face / PyTorch autoregressive generation loop
to establish baseline tokens-per-second, time-to-first-token (TTFT), and VRAM utilization.
"""

import time
import torch
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from memforge_ml.model_defaults import DEFAULT_TARGET_MODEL as DEFAULT_MODEL


class NaiveBaselineRunner:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.model_name = model_name

        print(f"[NaiveBaseline] Loading tokenizer and model: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        print("[NaiveBaseline] Model loaded successfully.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Executes a naive autoregressive token-by-token loop without KV caching or batching optimizations.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        first_token_time = None
        generated_tokens: List[int] = []

        curr_input_ids = input_ids.clone()

        with torch.no_grad():
            for step in range(max_new_tokens):
                step_start = time.perf_counter()

                # Naive forward pass over the entire growing sequence (re-reading words 1..N-1)
                outputs = self.model(curr_input_ids)
                next_token_logits = outputs.logits[:, -1, :]

                # Apply temperature
                if temperature > 0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        probs = probs.masked_fill(indices_to_remove, 0.0)
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                token_id = next_token.item()
                generated_tokens.append(token_id)

                if step == 0:
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    first_token_time = time.perf_counter() - start_time

                if token_id == self.tokenizer.eos_token_id:
                    break

                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)

        if self.device == "cuda":
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time

        num_tokens = len(generated_tokens)
        tokens_per_sec = num_tokens / total_time if total_time > 0 else 0.0
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if self.device == "cuda"
            else 0.0
        )

        decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return {
            "prompt": prompt,
            "generated_text": decoded_text,
            "num_generated_tokens": num_tokens,
            "prompt_length": prompt_len,
            "total_time_seconds": total_time,
            "time_to_first_token_seconds": first_token_time or total_time,
            "tokens_per_second": tokens_per_sec,
            "peak_vram_mb": peak_vram_mb,
        }


if __name__ == "__main__":
    runner = NaiveBaselineRunner()
    res = runner.generate("Explain the concept of virtual memory in operating systems in two sentences:")
    print("\n--- Naive Baseline Benchmark Result ---")
    print(f"Generated text: {res['generated_text']}")
    print(f"Tokens: {res['num_generated_tokens']} | Tokens/sec: {res['tokens_per_second']:.2f}")
    print(f"TTFT: {res['time_to_first_token_seconds']*1000:.2f}ms | Total: {res['total_time_seconds']:.2f}s")
