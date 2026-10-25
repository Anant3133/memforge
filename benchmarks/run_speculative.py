"""
MemForge Benchmark: Speculative Decoding Speedup & Acceptance Rate
Evaluates latency reduction factor and token acceptance percentage when pairing
a Draft model with a Target model.
"""

import sys
import os
import argparse
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memforge_ml.model_defaults import (
    DEFAULT_DRAFT_MODEL as DEFAULT_DRAFT,
    DEFAULT_TARGET_MODEL as DEFAULT_TARGET,
)
from memforge_ml.models.model_loader import ModelLoader
from memforge_ml.speculative.spec_engine import SpeculativeEngine

console = Console()

def run_speculative_benchmark(
    target_model_name: str = DEFAULT_TARGET,
    draft_model_name: str = DEFAULT_DRAFT,
    max_tokens: int = 64,
):
    console.print("[bold cyan]Running MemForge Speculative Decoding Benchmark[/bold cyan]")
    console.print(f"Target Model: [yellow]{target_model_name}[/yellow]")
    console.print(f"Draft Model:  [green]{draft_model_name}[/green]")

    # Load models
    target_model, target_tokenizer = ModelLoader.load_model_and_tokenizer(target_model_name)
    draft_model, draft_tokenizer = ModelLoader.load_model_and_tokenizer(draft_model_name)

    spec_engine = SpeculativeEngine(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        draft_model=draft_model,
        draft_tokenizer=draft_tokenizer,
        lookahead_k=4,
    )

    test_prompt = "Explain the difference between L1, L2, and L3 CPU caches in hardware architecture:"

    console.print("\n[dim]1. Running Speculative Engine Generation...[/dim]")
    spec_res = spec_engine.generate(test_prompt, max_new_tokens=max_tokens)

    table = Table(title="Speculative Decoding Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Generated Text", spec_res["generated_text"][:80] + "...")
    table.add_row("Total Generated Tokens", str(spec_res["num_generated_tokens"]))
    table.add_row("Draft Tokens Proposed", str(spec_res["total_drafted_tokens"]))
    table.add_row("Draft Tokens Accepted", str(spec_res["total_accepted_tokens"]))
    table.add_row("Draft Acceptance Rate", f"{spec_res['acceptance_rate']*100:.1f}%")
    table.add_row("Target Forward Passes", str(spec_res["target_forward_passes"]))
    table.add_row("Tokens per Forward Pass", f"{spec_res['tokens_per_forward_pass']:.2f}")
    table.add_row("Speculative Tokens / Sec", f"{spec_res['tokens_per_second']:.2f}")

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Speculative Decoding Benchmark")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--draft", type=str, default=DEFAULT_DRAFT)
    parser.add_argument("--tokens", type=int, default=64)
    args = parser.parse_args()
    run_speculative_benchmark(target_model_name=args.target, draft_model_name=args.draft, max_tokens=args.tokens)
