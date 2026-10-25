"""
MemForge Benchmark: Phase 1 Naive Baseline
Measures raw tokens/second, time-to-first-token, and memory usage without engine optimizations.
"""

import sys
import os
import argparse
from rich.console import Console
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memforge_ml.baseline.naive_runner import NaiveBaselineRunner
from memforge_ml.model_defaults import DEFAULT_TARGET_MODEL as DEFAULT_MODEL

console = Console()

def run_benchmark(model_name: str = DEFAULT_MODEL, max_tokens: int = 100):
    console.print(f"[bold cyan]Running MemForge Naive Baseline Benchmark[/bold cyan] on model: [yellow]{model_name}[/yellow]")

    runner = NaiveBaselineRunner(model_name=model_name)
    test_prompts = [
        "Explain what continuous batching is in large language model inference systems:",
        "Write a concise Python function to calculate the Fibonacci sequence recursively:",
        "Describe the architectural difference between virtual memory paging and PagedAttention:",
    ]

    results = []
    for prompt in test_prompts:
        console.print(f"\n[dim]Benchmarking prompt: '{prompt[:50]}...'[/dim]")
        res = runner.generate(prompt, max_new_tokens=max_tokens)
        results.append(res)

    table = Table(title="Phase 1 Naive Baseline Benchmark Results")
    table.add_column("Prompt Sample", style="cyan", no_wrap=True)
    table.add_column("Tokens", justify="right", style="green")
    table.add_column("Total Time (s)", justify="right", style="yellow")
    table.add_column("Tokens / Sec", justify="right", style="bold magenta")
    table.add_column("TTFT (ms)", justify="right", style="blue")
    table.add_column("Peak VRAM (MB)", justify="right", style="red")

    for r in results:
        table.add_row(
            r["prompt"][:40] + "...",
            str(r["num_generated_tokens"]),
            f"{r['total_time_seconds']:.2f}",
            f"{r['tokens_per_second']:.2f}",
            f"{r['time_to_first_token_seconds']*1000:.1f}",
            f"{r['peak_vram_mb']:.1f}",
        )

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MemForge Baseline Benchmark")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--tokens", type=int, default=64)
    args = parser.parse_args()
    run_benchmark(model_name=args.model, max_tokens=args.tokens)
