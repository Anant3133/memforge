#!/usr/bin/env bash
# Phase A, graphed, finer gather bucket (32 vs 128 tokens) on the models that missed at 2048.
cd "$(dirname "$0")/.."; . .venv/bin/activate; export HF_HOME=$PWD/.hf_cache CTX_BUCKET=32; O=benchmarks/results/runs
for r in 1 2 3; do
  for m in Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-8B; do
    n=$(basename $m); d=$O/phaseA-b32-r$r-$n; mkdir -p $d
    python benchmarks/run_context_sweep.py --model $m --device cuda --dtype fp16 --contexts 1024 2048 \
      --steps 32 --repeats 7 --skip-batched --graphed --out $d/context_sweep.json > $d/context_sweep.log 2>&1
    echo "$d exit=$?"
  done
done; echo B32_DONE
