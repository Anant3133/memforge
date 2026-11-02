#!/usr/bin/env bash
# Phase A with graphed arms: eager contiguous vs eager paged vs graphed contiguous vs graphed paged.
cd "$(dirname "$0")/.."; . .venv/bin/activate; export HF_HOME=$PWD/.hf_cache; O=benchmarks/results/runs
for r in 1 2 3; do
  for m in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-8B; do
    n=$(basename $m); d=$O/phaseA-graphed-r$r-$n; mkdir -p $d
    python benchmarks/run_context_sweep.py --model $m --device cuda --dtype fp16 --contexts 128 512 1024 2048 \
      --steps 32 --repeats 7 --skip-batched --graphed --out $d/context_sweep.json > $d/context_sweep.log 2>&1
    echo "$d exit=$?"
  done
done; echo PHASEA_GRAPHED_DONE
