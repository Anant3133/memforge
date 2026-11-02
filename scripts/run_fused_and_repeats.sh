#!/usr/bin/env bash
# Phase C with fusion (never run on these models), 3 repeats each, GPU.
cd "$(dirname "$0")/.."; . .venv/bin/activate; export HF_HOME=$PWD/.hf_cache; O=benchmarks/results/runs
for r in 1 2 3; do for m in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-8B; do
  n=$(basename $m); d=$O/exp-fused-r$r-$n; mkdir -p $d; echo "=== $d"
  python benchmarks/run_chunked_prefill.py --model $m --device cuda --dtype fp16 --prompt-tokens 512 --victim-tokens 16 --budgets 0 64 16 4 --fused --kv-budget-mb 1024 --out $d/chunked_prefill.json > $d/chunked_prefill.log 2>&1
  echo "exit=$?"
done; done; echo FUSED_DONE
