#!/usr/bin/env bash
# Item 4: repeat the CPU chunked-prefill (Phase C) result, which was a single run per model.
cd "$(dirname "$0")/.."; . .venv/bin/activate
export HF_HOME=$PWD/.hf_cache OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
O=benchmarks/results/runs
for r in 2 3 4; do
  for m in HuggingFaceTB/SmolLM2-360M-Instruct Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct; do
    n=$(basename $m); d=$O/cpu-r$r-$n; mkdir -p $d
    python benchmarks/run_chunked_prefill.py --model $m --device cpu --dtype fp32 --prompt-tokens 512 --victim-tokens 16 \
      --budgets 0 64 16 4 --kv-budget-mb 256 --out $d/chunked_prefill.json > $d/chunked_prefill.log 2>&1
    echo "$d exit=$?"
  done
done; echo CPU_REPEATS_DONE
