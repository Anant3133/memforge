#!/usr/bin/env bash
# Attempts to fix what is failing. Waits for the benchmark rounds so the GPU is quiet.
cd "$(dirname "$0")/.."
. .venv/bin/activate; export HF_HOME=$PWD/.hf_cache
until grep -q GPU_ALL_DONE benchmarks/results/runs/gpu_all.log && grep -q DL3_DONE .hf_download3.log; do sleep 20; done
Q=Qwen/Qwen2.5; Q3=Qwen/Qwen3; O=benchmarks/results/runs
spec() { # label target draft dtype lookahead tokens
  mkdir -p $O/$1; echo "=== $1"
  python benchmarks/run_speculative_batching.py --model $2 --draft-model $3 --device cuda --dtype $4 --batch-sizes 1 2 4 \
     --lookahead $5 --max-new-tokens $6 --kv-budget-mb 1024 --out $O/$1/speculative.json > $O/$1/speculative.log 2>&1
  echo "exit=$? $(grep -E 'draft acceptance rate|speedup \(' $O/$1/speculative.log | tr -s ' ' | tr '\n' ';' | cut -c1-200)"
}
# 1. draft search (acceptance, not size, decides)
spec exp-draft-q7b-from-0.5b  $Q-7B-Instruct $Q-0.5B-Instruct fp16 4 24
spec exp-draft-q7b-from-1.5b  $Q-7B-Instruct $Q-1.5B-Instruct fp16 4 24
spec exp-draft-q3-8b-from-0.6b $Q3-8B $Q3-0.6B fp16 4 24
spec exp-draft-q3-8b-from-1.7b $Q3-8B $Q3-1.7B fp16 4 24
spec exp-draft-q3-8b-from-4b   $Q3-8B $Q3-4B   fp16 4 24
# 2. lookahead sweep on the 8B with the 1.7B draft
for k in 2 3 6 8; do spec exp-lookahead-q3-8b-k$k $Q3-8B $Q3-1.7B fp16 $k 24; done
# 3. fp16 vs fp32 exactness (head_dim 128 model small enough for fp32)
spec exp-exact-q3-4b-fp16 $Q3-4B $Q3-0.6B fp16 4 24
spec exp-exact-q3-4b-fp32 $Q3-4B $Q3-0.6B fp32 4 24
spec exp-exact-q3-4b-bf16 $Q3-4B $Q3-0.6B bf16 4 24
# 4. stress / soak per model
for m in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-8B; do
  n=$(basename $m); mkdir -p $O/stress-$n; echo "=== stress $n"
  python benchmarks/run_stress_test.py --spawn-server --model $m --requests 110 --concurrency 8 --tokens 16 --kv-budget-mb 2048 > $O/stress-$n/stress.log 2>&1
  echo "exit=$? $(tail -2 $O/stress-$n/stress.log | tr '\n' ' ' | cut -c1-200)"
done
echo EXPERIMENTS_DONE
