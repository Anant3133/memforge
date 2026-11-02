#!/usr/bin/env bash
# Eager vs CUDA-graph draft, same pairs, back to back, 3 rounds.
cd "$(dirname "$0")/.."; . .venv/bin/activate; export HF_HOME=$PWD/.hf_cache; O=benchmarks/results/runs
Q=Qwen/Qwen2.5; Q3=Qwen/Qwen3
run() { # label target draft k extra
  d=$O/$1; mkdir -p $d
  python benchmarks/run_speculative_batching.py --model $2 --draft-model $3 --device cuda --dtype fp16 --batch-sizes 1 2 4 \
     --lookahead $4 --max-new-tokens 32 --kv-budget-mb 1024 $5 --out $d/speculative.json > $d/speculative.log 2>&1
  echo "$1 exit=$?"
}
for r in 1 2 3; do
  for k in 3 4; do
    run exp-graph-r$r-q3-8b-0.6b-k$k-eager $Q3-8B $Q3-0.6B $k
    run exp-graph-r$r-q3-8b-0.6b-k$k-graph $Q3-8B $Q3-0.6B $k --draft-cuda-graph
    run exp-graph-r$r-q3-8b-1.7b-k$k-eager $Q3-8B $Q3-1.7B $k
    run exp-graph-r$r-q3-8b-1.7b-k$k-graph $Q3-8B $Q3-1.7B $k --draft-cuda-graph
    run exp-graph-r$r-q7b-0.5b-k$k-eager $Q-7B-Instruct $Q-0.5B-Instruct $k
    run exp-graph-r$r-q7b-0.5b-k$k-graph $Q-7B-Instruct $Q-0.5B-Instruct $k --draft-cuda-graph
  done
done; echo GRAPH_BENCH_DONE
