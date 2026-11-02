#!/usr/bin/env bash
# eager vs per-sequence graph vs batched graph, back to back, 3 rounds, batch 1/2/4/8.
cd "$(dirname "$0")/.."; . .venv/bin/activate; export HF_HOME=$PWD/.hf_cache; O=benchmarks/results/runs
Q=Qwen/Qwen2.5; Q3=Qwen/Qwen3
run() { # label target draft k extra
  d=$O/$1; mkdir -p $d
  python benchmarks/run_speculative_batching.py --model $2 --draft-model $3 --device cuda --dtype fp16 --batch-sizes 1 2 4 8 \
     --lookahead $4 --max-new-tokens 32 --kv-budget-mb 512 $5 --out $d/speculative.json > $d/speculative.log 2>&1
  echo "$1 exit=$?"
}
for r in 1 2 3; do
  for k in 3; do
    run exp-bgraph-r$r-q3-8b-0.6b-k$k-eager $Q3-8B $Q3-0.6B $k
    run exp-bgraph-r$r-q3-8b-0.6b-k$k-perseq $Q3-8B $Q3-0.6B $k "--draft-cuda-graph per-seq"
    run exp-bgraph-r$r-q3-8b-0.6b-k$k-batched $Q3-8B $Q3-0.6B $k "--draft-cuda-graph batched"
    run exp-bgraph-r$r-q7b-0.5b-k$k-eager $Q-7B-Instruct $Q-0.5B-Instruct $k
    run exp-bgraph-r$r-q7b-0.5b-k$k-batched $Q-7B-Instruct $Q-0.5B-Instruct $k "--draft-cuda-graph batched"
  done
done; echo BGRAPH_BENCH_DONE
