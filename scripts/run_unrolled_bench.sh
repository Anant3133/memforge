#!/usr/bin/env bash
# eager vs batched (stepwise) vs batched-unrolled (one replay per round), back to back, 3 rounds. Waits for the Phase A sweep.
cd "$(dirname "$0")/.."; . .venv/bin/activate; export HF_HOME=$PWD/.hf_cache; O=benchmarks/results/runs
until grep -q ITEMS_1_2_DONE /tmp/claude-0/-workspace-memforge/15324639-26ce-4efb-b97c-87d44ee26572/tasks/b0x0jgktp.output 2>/dev/null; do sleep 30; done
Q=Qwen/Qwen2.5; Q3=Qwen/Qwen3
run() { d=$O/$1; mkdir -p $d
  python benchmarks/run_speculative_batching.py --model $2 --draft-model $3 --device cuda --dtype fp16 --batch-sizes 1 2 4 8 \
     --lookahead $4 --max-new-tokens 32 --kv-budget-mb 512 $5 --out $d/speculative.json > $d/speculative.log 2>&1
  echo "$1 exit=$?"; }
for r in 1 2 3; do
  run exp-ugraph-r$r-q3-8b-0.6b-eager $Q3-8B $Q3-0.6B 3
  run exp-ugraph-r$r-q3-8b-0.6b-batched $Q3-8B $Q3-0.6B 3 "--draft-cuda-graph batched"
  run exp-ugraph-r$r-q3-8b-0.6b-unrolled $Q3-8B $Q3-0.6B 3 "--draft-cuda-graph batched-unrolled"
  run exp-ugraph-r$r-q7b-0.5b-unrolled $Q-7B-Instruct $Q-0.5B-Instruct 3 "--draft-cuda-graph batched-unrolled"
done; echo UGRAPH_DONE
