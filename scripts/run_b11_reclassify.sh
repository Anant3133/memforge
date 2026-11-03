#!/usr/bin/env bash
# B11: re-run the four pre-explainer draft-pair experiments with the current
# benchmark so every mismatch is classified (tie vs real). Old runs are kept.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/.hf_cache
PY=.venv/bin/python
run() { # name target draft
  out=benchmarks/results/runs/$1; mkdir -p "$out"
  $PY benchmarks/run_speculative_batching.py --model "$2" --draft-model "$3" \
    --lookahead 4 --batch-sizes 1 2 4 --max-new-tokens 24 --kv-budget-mb 512 \
    --out "$out/speculative_batching.json" > "$out/log.txt" 2>&1
  echo "$1 exit=$?"
}
run b11-q3-8b-from-0.6b Qwen/Qwen3-8B Qwen/Qwen3-0.6B
run b11-q3-8b-from-1.7b Qwen/Qwen3-8B Qwen/Qwen3-1.7B
run b11-q7b-from-0.5b Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-0.5B-Instruct
run b11-q7b-from-1.5b Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-1.5B-Instruct
