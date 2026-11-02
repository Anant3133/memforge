#!/usr/bin/env bash
# Full GPU ladder, 3 rounds. 14B deliberately excluded (needs explicit approval).
cd "$(dirname "$0")/.."
S=HuggingFaceTB/SmolLM2; Q=Qwen/Qwen2.5; Q3=Qwen/Qwen3
export KV=1024 CTXS="128 512 1024 2048" REPEATS=9
for r in 1 2 3; do
  scripts/run_model_ladder.sh gpu-r$r-qwen7b   cuda fp16 $Q-7B-Instruct   $Q-0.5B-Instruct
  scripts/run_model_ladder.sh gpu-r$r-qwen3-8b cuda fp16 $Q3-8B           $Q3-0.6B
  scripts/run_model_ladder.sh gpu-r$r-qwen3b   cuda fp16 $Q-3B-Instruct   $Q-0.5B-Instruct
  scripts/run_model_ladder.sh gpu-r$r-qwen15   cuda fp16 $Q-1.5B-Instruct $Q-0.5B-Instruct
  scripts/run_model_ladder.sh gpu-r$r-qwen05   cuda fp16 $Q-0.5B-Instruct
  scripts/run_model_ladder.sh gpu-r$r-smol360  cuda fp16 $S-360M-Instruct $S-135M-Instruct
  scripts/run_model_ladder.sh gpu-r$r-smol135  cuda fp16 $S-135M-Instruct
  echo "ROUND $r DONE"
done
echo GPU_ALL_DONE
