#!/usr/bin/env bash
cd "$(dirname "$0")/.."
S=HuggingFaceTB/SmolLM2; Q=Qwen/Qwen2.5
scripts/run_model_ladder.sh cpu-smol360 cpu fp32 $S-360M-Instruct $S-135M-Instruct
scripts/run_model_ladder.sh cpu-qwen05  cpu fp32 $Q-0.5B-Instruct
scripts/run_model_ladder.sh cpu-qwen15  cpu fp32 $Q-1.5B-Instruct $Q-0.5B-Instruct
CTXS="128 512" REPEATS=5 scripts/run_model_ladder.sh cpu-qwen3b cpu fp32 $Q-3B-Instruct $Q-0.5B-Instruct
CTXS="128 512" REPEATS=3 scripts/run_model_ladder.sh cpu-qwen7b cpu fp32 $Q-7B-Instruct $Q-0.5B-Instruct
echo LADDER_DONE
