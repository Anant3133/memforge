#!/usr/bin/env bash
# GPU runs for models that fit in the headroom left by other tenants. Timings are CONTENDED.
cd "$(dirname "$0")/.."
S=HuggingFaceTB/SmolLM2; Q=Qwen/Qwen2.5
KV=200 CTXS="128 512 1024" REPEATS=9 scripts/run_model_ladder.sh gpu-smol360-contended cuda fp16 $S-360M-Instruct $S-135M-Instruct
KV=200 CTXS="128 512 1024" REPEATS=9 scripts/run_model_ladder.sh gpu-qwen05-contended cuda fp16 $Q-0.5B-Instruct
echo GPU_SMALL_DONE
