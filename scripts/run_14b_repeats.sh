#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for r in 2 3; do
  MEMFORGE_INT8=1 CTXS="128 512 1024 2048" REPEATS=5 KV=512 scripts/run_model_ladder.sh gpu-r$r-qwen14b-int8 cuda fp16 Qwen/Qwen3-14B
done; echo R14B_DONE
