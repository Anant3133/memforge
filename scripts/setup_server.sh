#!/usr/bin/env bash
# One-command setup for a fresh Linux GPU server (everything stays inside the project folder).
#   scripts/setup_server.sh            # do it
#   scripts/setup_server.sh --dry-run  # print the steps only
# Steps mirror docs/A100_SERVER_GUIDE.md section 1. Pins the combination that works:
# torch 2.8.0 + torchvision 0.23.0 (cu126) inside a venv, transformers 5.x.
set -euo pipefail
cd "$(dirname "$0")/.."
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
run() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }

run python3 -m venv --system-site-packages .venv
[ "$DRY" = 1 ] || . .venv/bin/activate
run pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
run pip install -U "transformers>=5,<6" accelerate huggingface_hub fastapi uvicorn websockets httpx \
    pybind11 pytest pytest-timeout pytest-asyncio psutil matplotlib rich tqdm
run python setup.py build_ext --inplace --force
echo "+ export HF_HOME=$PWD/.hf_cache   (models stay inside the project)"
[ "$DRY" = 1 ] || { export HF_HOME=$PWD/.hf_cache; python scripts/check_env.py; }
echo "+ CUDA_VISIBLE_DEVICES='' python -m pytest tests/python -q -p no:rerunfailures --timeout=300"
echo "setup steps complete$([ "$DRY" = 1 ] && echo ' (dry run: nothing executed)')"
