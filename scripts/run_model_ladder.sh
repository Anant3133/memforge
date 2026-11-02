#!/usr/bin/env bash
# Usage: scripts/run_model_ladder.sh <label> <device> <dtype> <hf-model> [draft-model]
# Runs the phase benchmarks for one model, archiving JSON under benchmarks/results/runs/<label>/.
set -u
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HOME=$PWD/.hf_cache OMP_NUM_THREADS=${OMP_NUM_THREADS:-32} MKL_NUM_THREADS=${OMP_NUM_THREADS:-32}
L=$1; D=$2; T=$3; M=$4; DR=${5:-}
O=benchmarks/results/runs/$L; mkdir -p "$O"
run() { n=$1; shift; echo "=== $L :: $n"; "$@" > "$O/$n.log" 2>&1; echo "exit=$? $(tail -1 $O/$n.log)"; }
C="--device $D --dtype $T"
run prefix_sharing  python benchmarks/run_prefix_sharing.py  --model $M $C --concurrency 1 4 --tokens 12 --prefix-tokens 256 --out $O/prefix_sharing.json
run context_sweep   python benchmarks/run_context_sweep.py   --model $M $C --contexts ${CTXS:-128 512 1024} --repeats ${REPEATS:-9} --warmup 2 --kv-budget-mb ${KV:-1024} --skip-batched --out $O/context_sweep.json
run chunked_prefill python benchmarks/run_chunked_prefill.py --model $M $C --prompt-tokens 512 --victim-tokens 16 --budgets 0 64 16 4 --kv-budget-mb ${KV:-256} --out $O/chunked_prefill.json
if [ -n "$DR" ]; then
run speculative     python benchmarks/run_speculative_batching.py --model $M --draft-model $DR $C --batch-sizes 1 2 4 --max-new-tokens 16 --kv-budget-mb ${KV:-256} --out $O/speculative.json
fi
