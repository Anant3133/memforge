#!/usr/bin/env bash
cd "$(dirname "$0")/.."
scripts/run_batched_graph_bench.sh
scripts/run_phaseA_graphed.sh
echo ITEMS_1_2_DONE
