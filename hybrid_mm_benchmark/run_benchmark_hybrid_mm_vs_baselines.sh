#!/usr/bin/env bash
set -euo pipefail

# Run from the inductor repo root so the local _inductor.hybrid_mm_eager_bridge import works.
# Example:
#   cd ~/home/inductor
#   bash /path/to/run_benchmark_hybrid_mm_vs_baselines.sh

export TRITON_SHARED_ENABLE_AMX="${TRITON_SHARED_ENABLE_AMX:-0}"

python3 benchmark_hybrid_mm_vs_baselines.py \
  --dtype float32 \
  --warmup 10 \
  --repeat 30 \
  --shape 1024,4096,4096 \
  --shape 2048,4096,4096 \
  --shape 1024,11008,4096 \
  --modes eager,inductor,hybrid \
  --json-out hybrid_mm_benchmark_results.json \
  --csv-out hybrid_mm_benchmark_results.csv
