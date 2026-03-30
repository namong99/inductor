#!/usr/bin/env bash
set -euo pipefail

# Adjust this if needed for your environment.
export TRITON_SHARED_ENABLE_AMX=${TRITON_SHARED_ENABLE_AMX:-0}
export TRITON_SHARED_OPT_PATH=${TRITON_SHARED_OPT_PATH:-/home/hosu99/home/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt}

python3 benchmark_hybrid_mm_vs_baselines.py \
  --dtype float32 \
  --warmup 5 \
  --repeat 20 \
  --modes eager,inductor,hybrid \
  --shape 4096,4096,4096 \
  --shape 8192,4096,4096 \
  --shape 4096,11008,4096 \
  --shape 4096,16384,4096 \
  --shape 8192,11008,4096 \
  --json-out hybrid_mm_benchmark_large_shapes.json \
  --csv-out hybrid_mm_benchmark_large_shapes.csv
