#!/usr/bin/env bash
set -euo pipefail

: "${TRITON_SHARED_ENABLE_AMX:=0}"
export TRITON_SHARED_ENABLE_AMX

python3 benchmark_hybrid_mm_cpu_ratio_sweep.py \
  --dtype float32 \
  --warmup 5 \
  --repeat 20 \
  --shape 4096,4096,4096 \
  --shape 4096,11008,4096 \
  --ratio 0.0 \
  --ratio 0.25 \
  --ratio 0.5 \
  --ratio 0.75 \
  --ratio 1.0 \
  --json-out hybrid_mm_cpu_ratio_sweep.json
