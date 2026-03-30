#!/usr/bin/env bash
set -euo pipefail

: "${TRITON_SHARED_ENABLE_AMX:=0}"
export TRITON_SHARED_ENABLE_AMX

python3 benchmark_hybrid_mm_component_breakdown.py \
  --dtype float32 \
  --cpu-ratio 0.5 \
  --warmup 5 \
  --repeat 20 \
  --shape 1024,1024,1024 \
  --json-out hybrid_mm_component_breakdown.json
