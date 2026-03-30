import argparse
import json
from typing import Any, Dict, List

from benchmark_common import (
    TritonCompileObserver,
    backend_diagnostics,
    choose_cpu_n,
    compile_mm,
    compute_stats,
    dump_json,
    make_inputs,
    parse_shape,
    sync_if_needed,
    tflops,
)


def run_ratio(torch, m, k, n, dtype, ratio, n_align, min_chunk_n, warmup, repeat):
    a_cuda, b_cuda = make_inputs(torch, m, k, n, dtype, device="cuda")
    cpu_n = choose_cpu_n(n, ratio, n_align, min_chunk_n)
    gpu_n = n - cpu_n

    cpu_obs = TritonCompileObserver()
    gpu_obs = TritonCompileObserver()
    cpu_compiled = None
    gpu_compiled = None
    a_cpu = None
    b_cpu = None

    if cpu_n > 0:
        a_cpu = a_cuda.to("cpu")
        b_cpu = b_cuda[:, :cpu_n].to("cpu")
        with cpu_obs:
            cpu_compiled = compile_mm(torch, "cpu", a_cpu, b_cpu, force_triton_shared=True)
    if gpu_n > 0:
        with gpu_obs:
            gpu_compiled = compile_mm(torch, "cuda", a_cuda, b_cuda[:, cpu_n:], force_triton_shared=False)

    for _ in range(warmup):
        left_cuda = None
        right_cuda = None
        if cpu_n > 0:
            left_cuda = cpu_compiled(a_cpu, b_cpu).to("cuda")
        if gpu_n > 0:
            right_cuda = gpu_compiled(a_cuda, b_cuda[:, cpu_n:])
        if left_cuda is not None and right_cuda is not None:
            _ = torch.cat([left_cuda, right_cuda], dim=1)
        elif left_cuda is not None:
            _ = left_cuda
        else:
            _ = right_cuda
        sync_if_needed(torch)

    times_ms: List[float] = []
    import time
    for _ in range(repeat):
        sync_if_needed(torch)
        t0 = time.perf_counter()
        left_cuda = None
        right_cuda = None
        if cpu_n > 0:
            lhs_cpu = a_cuda.to("cpu")
            rhs_cpu = b_cuda[:, :cpu_n].to("cpu")
            left_cuda = cpu_compiled(lhs_cpu, rhs_cpu).to("cuda")
        if gpu_n > 0:
            right_cuda = gpu_compiled(a_cuda, b_cuda[:, cpu_n:])
        if left_cuda is not None and right_cuda is not None:
            out = torch.cat([left_cuda, right_cuda], dim=1)
        elif left_cuda is not None:
            out = left_cuda
        else:
            out = right_cuda
        sync_if_needed(torch)
        t1 = time.perf_counter()
        _ = out
        times_ms.append((t1 - t0) * 1000.0)

    stats = compute_stats(times_ms)
    return {
        "shape": [m, k, n],
        "dtype": dtype,
        "cpu_ratio_requested": ratio,
        "cpu_n": cpu_n,
        "gpu_n": gpu_n,
        "cpu_target_seen": cpu_obs.cpu_target_seen,
        "cuda_target_seen": gpu_obs.cuda_target_seen,
        "median_ms": stats.median_ms,
        "mean_ms": stats.mean_ms,
        "min_ms": stats.min_ms,
        "max_ms": stats.max_ms,
        "stdev_ms": stats.stdev_ms,
        "p90_ms": stats.p90_ms,
        "p95_ms": stats.p95_ms,
        "tflops": tflops(m, k, n, stats.median_ms),
    }


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--shape", type=parse_shape, action="append", required=True)
    ap.add_argument("--ratio", type=float, action="append", required=True)
    ap.add_argument("--n-align", type=int, default=32)
    ap.add_argument("--min-chunk-n", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print(json.dumps({"status": "SKIP", "reason": "CUDA is required"}, indent=2))
        return 0

    results: List[Dict[str, Any]] = []
    for shape in args.shape:
        m, k, n = shape
        for ratio in args.ratio:
            results.append(
                run_ratio(torch, m, k, n, args.dtype, ratio, args.n_align, args.min_chunk_n, args.warmup, args.repeat)
            )

    payload = {
        "backend_diagnostics": backend_diagnostics(torch),
        "requested": {
            "dtype": args.dtype,
            "shapes": [list(s) for s in args.shape],
            "ratios": args.ratio,
            "n_align": args.n_align,
            "min_chunk_n": args.min_chunk_n,
            "warmup": args.warmup,
            "repeat": args.repeat,
        },
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    dump_json(args.json_out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
