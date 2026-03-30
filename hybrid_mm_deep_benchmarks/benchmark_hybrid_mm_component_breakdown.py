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
    timed_region_ms,
    tflops,
)


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--shape", type=parse_shape, action="append", required=True)
    ap.add_argument("--cpu-ratio", type=float, default=0.5)
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
    for m, k, n in args.shape:
        a_cuda, b_cuda = make_inputs(torch, m, k, n, args.dtype, device="cuda")
        cpu_n = choose_cpu_n(n, args.cpu_ratio, args.n_align, args.min_chunk_n)
        gpu_n = n - cpu_n
        a_cpu = a_cuda.to("cpu") if cpu_n > 0 else None
        b_cpu = b_cuda[:, :cpu_n].to("cpu") if cpu_n > 0 else None

        cpu_obs = TritonCompileObserver()
        gpu_obs = TritonCompileObserver()
        cpu_compiled = None
        gpu_compiled = None
        if cpu_n > 0:
            with cpu_obs:
                cpu_compiled = compile_mm(torch, "cpu", a_cpu, b_cpu, force_triton_shared=True)
        if gpu_n > 0:
            with gpu_obs:
                gpu_compiled = compile_mm(torch, "cuda", a_cuda, b_cuda[:, cpu_n:], force_triton_shared=False)

        # warmup
        for _ in range(args.warmup):
            left_cuda = None
            right_cuda = None
            if cpu_n > 0:
                left_cpu = cpu_compiled(a_cpu, b_cpu)
                left_cuda = left_cpu.to("cuda")
            if gpu_n > 0:
                right_cuda = gpu_compiled(a_cuda, b_cuda[:, cpu_n:])
            if left_cuda is not None and right_cuda is not None:
                out = torch.cat([left_cuda, right_cuda], dim=1)
            elif left_cuda is not None:
                out = left_cuda
            else:
                out = right_cuda
            sync_if_needed(torch)
            _ = out

        copy_to_cpu_ms = []
        cpu_mm_ms = []
        cpu_to_cuda_ms = []
        gpu_mm_ms = []
        cat_ms = []
        total_ms = []

        for _ in range(args.repeat):
            sync_if_needed(torch)
            _, total_start_ms = None, None
            import time
            t0 = time.perf_counter()

            if cpu_n > 0:
                (lhs_cpu, rhs_cpu), dt_copy = timed_region_ms(
                    torch, lambda: (a_cuda.to("cpu"), b_cuda[:, :cpu_n].to("cpu"))
                )
                _, dt_cpu_mm = timed_region_ms(torch, lambda: cpu_compiled(lhs_cpu, rhs_cpu))
                left_cpu = cpu_compiled(lhs_cpu, rhs_cpu)
                # recompute removed? let's avoid double-run by timing directly below
            else:
                dt_copy = 0.0
                dt_cpu_mm = 0.0
                left_cpu = None

            # need single execution per stage, rework stage timing inline
            sync_if_needed(torch)
            if cpu_n > 0:
                import time
                ts = time.perf_counter()
                lhs_cpu = a_cuda.to("cpu")
                rhs_cpu = b_cuda[:, :cpu_n].to("cpu")
                sync_if_needed(torch)
                te = time.perf_counter()
                dt_copy = (te - ts) * 1000.0

                ts = time.perf_counter()
                left_cpu = cpu_compiled(lhs_cpu, rhs_cpu)
                sync_if_needed(torch)
                te = time.perf_counter()
                dt_cpu_mm = (te - ts) * 1000.0

                ts = time.perf_counter()
                left_cuda = left_cpu.to("cuda")
                sync_if_needed(torch)
                te = time.perf_counter()
                dt_cpu_to_cuda = (te - ts) * 1000.0
            else:
                dt_copy = 0.0
                dt_cpu_mm = 0.0
                dt_cpu_to_cuda = 0.0
                left_cuda = None

            if gpu_n > 0:
                import time
                ts = time.perf_counter()
                right_cuda = gpu_compiled(a_cuda, b_cuda[:, cpu_n:])
                sync_if_needed(torch)
                te = time.perf_counter()
                dt_gpu_mm = (te - ts) * 1000.0
            else:
                right_cuda = None
                dt_gpu_mm = 0.0

            import time
            ts = time.perf_counter()
            if left_cuda is not None and right_cuda is not None:
                out = torch.cat([left_cuda, right_cuda], dim=1)
            elif left_cuda is not None:
                out = left_cuda
            else:
                out = right_cuda
            sync_if_needed(torch)
            te = time.perf_counter()
            dt_cat = (te - ts) * 1000.0
            t1 = time.perf_counter()

            copy_to_cpu_ms.append(dt_copy)
            cpu_mm_ms.append(dt_cpu_mm)
            cpu_to_cuda_ms.append(dt_cpu_to_cuda)
            gpu_mm_ms.append(dt_gpu_mm)
            cat_ms.append(dt_cat)
            total_ms.append((t1 - t0) * 1000.0)

        total_stats = compute_stats(total_ms)
        results.append(
            {
                "shape": [m, k, n],
                "dtype": args.dtype,
                "cpu_ratio_requested": args.cpu_ratio,
                "cpu_n": cpu_n,
                "gpu_n": gpu_n,
                "cpu_target_seen": cpu_obs.cpu_target_seen,
                "cuda_target_seen": gpu_obs.cuda_target_seen,
                "cpu_compile_calls": cpu_obs.calls,
                "gpu_compile_calls": gpu_obs.calls,
                "components": {
                    "copy_to_cpu_ms": compute_stats(copy_to_cpu_ms).__dict__,
                    "cpu_mm_ms": compute_stats(cpu_mm_ms).__dict__,
                    "cpu_to_cuda_ms": compute_stats(cpu_to_cuda_ms).__dict__,
                    "gpu_mm_ms": compute_stats(gpu_mm_ms).__dict__,
                    "cat_ms": compute_stats(cat_ms).__dict__,
                    "total_ms": total_stats.__dict__,
                },
                "total_tflops": tflops(m, k, n, total_stats.median_ms),
            }
        )

    payload = {
        "backend_diagnostics": backend_diagnostics(torch),
        "requested": {
            "dtype": args.dtype,
            "shapes": [list(s) for s in args.shape],
            "cpu_ratio": args.cpu_ratio,
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
