#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch


def _import_hybrid_backend() -> Callable:
    # Prefer the local repo implementation when running from the inductor checkout.
    try:
        from _inductor.hybrid_mm_eager_bridge import hybrid_mm_eager_bridge_backend
        return hybrid_mm_eager_bridge_backend
    except Exception:
        pass
    try:
        from torch._inductor import hybrid_mm_eager_bridge_backend
        return hybrid_mm_eager_bridge_backend
    except Exception as exc:
        raise RuntimeError(
            "Unable to import hybrid_mm_eager_bridge_backend from either local _inductor or torch._inductor"
        ) from exc


class MMModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mm(a, b)


@dataclass
class TimingStats:
    first_run_ms: Optional[float]
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    p90_ms: float
    p95_ms: float
    tflops: float


@dataclass
class ModeResult:
    shape: Tuple[int, int, int]
    dtype: str
    mode: str
    status: str
    output_device: Optional[str]
    max_abs: Optional[float]
    max_rel: Optional[float]
    first_run_ms: Optional[float]
    median_ms: Optional[float]
    mean_ms: Optional[float]
    min_ms: Optional[float]
    max_ms: Optional[float]
    stdev_ms: Optional[float]
    p90_ms: Optional[float]
    p95_ms: Optional[float]
    tflops: Optional[float]
    speedup_vs_eager: Optional[float]
    speedup_vs_inductor: Optional[float]
    note: Optional[str]
    debug: Optional[Dict[str, Any]]


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


PRESETS = {
    # square-ish baseline
    "square_small": [(1024, 1024, 1024)],
    "square_medium": [(2048, 2048, 2048)],
    "square_large": [(4096, 4096, 4096)],
    # LLM-ish projections: [tokens*batch, hidden] x [hidden, 4*hidden] etc.
    "llm_proj_small": [(512, 4096, 4096), (512, 11008, 4096)],
    "llm_proj_medium": [(1024, 4096, 4096), (1024, 11008, 4096)],
    "llm_proj_large": [(2048, 4096, 4096), (2048, 11008, 4096)],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark hybrid_mm_eager_bridge vs baseline mm paths")
    p.add_argument("--dtype", default="float32", choices=sorted(DTYPE_MAP.keys()))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disable-tf32", action="store_true")
    p.add_argument("--m", type=int, nargs="*", default=[])
    p.add_argument("--n", type=int, nargs="*", default=[])
    p.add_argument("--k", type=int, nargs="*", default=[])
    p.add_argument("--shape", action="append", default=[], help="Shape triplet as M,N,K; can be repeated")
    p.add_argument("--preset", action="append", default=[], help="Preset name; can be repeated")
    p.add_argument("--m-list", default="", help="Comma-separated M list")
    p.add_argument("--n-list", default="", help="Comma-separated N list")
    p.add_argument("--k-list", default="", help="Comma-separated K list")
    p.add_argument("--mnk-grid", action="store_true", help="Interpret m/n/k lists as cartesian product")
    p.add_argument("--modes", default="eager,inductor,hybrid", help="Comma-separated modes: eager,inductor,hybrid")
    p.add_argument("--skip-correctness", action="store_true")
    p.add_argument("--json-out", default="")
    p.add_argument("--csv-out", default="")
    p.add_argument("--tag", default="")
    return p.parse_args()


def _parse_csv_ints(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def resolve_shapes(args: argparse.Namespace) -> List[Tuple[int, int, int]]:
    shapes: List[Tuple[int, int, int]] = []
    for spec in args.shape:
        parts = [int(x.strip()) for x in spec.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid --shape={spec!r}; expected M,N,K")
        m, n, k = parts
        shapes.append((m, n, k))

    for name in args.preset:
        if name not in PRESETS:
            raise ValueError(f"Unknown --preset={name!r}. Available: {sorted(PRESETS)}")
        shapes.extend(PRESETS[name])

    m_list = list(args.m) + _parse_csv_ints(args.m_list)
    n_list = list(args.n) + _parse_csv_ints(args.n_list)
    k_list = list(args.k) + _parse_csv_ints(args.k_list)

    if m_list or n_list or k_list:
        if not (m_list and n_list and k_list):
            raise ValueError("When using explicit m/n/k lists, provide all of --m/--n/--k or --m-list/--n-list/--k-list")
        if args.mnk_grid:
            for m in m_list:
                for n in n_list:
                    for k in k_list:
                        shapes.append((m, n, k))
        else:
            if not (len(m_list) == len(n_list) == len(k_list)):
                raise ValueError("Without --mnk-grid, m/n/k lists must have the same length")
            shapes.extend(zip(m_list, n_list, k_list))

    if not shapes:
        shapes = [(1024, 4096, 4096)]
    # preserve order, drop duplicates
    dedup: List[Tuple[int, int, int]] = []
    seen = set()
    for s in shapes:
        if s not in seen:
            dedup.append(s)
            seen.add(s)
    return dedup


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    warmup: int,
    repeat: int,
) -> TimingStats:
    device = a.device

    _synchronize(device)
    t0 = time.perf_counter()
    out = fn(a, b)
    _synchronize(device)
    first_run_ms = (time.perf_counter() - t0) * 1000.0

    for _ in range(warmup):
        out = fn(a, b)
    _synchronize(device)

    samples: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(a, b)
        _synchronize(device)
        samples.append((time.perf_counter() - t0) * 1000.0)

    if not samples:
        raise RuntimeError("No timing samples collected")
    samples_sorted = sorted(samples)
    median_ms = statistics.median(samples)
    mean_ms = statistics.mean(samples)
    stdev_ms = statistics.stdev(samples) if len(samples) >= 2 else 0.0
    p90_ms = samples_sorted[min(len(samples_sorted) - 1, math.ceil(0.90 * len(samples_sorted)) - 1)]
    p95_ms = samples_sorted[min(len(samples_sorted) - 1, math.ceil(0.95 * len(samples_sorted)) - 1)]
    return TimingStats(
        first_run_ms=first_run_ms,
        median_ms=median_ms,
        mean_ms=mean_ms,
        min_ms=min(samples),
        max_ms=max(samples),
        stdev_ms=stdev_ms,
        p90_ms=p90_ms,
        p95_ms=p95_ms,
        tflops=0.0,
    )


def _tflops(m: int, n: int, k: int, median_ms: float) -> float:
    ops = 2.0 * float(m) * float(n) * float(k)
    sec = median_ms / 1000.0
    if sec <= 0.0:
        return float("nan")
    return ops / sec / 1.0e12


def _mode_to_callable(mode: str) -> Tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Optional[Dict[str, Any]]]:
    module = MMModule().eval().cuda()
    if mode == "eager":
        return module.forward, None
    if mode == "inductor":
        compiled = torch.compile(module, backend="inductor", fullgraph=True)
        return compiled, None
    if mode == "hybrid":
        backend = _import_hybrid_backend()
        compiled = torch.compile(module, backend=backend, fullgraph=True)
        debug = getattr(compiled, "_hybrid_mm_debug", None)
        if debug is not None:
            try:
                debug = asdict(debug)
            except Exception:
                debug = {"repr": repr(debug)}
        return compiled, debug
    raise ValueError(f"Unsupported mode: {mode}")


def _compare(ref: torch.Tensor, out: torch.Tensor) -> Tuple[float, float]:
    diff = (out - ref).abs()
    max_abs = float(diff.max().item())
    denom = ref.abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item())
    return max_abs, max_rel


def _write_csv(path: str, rows: Sequence[ModeResult]) -> None:
    import csv

    if not path:
        return
    fieldnames = list(ModeResult.__annotations__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            d = asdict(row)
            d["shape"] = list(row.shape)
            writer.writerow(d)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print(json.dumps({"status": "FAIL", "reason": "CUDA is not available"}, indent=2))
        return 1

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = DTYPE_MAP[args.dtype]
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    shapes = resolve_shapes(args)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    report: Dict[str, Any] = {
        "backend_diagnostics": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "TRITON_SHARED_ENABLE_AMX": os.environ.get("TRITON_SHARED_ENABLE_AMX"),
            "TRITON_SHARED_OPT_PATH": os.environ.get("TRITON_SHARED_OPT_PATH"),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "requested": {
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "modes": modes,
            "shapes": shapes,
            "tag": args.tag,
        },
    }

    all_rows: List[ModeResult] = []

    for (m, n, k) in shapes:
        a = torch.randn((m, k), device=device, dtype=dtype)
        b = torch.randn((k, n), device=device, dtype=dtype)

        reference = torch.mm(a, b)
        shape_rows: List[ModeResult] = []
        median_by_mode: Dict[str, float] = {}

        for mode in modes:
            debug: Optional[Dict[str, Any]] = None
            try:
                fn, debug = _mode_to_callable(mode)
                stats = _measure(fn, a, b, warmup=args.warmup, repeat=args.repeat)
                out = fn(a, b)
                _synchronize(device)
                max_abs, max_rel = (None, None) if args.skip_correctness else _compare(reference, out)
                stats.tflops = _tflops(m, n, k, stats.median_ms)
                row = ModeResult(
                    shape=(m, n, k),
                    dtype=args.dtype,
                    mode=mode,
                    status="PASS",
                    output_device=str(out.device),
                    max_abs=max_abs,
                    max_rel=max_rel,
                    first_run_ms=stats.first_run_ms,
                    median_ms=stats.median_ms,
                    mean_ms=stats.mean_ms,
                    min_ms=stats.min_ms,
                    max_ms=stats.max_ms,
                    stdev_ms=stats.stdev_ms,
                    p90_ms=stats.p90_ms,
                    p95_ms=stats.p95_ms,
                    tflops=stats.tflops,
                    speedup_vs_eager=None,
                    speedup_vs_inductor=None,
                    note=None,
                    debug=debug,
                )
                median_by_mode[mode] = stats.median_ms
            except Exception as exc:
                row = ModeResult(
                    shape=(m, n, k),
                    dtype=args.dtype,
                    mode=mode,
                    status="FAIL",
                    output_device=None,
                    max_abs=None,
                    max_rel=None,
                    first_run_ms=None,
                    median_ms=None,
                    mean_ms=None,
                    min_ms=None,
                    max_ms=None,
                    stdev_ms=None,
                    p90_ms=None,
                    p95_ms=None,
                    tflops=None,
                    speedup_vs_eager=None,
                    speedup_vs_inductor=None,
                    note=f"{type(exc).__name__}: {exc}",
                    debug=debug,
                )
            shape_rows.append(row)

        eager_ms = median_by_mode.get("eager")
        inductor_ms = median_by_mode.get("inductor")
        for row in shape_rows:
            if row.status != "PASS" or row.median_ms is None:
                continue
            if eager_ms and row.median_ms > 0:
                row.speedup_vs_eager = eager_ms / row.median_ms
            if inductor_ms and row.median_ms > 0:
                row.speedup_vs_inductor = inductor_ms / row.median_ms

        all_rows.extend(shape_rows)

    report["results"] = [asdict(r) for r in all_rows]
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(text)
    if args.csv_out:
        _write_csv(args.csv_out, all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
