import argparse
import contextlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def get_cpu_model() -> str:
    for cmd in (
        ["bash", "-lc", "lscpu | sed -n 's/^Model name:\\s*//p' | head -1"],
        ["bash", "-lc", "grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //'"],
    ):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
            if out:
                return out
        except Exception:
            pass
    p = platform.processor().strip()
    return p or "unknown"


def backend_diagnostics(torch) -> Dict[str, Any]:
    diag = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_model": get_cpu_model(),
        "TRITON_SHARED_ENABLE_AMX": os.environ.get("TRITON_SHARED_ENABLE_AMX"),
        "TRITON_SHARED_OPT_PATH": os.environ.get("TRITON_SHARED_OPT_PATH"),
    }
    try:
        diag["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except Exception:
        diag["float32_matmul_precision"] = None
    try:
        diag["allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        diag["allow_tf32"] = None
    try:
        import triton  # noqa: F401
        diag["triton_import_ok"] = True
        diag["triton_version"] = getattr(__import__("triton"), "__version__", "unknown")
        try:
            from triton.backends import backends as _triton_backends
            keys = sorted(list(getattr(_triton_backends, "keys", lambda: [])()))
            diag["registered_triton_backends"] = keys
            diag["has_triton_shared_backend"] = "triton_shared" in keys
        except Exception:
            diag["registered_triton_backends"] = None
            diag["has_triton_shared_backend"] = None
    except Exception:
        diag["triton_import_ok"] = False
        diag["triton_version"] = None
        diag["registered_triton_backends"] = None
        diag["has_triton_shared_backend"] = None
    return diag


@dataclass
class Stats:
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    p90_ms: float
    p95_ms: float


def compute_stats(values_ms: Sequence[float]) -> Stats:
    vals = list(values_ms)
    vals_sorted = sorted(vals)
    n = len(vals_sorted)

    def percentile(p: float) -> float:
        if n == 1:
            return vals_sorted[0]
        idx = (n - 1) * p
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return vals_sorted[lo]
        frac = idx - lo
        return vals_sorted[lo] * (1.0 - frac) + vals_sorted[hi] * frac

    return Stats(
        median_ms=statistics.median(vals),
        mean_ms=statistics.mean(vals),
        min_ms=min(vals),
        max_ms=max(vals),
        stdev_ms=statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        p90_ms=percentile(0.90),
        p95_ms=percentile(0.95),
    )


@contextlib.contextmanager
def patch_inductor_config(torch, updates: Dict[str, Any]):
    cfg = torch._inductor.config
    prior = {}
    try:
        for k, v in updates.items():
            if hasattr(cfg, k):
                prior[k] = getattr(cfg, k)
                setattr(cfg, k, v)
        yield
    finally:
        for k, v in prior.items():
            setattr(cfg, k, v)


class TritonCompileObserver:
    def __init__(self):
        self.calls: List[str] = []
        self.cpu_target_seen = False
        self.cuda_target_seen = False
        self._orig = None
        self._module = None

    def __enter__(self):
        try:
            import triton
            self._module = triton
            self._orig = triton.compile
        except Exception:
            return self

        def wrapped(*args, **kwargs):
            target = kwargs.get("target", None)
            target_repr = repr(target)
            self.calls.append(target_repr)
            if "backend='cpu'" in target_repr:
                self.cpu_target_seen = True
            if "backend='cuda'" in target_repr:
                self.cuda_target_seen = True
            return self._orig(*args, **kwargs)

        self._module.compile = wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._module is not None and self._orig is not None:
            self._module.compile = self._orig
        return False


def compile_mm(torch, device: str, lhs, rhs, *, force_triton_shared: bool = False):
    def mm_fn(x, y):
        return torch.mm(x, y)

    if device == "cpu":
        updates = {
            "cpu_backend": "triton_shared",
            "force_triton_shared_mm": True if force_triton_shared else getattr(torch._inductor.config, "force_triton_shared_mm", False),
            "max_autotune": True,
            "max_autotune_gemm": True,
            "max_autotune_gemm_backends": "TRITON",
            "force_disable_caches": True,
        }
    else:
        updates = {
            "force_disable_caches": True,
        }

    with patch_inductor_config(torch, updates):
        compiled = torch.compile(mm_fn, backend="inductor")
        _ = compiled(lhs, rhs)
    return compiled


def sync_if_needed(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_region_ms(torch, fn):
    sync_if_needed(torch)
    t0 = time.perf_counter()
    out = fn()
    sync_if_needed(torch)
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0


def parse_shape(text: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"shape must be M,K,N but got: {text}")
    return parts[0], parts[1], parts[2]


def choose_cpu_n(n_total: int, ratio: float, align: int, min_chunk_n: int) -> int:
    ratio = max(0.0, min(1.0, ratio))
    if ratio <= 0.0:
        return 0
    if ratio >= 1.0:
        return n_total
    raw = int(math.floor(n_total * ratio))
    cpu_n = (raw // max(1, align)) * max(1, align)
    if cpu_n <= 0:
        cpu_n = max(align, min_chunk_n)
    cpu_n = max(cpu_n, min_chunk_n)
    if cpu_n >= n_total:
        cpu_n = n_total - min_chunk_n
    cpu_n = max(0, min(cpu_n, n_total))
    return cpu_n


def make_inputs(torch, m: int, k: int, n: int, dtype_str: str, device: str = "cuda"):
    dtype = getattr(torch, dtype_str)
    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)
    return a, b


def tflops(m: int, k: int, n: int, median_ms: float) -> float:
    flop = 2.0 * float(m) * float(k) * float(n)
    sec = median_ms / 1000.0
    return flop / sec / 1.0e12


def dump_json(path: Optional[str], payload: Any):
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
