#!/usr/bin/env python3
import json
from contextlib import contextmanager
from typing import Any

import torch

from _inductor.hybrid_mm_eager_bridge import hybrid_mm_eager_bridge_backend


def _json_default(obj: Any):
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    return repr(obj)


def has_triton_shared_backend() -> bool:
    try:
        import triton
    except Exception:
        return False
    return "triton_shared" in getattr(triton.backends, "backends", {})


class TritonCompileObserver:
    def __init__(self):
        self.cpu_target_seen = False
        self.cuda_target_seen = False
        self.calls = []

    def record(self, target):
        target_repr = repr(target)
        self.calls.append(target_repr)
        if "backend='cpu'" in target_repr:
            self.cpu_target_seen = True
        if "backend='cuda'" in target_repr:
            self.cuda_target_seen = True


@contextmanager
def observe_triton_compile():
    import triton

    observer = TritonCompileObserver()
    original = triton.compile

    def wrapped(*args, **kwargs):
        target = kwargs.get("target")
        if target is None:
            for arg in args:
                rep = repr(arg)
                if "backend='cpu'" in rep or "backend='cuda'" in rep:
                    target = arg
                    break
        observer.record(target)
        return original(*args, **kwargs)

    triton.compile = wrapped
    try:
        yield observer
    finally:
        triton.compile = original


def make_mm_graph_module() -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    lhs = graph.placeholder("lhs")
    rhs = graph.placeholder("rhs")
    out = graph.call_function(torch.mm, args=(lhs, rhs))
    graph.output(out)
    return torch.fx.GraphModule({}, graph)


def main() -> int:
    diagnostics = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "has_triton_shared_backend": has_triton_shared_backend(),
        "TRITON_SHARED_ENABLE_AMX": __import__("os").environ.get("TRITON_SHARED_ENABLE_AMX"),
        "TRITON_SHARED_OPT_PATH": __import__("os").environ.get("TRITON_SHARED_OPT_PATH"),
    }
    print(json.dumps({"backend_diagnostics": diagnostics}, indent=2, default=_json_default))

    if not torch.cuda.is_available():
        print(json.dumps([{"test": "hybrid_mm_eager_bridge_backend", "status": "SKIP", "reason": "CUDA is not available"}], indent=2))
        return 0
    if not has_triton_shared_backend():
        print(json.dumps([{"test": "hybrid_mm_eager_bridge_backend", "status": "SKIP", "reason": "triton_shared backend not registered"}], indent=2))
        return 0

    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    gm = make_mm_graph_module()

    with observe_triton_compile() as observer:
        compiled = hybrid_mm_eager_bridge_backend(gm, [a, b])
        out = compiled(a, b)

    ref = torch.mm(a, b)
    max_abs = float((out - ref).abs().max().item())
    debug = getattr(compiled, "_hybrid_mm_debug")
    cuda_path_inferred = (
        out.device.type == "cuda"
        and out.shape == ref.shape
        and "torch.mm" in (debug.gpu_graph or "")
    )
    ok = (
        debug.mode == "hybrid_mm_eager_bridge"
        and observer.cpu_target_seen
        and cuda_path_inferred
        and max_abs < 1e-3
        and "torch.mm" in (debug.cpu_graph or "")
        and "torch.mm" in (debug.gpu_graph or "")
        and "to(" not in (debug.cpu_graph or "")
        and "to(" not in (debug.gpu_graph or "")
    )

    results = [{
        "test": "hybrid_mm_eager_bridge_backend",
        "status": "PASS" if ok else "FAIL",
        "cpu_target_seen": observer.cpu_target_seen,
        "cuda_target_seen": observer.cuda_target_seen,
        "cuda_path_inferred": cuda_path_inferred,
        "compile_calls": observer.calls,
        "debug": {
            "mode": debug.mode,
            "cpu_n": debug.cpu_n,
            "total_n": debug.total_n,
            "cpu_graph": debug.cpu_graph,
            "gpu_graph": debug.gpu_graph,
            "cpu_compile_options": debug.cpu_compile_options,
            "gpu_compile_options": debug.gpu_compile_options,
        },
        "output_device": str(out.device),
        "output_shape": list(out.shape),
        "max_abs": max_abs,
        "note": "PASS does not require observer.cuda_target_seen; CUDA path is inferred from CUDA output plus GPU shard graph.",
    }]
    print(json.dumps(results, indent=2, default=_json_default))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
