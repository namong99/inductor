#!/usr/bin/env python3
import json

import torch

from _inductor.hybrid_mm_eager_bridge import hybrid_mm_eager_bridge_backend


def make_non_mm_graph_module() -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    lhs = graph.placeholder("lhs")
    rhs = graph.placeholder("rhs")
    out = graph.call_function(torch.add, args=(lhs, rhs))
    graph.output(out)
    return torch.fx.GraphModule({}, graph)


def main() -> int:
    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    gm = make_non_mm_graph_module()
    compiled = hybrid_mm_eager_bridge_backend(gm, [x, y])
    out = compiled(x, y)
    ref = x + y
    debug = getattr(compiled, "_hybrid_mm_debug")
    ok = debug.mode == "fallback" and torch.allclose(out, ref)
    print(json.dumps([{
        "test": "hybrid_mm_eager_bridge_fallback",
        "status": "PASS" if ok else "FAIL",
        "mode": debug.mode,
        "max_abs": float((out - ref).abs().max().item()),
    }], indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
