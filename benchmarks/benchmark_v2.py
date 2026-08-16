"""Repeatable benchmark for the public VEQPy dictionary and Module paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import veqpy
from veqpy.demo_case import make_demo_inputs


def topology() -> dict[str, object]:
    """Return the ordinary topology mapping used by the benchmark."""

    return {
        "h_count": 3,
        "v_count": 3,
        "kappa_count": 3,
        "psin_count": 6,
        "F_count": 0,
        "c_counts": (3, 3, 3),
        "s_counts": (3, 3),
        "Nr": 8,
        "Nt": 12,
        "route": "PF",
        "coordinate": "psin",
        "constraint": "ip",
        "quadrature": "legendre",
        "calculus": "spectral",
    }


def measure(backend: str, repeats: int) -> dict[str, object]:
    """Measure repeated non-materializing solves on one prepared Module."""

    module = veqpy.build(topology=topology(), backend=backend, verbose=False)
    try:
        boundary, source, targets = make_demo_inputs()
        rows = []
        for _ in range(repeats):
            started = perf_counter()
            record = module.solve(
                boundary=boundary,
                source=source,
                targets=targets,
                materialize=False,
                verbose=False,
            )
            rows.append(
                {
                    "elapsed_ms": (perf_counter() - started) * 1000.0,
                    "record_elapsed_ms": record.elapsed_ms,
                    "accepted": record.accepted,
                    "residual_norm": record.residual_norm,
                    "evaluations": record.evaluations,
                }
            )
        return {"backend": backend, "repeats": repeats, "rows": rows}
    finally:
        module.close()


def main() -> int:
    """Run the selected backend benchmark."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("numba", "cxx", "cxx-strict", "cxx-relaxed", "cxx-enzyme"),
        default="numba",
    )
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    try:
        result = measure(args.backend, args.repeat)
    except Exception as error:
        print(json.dumps({"backend": args.backend, "status": "unavailable", "error": str(error)}))
        return 2 if args.backend != "numba" else 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
