"""Small repeatable benchmark for the VEQPy 2.x Module and Kernel paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import veqpy
from veqpy.demo_case import make_demo_plasma


def topology() -> veqpy.KernelTopology:
    return veqpy.KernelTopology(
        h_count=3,
        v_count=3,
        kappa_count=3,
        psin_count=6,
        F_count=0,
        c_counts=(3, 3, 3),
        s_counts=(3, 3),
        Nr=16,
        Nt=16,
        route="PF",
        coordinate="psin",
        nodes="uniform",
        constraint="ip",
        sample_count=51,
    )


def measure(backend: str, repeats: int) -> dict[str, object]:
    module = veqpy.VEQ(topology=topology(), backend=backend)
    try:
        plasma = make_demo_plasma()
        rows = []
        for _ in range(repeats):
            started = perf_counter()
            record = module.run(plasma=plasma, materialize=False)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("numba", "cxx"), default="numba")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    try:
        result = measure(args.backend, args.repeat)
    except Exception as error:
        print(json.dumps({"backend": args.backend, "status": "unavailable", "error": str(error)}))
        return 2 if args.backend == "cxx" else 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
