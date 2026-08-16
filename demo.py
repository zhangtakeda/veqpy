"""Minimal external-user demo for the VEQPy 2.x dictionary API."""

from __future__ import annotations

import veqpy
from veqpy.demo_case import make_demo_inputs

TOPOLOGY = {
    "Nr": 8,
    "Nt": 12,
    "route": "PF",
    "coordinate": "psin",
    "constraint": "ip",
    "h_count": 3,
    "v_count": 3,
    "kappa_count": 3,
    "psin_count": 6,
    "F_count": 0,
    "c_counts": (3, 3, 3),
    "s_counts": (3, 3),
    "quadrature": "legendre",
    "calculus": "spectral",
    "L_max": 5,
    "M_max": 2,
    "K_max": 2,
}


def main() -> int:
    """Run one solve using only ordinary mappings."""

    module = veqpy.build(topology=TOPOLOGY, backend="numba")
    try:
        boundary, source, targets = make_demo_inputs()
        result = module.solve(boundary=boundary, source=source, targets=targets)
    finally:
        module.close()

    print("VEQPy minimal Module demo")
    print(f"success: {result.accepted}")
    print(f"residual: {result.residual_norm:.3e}")
    print(f"nfev: {result.evaluations}")
    print(f"source capacity: {result.source_capacity} (epoch {result.capacity_epoch})")
    return 0 if result.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
