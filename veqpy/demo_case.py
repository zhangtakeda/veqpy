"""Small dependency-free dictionary fixture used by the VEQPy CLI smoke path."""

from __future__ import annotations

import numpy as np


def make_demo_inputs() -> tuple[dict, dict, dict]:
    """Return boundary, source, and targets for a minimal PF/psin solve."""

    r = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    psin = r * r
    boundary = {
        "a": 0.9,
        "R0": 3.0,
        "Z0": 0.0,
        "B0": 5.0,
        "kappa_lcfs": 1.5,
        "c_lcfs": np.zeros(3, dtype=np.float64),
        "s_lcfs": np.zeros(2, dtype=np.float64),
    }
    source = {
        "psin": psin,
        "P_psin": np.full(8, -200_535.22829579, dtype=np.float64),
        "FF_psin": np.full(8, -0.042, dtype=np.float64),
        "P0": 1_000.0,
    }
    targets = {"Ip": 3_342_713.461377374}
    return boundary, source, targets


__all__ = ["make_demo_inputs"]
