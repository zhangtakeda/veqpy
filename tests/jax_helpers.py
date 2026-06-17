from __future__ import annotations

import numpy as np
from helpers import tiny_boundary

from veqpy.model import Grid, Problem


def tiny_pf_rho_grid_problem(grid: Grid) -> Problem:
    return Problem(
        route="PF",
        coordinate="rho",
        nodes="grid",
        active_profiles={"h": 1, "k": 1, "s1": 1},
        boundary=tiny_boundary(),
        heat_input=np.full(grid.Nr, 1.0e6, dtype=np.float64),
        current_input=np.ones(grid.Nr, dtype=np.float64),
        Ip=3.0e6,
    )
