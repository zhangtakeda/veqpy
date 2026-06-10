"""Build the compact demo-equilibrium figure used by the manuscript.

The solve uses the public VEQPy API and delegates layout to
``Equilibrium.plot(plot_all=False)``.  Only the output paths and warmup count
are script-local.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from config import (
    FIGURE_FACE_COLOR,
    MU0,
    SAVE_DPI,
    SAVE_TRANSPARENT,
    demo_psin_reference_profiles,
    save_figure_outputs,
)

from veqpy.model import Boundary, Grid
from veqpy.operator import Operator, OperatorCase
from veqpy.solver import Solver, SolverConfig

PNG_PATH = "03-demo-equilibrium.png"
PDF_PATH = None

WARMUP_TIMES = 10
SOURCE_SAMPLE_COUNT = 51

GRID = Grid(
    Nr=64,
    Nt=64,
    quadrature_scheme="legendre",
)
SNAPSHOT_GRID = Grid(
    Nr=128,
    Nt=256,
    quadrature_scheme="uniform",
    L_max=GRID.L_max,
    M_max=GRID.M_max,
)
BOUNDARY = Boundary(
    a=1.05 / 1.85,
    R0=1.05,
    Z0=0.0,
    B0=3.0,
    ka=2.2,
    s_offsets=np.array([0.0, float(np.arcsin(0.5))]),
)
CONFIG = SolverConfig(
    method="hybr",
    enable_verbose=False,
    enable_warmstart=False,
)
COEFFS = {
    "psin": [0.0] * 5,
    "h": [0.0] * 3,
    "k": [0.0] * 5,
    "s1": [0.0] * 3,
}


def warmup_solver(solver: Solver) -> None:
    """Run silent solves so output timings exclude first-call startup overhead."""
    for _ in range(WARMUP_TIMES):
        solver.solve(enable_verbose=False, enable_history=False)


def build_operator_case() -> OperatorCase:
    """Construct the PF operator case rendered in this figure."""
    psin = np.linspace(0.0, 1.0, SOURCE_SAMPLE_COUNT)
    current_input, heat_input = demo_psin_reference_profiles(psin)
    return OperatorCase(
        route="PF",
        profile_coeffs=COEFFS,
        boundary=BOUNDARY,
        heat_input=heat_input / MU0,
        current_input=current_input,
        coordinate="psin",
        nodes="uniform",
        Ip=3.0e6,
    )


def solve_reference_equilibrium() -> object:
    """Solve the reference equilibrium and return a plotting-ready snapshot."""
    operator = Operator(grid=GRID, case=build_operator_case())
    solver = Solver(operator=operator, config=CONFIG)
    warmup_solver(solver)
    solver.solve()
    return solver.build_equilibrium().resample(SNAPSHOT_GRID)


def build_equilibrium_figure(equilibrium) -> plt.Figure:
    """Build the standalone equilibrium visualization with VEQPy's internal plotter."""
    return equilibrium.plot(outpath=None, plot_all=False)


def main() -> None:
    equilibrium = solve_reference_equilibrium()
    fig = build_equilibrium_figure(equilibrium)

    saved_paths = save_figure_outputs(
        fig,
        png_path=PNG_PATH,
        pdf_path=PDF_PATH,
        dpi=SAVE_DPI,
        transparent=SAVE_TRANSPARENT,
        facecolor=FIGURE_FACE_COLOR,
    )
    plt.close(fig)

    for path in saved_paths:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
