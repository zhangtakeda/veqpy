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
    CASE_REFERENCE_PROFILE_LENGTHS,
    FIGURE_FACE_COLOR,
    MU0,
    SAVE_DPI,
    SAVE_TRANSPARENT,
    SCRIPT_CONSOLE,
    active_profiles_from_coeffs,
    demo_psin_reference_profiles,
    figure_path,
    format_script_sci,
    make_script_table,
    print_output_table,
    print_script_config,
    print_script_table,
    save_figure_outputs,
    script_progress,
)

from veqpy.model import Boundary, Grid, Problem
from veqpy.operator import Operator
from veqpy.solver import Solver, SolverConfig

PNG_PATH = figure_path("03-demo-equilibrium.png")
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
)
COEFFS = CASE_REFERENCE_PROFILE_LENGTHS["demo(psin)"]


def warmup_solver(solver: Solver) -> None:
    """Run silent solves so output timings exclude first-call startup overhead."""
    for _ in range(WARMUP_TIMES):
        solver.solve(enable_verbose=False, enable_history=False)


def build_problem() -> Problem:
    """Construct the PF problem rendered in this figure."""
    psin = np.linspace(0.0, 1.0, SOURCE_SAMPLE_COUNT)
    current_input, heat_input = demo_psin_reference_profiles(psin)
    return Problem(
        route="PF",
        active_profiles=active_profiles_from_coeffs(COEFFS),
        boundary=BOUNDARY,
        heat_input=heat_input / MU0,
        current_input=current_input,
        coordinate="psin",
        nodes="uniform",
        Ip=3.0e6,
    )


def solve_reference_equilibrium() -> object:
    """Solve the reference equilibrium and return a plotting-ready snapshot."""
    operator = Operator(grid=GRID, case=build_problem())
    solver = Solver(operator=operator, config=CONFIG)
    warmup_solver(solver)
    solver.solve()
    return solver.build_equilibrium().resample(SNAPSHOT_GRID)


def build_equilibrium_figure(equilibrium) -> plt.Figure:
    """Build the standalone equilibrium visualization with VEQPy's internal plotter."""
    return equilibrium.plot(outpath=None, plot_all=False)


def main() -> None:
    print_script_config(
        SCRIPT_CONSOLE,
        "figure 03: controlled PF reference equilibrium",
        (
            ("route", "PF(psin, uniform)"),
            ("solve grid", f"{GRID.Nr}x{GRID.Nt}"),
            ("snapshot grid", f"{SNAPSHOT_GRID.Nr}x{SNAPSHOT_GRID.Nt}"),
        ),
    )
    with script_progress(SCRIPT_CONSOLE) as progress:
        task = progress.add_task("", total=3, current="solve", phase="[cyan]run[/]")
        equilibrium = solve_reference_equilibrium()
        progress.update(task, advance=1, current="plot", phase="[cyan]run[/]")
        fig = build_equilibrium_figure(equilibrium)
        progress.update(task, advance=1, current="save", phase="[cyan]run[/]")

        saved_paths = save_figure_outputs(
            fig,
            png_path=PNG_PATH,
            pdf_path=PDF_PATH,
            dpi=SAVE_DPI,
            transparent=SAVE_TRANSPARENT,
            facecolor=FIGURE_FACE_COLOR,
        )
        progress.update(task, advance=1, current="save", phase="[green]done[/]")
    plt.close(fig)

    summary = make_script_table(
        "controlled PF reference equilibrium",
        [("quantity", "left"), ("value", "right")],
    )
    q95 = float(np.interp(0.95, np.asarray(equilibrium.psin), np.asarray(equilibrium.q)))
    summary.add_row("Ip [A]", format_script_sci(float(equilibrium.Ip)))
    summary.add_row("beta_t", format_script_sci(float(equilibrium.beta_t)))
    summary.add_row("q95", format_script_sci(q95))
    print_script_table(SCRIPT_CONSOLE, summary)
    print_output_table(
        SCRIPT_CONSOLE,
        [("Figure 03", path, "Controlled PF reference equilibrium") for path in saved_paths],
    )


if __name__ == "__main__":
    main()
