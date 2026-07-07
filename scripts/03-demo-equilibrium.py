"""Build the compact demo-equilibrium figure used by the manuscript.

The solve uses the public VEQPy API and delegates layout to
``Equilibrium.plot(plot_all=False)``.  Only the output paths and warmup count
are script-local.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _cases import (
    CASE_REFERENCE_PROFILE_LENGTHS,
    MU0,
)
from _common import figure_path, save_figure_outputs
from _kernel_cases import (
    active_profiles_from_coeffs,
    demo_psin_reference_profiles,
)
from _plotting import (
    FIGURE_FACE_COLOR,
    SAVE_DPI,
    SAVE_TRANSPARENT,
)
from _reporting import (
    SCRIPT_CONSOLE,
    print_output_table,
    print_script_config,
    script_progress,
)

import veqpy as veq
from veqpy.model import Boundary, Grid

PNG_PATH = figure_path("03-demo-equilibrium.png")
PDF_PATH = None

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
COEFFS = CASE_REFERENCE_PROFILE_LENGTHS["demo(psin)"]


def build_kernel() -> tuple[veq.Kernel, veq.KernelBoundary, veq.KernelSource]:
    """Construct the PF Kernel case rendered in this figure."""
    psin = np.linspace(0.0, 1.0, SOURCE_SAMPLE_COUNT)
    current_input, heat_input = demo_psin_reference_profiles(psin)
    active = active_profiles_from_coeffs(COEFFS)
    topology = veq.KernelTopology(
        h_count=active.get("h", 0),
        v_count=active.get("v", 0),
        kappa_count=active.get("k", 0),
        psin_count=active.get("psin", 0),
        F_count=active.get("F", 0),
        c_counts=(),
        s_counts=(active.get("s1", 0),),
        Nr=GRID.Nr,
        Nt=GRID.Nt,
        route="PF",
        coordinate="psin",
        nodes="uniform",
        ip_constraint=True,
        sample_count=SOURCE_SAMPLE_COUNT,
        M_max=GRID.M_max,
        K_max=GRID.K_max,
    )
    kernel = veq.build(
        topology=topology,
        recipe=veq.KernelRecipe(backend="numba"),
        config=veq.KernelConfig(method="powell", initial="cold", continuation="cold"),
    )
    boundary = veq.KernelBoundary(
        a=BOUNDARY.a,
        R0=BOUNDARY.R0,
        Z0=BOUNDARY.Z0,
        B0=BOUNDARY.B0,
        ka=BOUNDARY.ka,
        c_offsets=BOUNDARY.c_offsets,
        s_offsets=BOUNDARY.s_offsets[1:],
    )
    source = veq.KernelSource(
        heat_profile=heat_input / MU0,
        current_profile=current_input,
        Ip=3.0e6,
    )
    return kernel, boundary, source


def solve_reference_equilibrium() -> object:
    """Solve the reference equilibrium and return a plotting-ready snapshot."""
    kernel, boundary, source = build_kernel()
    try:
        result = kernel.solve(boundary, source)
        if not result.success:
            raise RuntimeError(f"demo Kernel solve failed with residual {result.raw_norm:.3e}")
        return kernel.build_equilibrium().resample(SNAPSHOT_GRID)
    finally:
        kernel.close()


def build_equilibrium_figure(equilibrium) -> plt.Figure:
    """Build the standalone equilibrium visualization with VEQPy's internal plotter."""
    return equilibrium.plot(outpath=None, plot_all=False)


def main() -> None:
    print_script_config(
        SCRIPT_CONSOLE,
        "figure 03: demo equilibrium",
        (
            ("backend", "numba"),
            ("grid", f"{GRID.Nr}x{GRID.Nt}"),
            ("snapshot", f"{SNAPSHOT_GRID.Nr}x{SNAPSHOT_GRID.Nt}"),
            ("source samples", SOURCE_SAMPLE_COUNT),
        ),
    )
    with script_progress(SCRIPT_CONSOLE) as progress:
        task = progress.add_task("", total=2, current="solve kernel", phase="[cyan]solve[/]")
        equilibrium = solve_reference_equilibrium()
        progress.update(task, advance=1, current="render figure", phase="[cyan]run[/]")
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
        progress.update(task, advance=1, current="render figure", phase="[green]done[/]")

    print_output_table(
        SCRIPT_CONSOLE,
        [("Figure 03", path, "Compact demo equilibrium") for path in saved_paths],
    )


if __name__ == "__main__":
    main()
