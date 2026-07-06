"""No-argument GEQDSK-to-Kernel workflow demo.

Run it directly to fit a GEQDSK boundary, solve a small Kernel API equilibrium,
and write a comparison figure plus a serialized ``Equilibrium`` snapshot.

Set ``VEQPY_GEQDSK`` to choose another input file. Set ``VEQPY_OUTPUT_DIR`` to
choose the output directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from veqlib import Kernel, KernelBoundary, KernelConfig, KernelRecipe, KernelSource, KernelTopology
from veqpy.model import Boundary, Geqdsk, Grid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEQDSK = PROJECT_ROOT / "data" / "SOLOVEV.geqdsk"


def ensure_output_dir() -> Path:
    env_out = os.environ.get("VEQPY_OUTPUT_DIR")
    outdir = Path(env_out) if env_out else Path.cwd() / "outputs" / "geqdsk_workflow"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def resolve_geqdsk_path() -> Path:
    env_path = os.environ.get("VEQPY_GEQDSK")
    return Path(env_path).expanduser().resolve() if env_path else DEFAULT_GEQDSK


def fit_kernel_boundary(geqdsk: Geqdsk) -> KernelBoundary:
    fitted = Boundary.from_geqdsk(geqdsk, M=1, N=1, maxtol=1.0)
    return KernelBoundary(
        a=fitted.a,
        R0=fitted.R0,
        Z0=fitted.Z0,
        B0=fitted.B0,
        ka=fitted.ka,
        c_offsets=fitted.c_offsets,
        s_offsets=fitted.s_offsets,
    )


def build_topology(geqdsk: Geqdsk) -> KernelTopology:
    sample_count = int(max(geqdsk.P_psi.size, geqdsk.FF_psi.size, 9))
    return KernelTopology(
        h_count=2,
        v_count=0,
        kappa_count=2,
        psin_count=3,
        F_count=0,
        c_counts=(),
        s_counts=(2,),
        Nr=8,
        Nt=8,
        route="PF",
        coordinate="psin",
        nodes="uniform",
        ip_constraint=True,
        sample_count=sample_count,
    )


def build_source(geqdsk: Geqdsk, sample_count: int) -> KernelSource:
    heat_profile = _profile_or_default(geqdsk.P_psi, sample_count, scale=1.0e6)
    current_profile = _profile_or_default(geqdsk.FF_psi, sample_count, scale=1.0)
    return KernelSource(
        heat_profile=heat_profile,
        current_profile=current_profile,
        Ip=abs(float(geqdsk.Ip)) if geqdsk.Ip else 1.0e6,
        case_name=Path(resolve_geqdsk_path()).stem,
    )


def _profile_or_default(values: np.ndarray, sample_count: int, *, scale: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1 and arr.size == sample_count and np.all(np.isfinite(arr)):
        return arr
    axis = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    return scale * (1.0 - axis)


def build_surface_from_psin(equilibrium, level: float) -> np.ndarray:
    psin = np.asarray(equilibrium.psin, dtype=np.float64)
    rho = np.asarray(equilibrium.rho, dtype=np.float64)
    order = np.argsort(psin)
    psin_unique, unique_idx = np.unique(psin[order], return_index=True)
    rho_level = float(np.interp(float(level), psin_unique, rho[order][unique_idx]))
    r_values = np.array(
        [np.interp(rho_level, rho, equilibrium.R[:, idx]) for idx in range(equilibrium.grid.Nt)],
        dtype=np.float64,
    )
    z_values = np.array(
        [np.interp(rho_level, rho, equilibrium.Z[:, idx]) for idx in range(equilibrium.grid.Nt)],
        dtype=np.float64,
    )
    return np.column_stack((r_values, z_values))


def close_curve(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected an (N, 2) curve, got {arr.shape}")
    return np.vstack((arr, arr[:1]))


def compute_rz_limits(curves: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    stacked = np.vstack([np.asarray(curve, dtype=np.float64) for curve in curves if curve.size])
    r_min = float(np.min(stacked[:, 0]))
    r_max = float(np.max(stacked[:, 0]))
    z_min = float(np.min(stacked[:, 1]))
    z_max = float(np.max(stacked[:, 1]))
    r_pad = max((r_max - r_min) * 0.07, 1.0e-3)
    z_pad = max((z_max - z_min) * 0.07, 1.0e-3)
    return (r_min - r_pad, r_max + r_pad), (z_min - z_pad, z_max + z_pad)


def plot_workflow_summary(
    figure_path: Path,
    *,
    geqdsk: Geqdsk,
    equilibrium,
    result,
) -> None:
    levels = tuple(np.linspace(0.15, 1.0, 8, dtype=np.float64))
    surfaces = [build_surface_from_psin(equilibrium, float(level)) for level in levels]
    boundary = np.asarray(geqdsk.boundary, dtype=np.float64)
    rz_limits = compute_rz_limits([*surfaces, boundary])

    fig, (ax_shape, ax_source) = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.4),
        constrained_layout=True,
    )
    colors = plt.cm.viridis(np.linspace(0.2, 0.92, len(surfaces)))
    for surface, color in zip(surfaces, colors, strict=True):
        curve = close_curve(surface)
        ax_shape.plot(curve[:, 0], curve[:, 1], color=color, linewidth=1.0)
    if boundary.size:
        target = close_curve(boundary)
        ax_shape.plot(target[:, 0], target[:, 1], color="black", linewidth=1.4, label="GEQDSK")
    ax_shape.scatter([geqdsk.Raxis], [geqdsk.Zaxis], marker="x", color="#d62728", s=36)
    ax_shape.set_title("Kernel equilibrium")
    ax_shape.set_xlabel("R [m]")
    ax_shape.set_ylabel("Z [m]")
    ax_shape.set_xlim(*rz_limits[0])
    ax_shape.set_ylim(*rz_limits[1])
    ax_shape.set_aspect("equal", adjustable="box")
    ax_shape.grid(True, linestyle=":", alpha=0.35)
    ax_shape.legend(loc="upper right")

    psin_axis = np.linspace(0.0, 1.0, geqdsk.P_psi.size, dtype=np.float64)
    ax_source.plot(psin_axis, np.asarray(geqdsk.P_psi, dtype=np.float64), label="P_psi")
    ax_source.plot(psin_axis, np.asarray(geqdsk.FF_psi, dtype=np.float64), label="FF_psi")
    ax_source.set_title(f"raw source profiles; success={result.success}")
    ax_source.set_xlabel("psin")
    ax_source.set_ylabel("GEQDSK value")
    ax_source.grid(True, linestyle=":", alpha=0.35)
    ax_source.legend(loc="best")

    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def main() -> None:
    outdir = ensure_output_dir()
    figure_path = outdir / "demo_geqdsk_workflow.png"
    equilibrium_path = outdir / "demo_geqdsk_equilibrium.json"

    geqdsk_path = resolve_geqdsk_path()
    geqdsk = Geqdsk(geqdsk_path)
    boundary = fit_kernel_boundary(geqdsk)
    topology = build_topology(geqdsk)
    source = build_source(geqdsk, topology.sample_count)
    kernel = Kernel(
        topology=topology,
        recipe=KernelRecipe(backend="numba", layout="degree"),
        config=KernelConfig(
            method="levenberg-marquardt",
            initial="cold-zeros",
            norm="none",
            max_evaluations=16,
        ),
    )

    result = kernel.solve(boundary, source)
    equilibrium = kernel.build_equilibrium()
    plot_grid = Grid(
        Nr=64,
        Nt=128,
        quadrature_scheme="uniform",
        L_max=topology.L_max,
        M_max=topology.M_max,
        K_max=topology.K_max,
    )
    plot_equilibrium = equilibrium.resample(grid=plot_grid)
    plot_workflow_summary(
        figure_path,
        geqdsk=geqdsk,
        equilibrium=plot_equilibrium,
        result=result,
    )
    equilibrium.write(str(equilibrium_path))

    print("VEQPy GEQDSK Kernel workflow")
    print(f"  input: {geqdsk_path}")
    print(f"  saved figure: {figure_path}")
    print(f"  saved equilibrium: {equilibrium_path}")
    print(f"  success: {result.success}")
    print(f"  raw_norm: {result.raw_norm:.6e}")
    print(f"  Ip [MA]: {equilibrium.Ip / 1.0e6:.6f}")
    kernel.close()


if __name__ == "__main__":
    main()
