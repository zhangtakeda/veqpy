"""Minimal no-argument VEQPy demo script.

Run it directly to get one solved equilibrium plus a simple flux-surface plot.
This is meant to be the smallest user-facing workflow example in the repo.

All inputs are declared in this file so the example remains independent from
the manuscript ``scripts/`` directory.

Note: The first run may be slower due to JIT compilation.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from veqpy.facade import Kernel, KernelBoundary, KernelConfig, KernelSource, KernelTopology
from veqpy.model import Grid

MU0 = 4.0e-7 * np.pi
SOURCE_SAMPLE_COUNT = 51
DEFAULT_LEVELS = tuple(np.linspace(0.1, 1.0, 10, dtype=np.float64))
DEFAULT_SURFACE_COLORS = tuple(plt.cm.viridis(np.linspace(0.18, 0.9, len(DEFAULT_LEVELS))))


def ensure_output_dir() -> Path:
    env_out = os.environ.get("VEQPY_OUTPUT_DIR")
    outdir = Path(env_out) if env_out else Path.cwd() / "outputs" / "minimal_equilibrium"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def build_demo_boundary() -> KernelBoundary:
    return KernelBoundary(
        a=1.05 / 1.85,
        R0=1.05,
        Z0=0.0,
        B0=3.0,
        ka=2.2,
        s_offsets=np.array([0.0, float(np.arcsin(0.5))], dtype=np.float64),
    )


def pf_reference_profiles(psin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75
    alpha_p = 5.0
    alpha_f = 3.32
    exp_ap = np.exp(alpha_p)
    exp_af = np.exp(alpha_f)
    den_p = 1.0 + exp_ap * (alpha_p - 1.0)
    den_f = 1.0 + exp_af * (alpha_f - 1.0)
    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    return current_input.astype(np.float64), heat_input.astype(np.float64)


def build_surface_from_psin(equilibrium, level: float) -> np.ndarray:
    psin = np.asarray(equilibrium.psin, dtype=np.float64)
    rho = np.asarray(equilibrium.rho, dtype=np.float64)
    order = np.argsort(psin)
    psin_unique, unique_idx = np.unique(psin[order], return_index=True)
    rho_level = float(np.interp(float(level), psin_unique, rho[order][unique_idx]))
    R = np.array(
        [np.interp(rho_level, rho, equilibrium.R[:, idx]) for idx in range(equilibrium.grid.Nt)],
        dtype=np.float64,
    )
    Z = np.array(
        [np.interp(rho_level, rho, equilibrium.Z[:, idx]) for idx in range(equilibrium.grid.Nt)],
        dtype=np.float64,
    )
    return np.column_stack((R, Z))


def close_curve(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected an (N, 2) curve, got {arr.shape}")
    return np.vstack((arr, arr[:1]))


def compute_rz_limits(
    curves: list[np.ndarray], *, pad_fraction: float = 0.06
) -> tuple[tuple[float, float], tuple[float, float]]:
    valid = []
    for curve in curves:
        arr = np.asarray(curve, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.size:
            valid.append(arr)
    if not valid:
        raise ValueError("At least one non-empty curve is required")
    stacked = np.vstack(valid)
    r_min = float(np.min(stacked[:, 0]))
    r_max = float(np.max(stacked[:, 0]))
    z_min = float(np.min(stacked[:, 1]))
    z_max = float(np.max(stacked[:, 1]))
    r_pad = max((r_max - r_min) * pad_fraction, 1.0e-3)
    z_pad = max((z_max - z_min) * pad_fraction, 1.0e-3)
    return (r_min - r_pad, r_max + r_pad), (z_min - z_pad, z_max + z_pad)


def plot_equilibrium_surfaces(
    ax: plt.Axes, equilibrium, *, levels: tuple[float, ...] = DEFAULT_LEVELS
) -> None:
    first_label = "VEQPy surfaces"
    for index, level in enumerate(levels):
        surface = build_surface_from_psin(equilibrium, float(level))
        color = DEFAULT_SURFACE_COLORS[min(index, len(DEFAULT_SURFACE_COLORS) - 1)]
        ax.plot(
            close_curve(surface)[:, 0],
            close_curve(surface)[:, 1],
            color=color,
            linewidth=1.1,
            label=first_label,
        )
        first_label = None
    boundary = build_surface_from_psin(equilibrium, 1.0)
    ax.plot(
        close_curve(boundary)[:, 0], close_curve(boundary)[:, 1], color="#111111", linewidth=1.8
    )


def style_surface_axis(
    ax: plt.Axes, *, title: str, rz_limits: tuple[tuple[float, float], tuple[float, float]]
) -> None:
    ax.set_title(title)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_xlim(*rz_limits[0])
    ax.set_ylim(*rz_limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.tick_params(direction="in", top=True, right=True)


def main() -> None:
    outdir = ensure_output_dir()
    figure_path = outdir / "demo_flux_surfaces.png"
    equilibrium_path = outdir / "demo_equilibrium.json"

    boundary = build_demo_boundary()
    psin = np.linspace(0.0, 1.0, SOURCE_SAMPLE_COUNT, dtype=np.float64)
    current_input, heat_input = pf_reference_profiles(psin)
    pressure_input = heat_input / MU0
    topology = KernelTopology(
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
        sample_count=SOURCE_SAMPLE_COUNT,
    )
    source = KernelSource(
        heat_profile=pressure_input,
        current_profile=current_input,
        Ip=3.0e6,
    )
    plot_grid = Grid(
        Nr=128,
        Nt=256,
        quadrature_scheme="uniform",
        L_max=topology.L_max,
        M_max=topology.M_max,
        K_max=topology.K_max,
    )
    kernel = Kernel(
        topology=topology,
        config=KernelConfig(
            method="levenberg-marquardt",
            initial="cold-zeros",
            norm="none",
            max_evaluations=16,
        ),
    )

    result = kernel.solve(boundary, source)
    print(result)
    equilibrium = kernel.build_equilibrium()
    equilibrium.plot(outdir / "demo_equilibrium.png", grid=plot_grid)
    plot_equilibrium = equilibrium.resample(grid=plot_grid)

    boundary_curve = build_surface_from_psin(plot_equilibrium, 1.0)
    rz_limits = compute_rz_limits([boundary_curve])
    fig, ax = plt.subplots(figsize=(7.4, 6.6), constrained_layout=True)
    plot_equilibrium_surfaces(ax, plot_equilibrium, levels=DEFAULT_LEVELS)
    ax.scatter(
        [equilibrium.R0],
        [equilibrium.Z0],
        marker="x",
        color="#d62728",
        s=42,
        linewidths=1.4,
        label="Boundary (R0, Z0)",
    )
    style_surface_axis(ax, title="VEQPy Demo Flux Surfaces", rz_limits=rz_limits)
    ax.legend(loc="upper right")
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    # Save one serialized equilibrium so users can inspect the solved payload.
    equilibrium.write(str(equilibrium_path))

    print(f"Saved figure      : {figure_path}")
    print(f"Saved equilibrium : {equilibrium_path}")
    print(f"Kernel success    : {result.success}")
    print(f"Raw residual norm : {result.raw_norm:.6e}")
    print(f"Ip [MA]           : {equilibrium.Ip / 1.0e6:.6f}")
    print(f"beta_t            : {equilibrium.beta_t:.6e}")
    kernel.close()


if __name__ == "__main__":
    main()
