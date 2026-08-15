"""Solve the bundled Solovev GEQDSK case through the VEQPy Module API."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from fusionprime_base import Equilibrium, Geometry, Plasma

import veqpy as veq
from veqpy.demo_case import make_demo_plasma
from veqpy.io import export_geqdsk
from veqpy.kernels.boundary_fit import fit_boundary_params

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "SOLOVEV.geqdsk"
OUTPUT = ROOT / "data" / "solovev-veqpy.geqdsk"
FIGURE = ROOT / "data" / "solovev-veqpy-comparison.png"


def _reference_plasma(reference: veq.Geqdsk) -> Plasma:
    """Lower GEQDSK roots into a frozen FusionPRIME Plasma context."""

    order = 8
    radial_coefficients = 11
    fit = fit_boundary_params(
        reference.boundary[:, 0],
        reference.boundary[:, 1],
        c_order=order,
        s_order=order,
        maxtol=1.0e-2,
        method="least-square",
    )
    delta_psi = float(reference.psi_bound - reference.psi_axis)
    radial_nodes = np.linspace(0.0, 1.0, reference.P_psi.size, dtype=np.float64)
    geometry = Geometry(
        Nr=radial_nodes.size,
        Nt=64,
        radial_rule="uniform",
        radial_calculus="spectral",
        K_max=None,
        R0=float(fit["R0"]),
        Z0=float(fit["Z0"]),
        a=float(fit["a"]),
        kappa_lcfs=float(fit["ka"]),
        c_lcfs=np.asarray(fit["c_offsets"], dtype=np.float64),
        s_lcfs=np.asarray(fit["s_offsets"], dtype=np.float64)[1:],
        h_coeffs=np.zeros(radial_coefficients, dtype=np.float64),
        v_coeffs=np.zeros(radial_coefficients, dtype=np.float64),
        kappa_coeffs=np.zeros(radial_coefficients, dtype=np.float64),
        c_coeffs=np.zeros((order + 1, radial_coefficients), dtype=np.float64),
        s_coeffs=np.zeros((order, radial_coefficients), dtype=np.float64),
    )
    equilibrium = Equilibrium(
        geometry=geometry,
        FF_psi=np.asarray(reference.FF_psi, dtype=np.float64),
        P_psi=np.asarray(reference.P_psi, dtype=np.float64),
        psi_r=delta_psi * 2.0 * radial_nodes,
        psi_rr=np.full(radial_nodes.size, 2.0 * delta_psi, dtype=np.float64),
        B0=float(reference.Bt0),
        P0=float(reference.P[-1]),
    ).freeze()
    fixture = make_demo_plasma()
    return Plasma(
        equilibrium=equilibrium,
        kinetic=fixture.kinetic,
        current=fixture.current,
        flux=fixture.flux,
        source=fixture.source,
    ).freeze()


def _topology(source_count: int) -> veq.KernelTopology:
    return veq.KernelTopology(
        h_count=11,
        v_count=11,
        kappa_count=11,
        psin_count=10,
        F_count=0,
        c_counts=(11,) * 9,
        s_counts=(11,) * 8,
        Nr=32,
        Nt=32,
        route="PF",
        coordinate="psin",
        nodes="uniform",
        constraint="ip",
        sample_count=source_count,
    )


def plot_geqdsk_comparison(
    reference: veq.Geqdsk,
    solved: veq.Geqdsk,
    output: Path,
) -> None:
    """Compare flux surfaces, the LCFS, and PF profiles from two GEQDSK files."""

    cases = (
        (reference, "SOLOVEV input", "black", "-"),
        (solved, "VEQPy Numba", "tab:orange", "--"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    geometry = axes[0, 0]
    levels = np.linspace(0.1, 0.9, 9)
    for geqdsk, label, color, linestyle in cases:
        R = np.linspace(geqdsk.Rmin, geqdsk.Rmax, geqdsk.NR)
        Z = np.linspace(geqdsk.Zmin, geqdsk.Zmax, geqdsk.NZ)
        psin = (geqdsk.psi - geqdsk.psi_axis) / (geqdsk.psi_bound - geqdsk.psi_axis)
        geometry.contour(
            R,
            Z,
            psin.T,
            levels=levels,
            colors=color,
            linestyles=linestyle,
            linewidths=0.8,
        )
        geometry.plot(
            geqdsk.boundary[:, 0],
            geqdsk.boundary[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
        geometry.plot(geqdsk.Raxis, geqdsk.Zaxis, marker="x", color=color)
    geometry.set(xlabel="R [m]", ylabel="Z [m]", title="Normalized flux surfaces and LCFS")
    geometry.set_aspect("equal", adjustable="box")
    geometry.grid(alpha=0.25)

    profiles = (
        (axes[0, 1], "P", 1.0e-6, "Pressure", "P [MPa]"),
        (axes[1, 0], "P_psi", 1.0e-3, "Pressure derivative", "P_psi [kPa/Wb]"),
        (axes[1, 1], "FF_psi", 1.0, "Toroidal-field source", "FF_psi"),
    )
    for axis, name, scale, title, ylabel in profiles:
        for geqdsk, label, color, linestyle in cases:
            values = np.asarray(getattr(geqdsk, name), dtype=np.float64)
            psin = np.linspace(0.0, 1.0, values.size)
            axis.plot(psin, scale * values, color=color, linestyle=linestyle, linewidth=1.8, label=label)
        axis.set(xlabel="psin", ylabel=ylabel, title=title)
        axis.grid(alpha=0.25)
        axis.legend()

    fig.suptitle("SOLOVEV.geqdsk vs solovev-veqpy.geqdsk")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> int:
    reference = veq.Geqdsk(INPUT)
    plasma = _reference_plasma(reference)
    topology = _topology(reference.P_psi.size)
    module = veq.VEQ(topology=topology, backend="numba", config=veq.KernelConfig(max_evaluations=2000))
    try:
        result = module.run(plasma=plasma)
        if not result.accepted or result.equilibrium is None:
            raise RuntimeError(f"VEQPy solve failed with residual {result.residual_norm:.3e}")
        solved = export_geqdsk(
            result.equilibrium,
            R_range=(reference.Rmin, reference.Rmax),
            Z_range=(reference.Zmin, reference.Zmax),
            NR=reference.NR,
            NZ=reference.NZ,
            header="Solovev equilibrium solved by VEQPy Numba",
            limiter=reference.limiter,
            psi_axis=reference.psi_axis,
        )
        solved.write(OUTPUT)
    finally:
        module.close()

    exported = veq.Geqdsk(OUTPUT)
    plot_geqdsk_comparison(reference, exported, FIGURE)
    print("VEQPy Numba GEQDSK demo")
    print(f"success: {result.accepted}")
    print(f"residual: {result.residual_norm:.3e}")
    print(f"nfev: {result.evaluations}")
    print(f"output: {OUTPUT.resolve()}")
    print(f"comparison: {FIGURE.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
