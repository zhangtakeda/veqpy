"""Solve the bundled Solovev GEQDSK case with the VEQPy Numba kernel."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import veqpy as veq

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "TEST.geqdsk"
OUTPUT = ROOT / "data" / "solovev-veqpy.geqdsk"
FIGURE = ROOT / "data" / "solovev-veqpy-comparison.png"


# Read the reference LCFS, PF profiles, machine field, and plasma current.
reference = veq.Geqdsk(INPUT)
boundary = veq.KernelBoundary(
    B0=reference.Bt0,
    R_boundary=reference.boundary[:, 0],
    Z_boundary=reference.boundary[:, 1],
    c_order=10,
    s_order=10,
    fit_maxtol=1.0,
    method="least-square",
).fit(backend="numba")
source = veq.KernelSource(
    P_psin=(reference.psi_bound - reference.psi_axis) * reference.P_psi,
    p0=float(reference.P[-1]),
    FF_psin=(reference.psi_bound - reference.psi_axis) * reference.FF_psi,
    Ip=abs(reference.Ip),
    case_name="solovev-geqdsk",
)

# Use enough vertical-shift and Fourier freedom for vertically asymmetric g-files.
topology = veq.KernelTopology(
    h_count=10,
    v_count=10,
    kappa_count=10,
    psin_count=10,
    F_count=0,
    c_counts=(5, 5, 5, 5, 5, 5, 5, 5, 5),
    s_counts=(5, 5, 5, 5, 5, 5, 5, 5, 5),
    Nr=32,
    Nt=32,
    route="PF",
    coordinate="psin",
    nodes="uniform",
    constraint="ip",
    sample_count=reference.P_psi.size,
)
kernel = veq.build(
    topology=topology,
    recipe=veq.KernelRecipe(backend="numba"),
    config=veq.KernelConfig(
        method="powell",
        max_residual=1.0e-6,
        max_evaluations=2000,
        initial="cold",
        continuation="cold",
    ),
)

result = kernel.solve(boundary=boundary, source=source)
if not result.success:
    raise RuntimeError(f"VEQPy solve failed with residual {result.raw_norm:.3e}")

equilibrium = kernel.build_equilibrium()
solved_geqdsk = equilibrium.to_geqdsk(
    R_range=(reference.Rmin, reference.Rmax),
    Z_range=(reference.Zmin, reference.Zmax),
    NR=reference.NR,
    NZ=reference.NZ,
    header="Solovev equilibrium solved by VEQPy Numba",
    limiter=reference.limiter,
    psi_axis=reference.psi_axis,
)
solved_geqdsk.write(OUTPUT)
exported = veq.Geqdsk(OUTPUT)
kernel.close()


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
    geometry.set(
        xlabel=r"$R$ [m]",
        ylabel=r"$Z$ [m]",
        title=r"Normalized $\psi$ surfaces and LCFS",
    )
    geometry.set_aspect("equal", adjustable="box")
    geometry.grid(alpha=0.25)

    profiles = (
        (axes[0, 1], "P", 1.0e-6, "Pressure", r"$P$ [MPa]"),
        (axes[1, 0], "P_psi", 1.0e-3, "Pressure derivative", r"$P_\psi$ [kPa/Wb]"),
        (axes[1, 1], "FF_psi", 1.0, "Toroidal-field source", r"$FF_\psi$"),
    )
    for axis, name, scale, title, ylabel in profiles:
        for geqdsk, label, color, linestyle in cases:
            values = np.asarray(getattr(geqdsk, name), dtype=np.float64)
            psin = np.linspace(0.0, 1.0, values.size)
            axis.plot(
                psin,
                scale * values,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=label,
            )
        axis.set(xlabel=r"$\psi_n$", ylabel=ylabel, title=title)
        axis.grid(alpha=0.25)
        axis.legend()

    fig.suptitle("SOLOVEV.geqdsk vs solovev-veqpy.geqdsk")
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


plot_geqdsk_comparison(reference, exported, FIGURE)

print("VEQPy Numba GEQDSK demo")
print(f"success: {result.success}")
print(f"residual: {result.raw_norm:.3e}")
print(f"nfev: {result.nfev}")
print(f"output: {OUTPUT.resolve()}")
print(f"comparison: {FIGURE.resolve()}")
