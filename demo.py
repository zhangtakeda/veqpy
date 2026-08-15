"""Minimal external-user demo for the VEQPy 2.x Module API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import veqpy as veq
from veqpy.demo_case import make_demo_plasma


def _topology() -> veq.KernelTopology:
    """Return the reusable topology used by the bundled demo."""

    return veq.KernelTopology(
        h_count=3,
        v_count=3,
        kappa_count=3,
        psin_count=6,
        F_count=0,
        c_counts=(3, 3, 3),
        s_counts=(3, 3),
        Nr=16,
        Nt=16,
        route="PF",
        coordinate="psin",
        nodes="uniform",
        constraint="ip",
        sample_count=51,
    )


def _write_figure(equilibrium: object, output: Path, *, initial: bool = False) -> None:
    """Render one lightweight geometry/flux figure using the optional plot extra."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(equilibrium.psin, dtype=np.float64)
    fig, axis = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    levels = np.linspace(0.1, 0.9, 9)
    axis.contour(
        equilibrium.R,
        equilibrium.Z,
        values[:, None] * np.ones_like(equilibrium.R),
        levels=levels,
        colors="tab:blue",
        linewidths=0.8,
    )
    axis.plot(equilibrium.R_lcfs, equilibrium.Z_lcfs, color="tab:orange", linewidth=2.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set(xlabel="R [m]", ylabel="Z [m]", title="VEQPy initial" if initial else "VEQPy result")
    axis.grid(alpha=0.25)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> int:
    """Run one solve and write the same small artifacts as the old demo."""

    topology = _topology()
    module = veq.VEQ(topology=topology, backend="numba")
    try:
        result = module.run(plasma=make_demo_plasma())
        if not result.accepted or result.equilibrium is None:
            raise RuntimeError(f"VEQPy solve failed with residual {result.residual_norm:.3e}")
        equilibrium = result.equilibrium
        initial = equilibrium.replace(
            psi_r=np.asarray(equilibrium.psi_r, dtype=np.float64),
            psi_rr=np.asarray(equilibrium.psi_rr, dtype=np.float64),
        )
        _write_figure(initial, Path("demo_init.png"), initial=True)
        _write_figure(equilibrium, Path("demo_result.png"))
        equilibrium.write("demo_equilibrium.json")
    finally:
        module.close()

    print("VEQPy minimal Module demo")
    print(f"success: {result.accepted}")
    print(f"residual: {result.residual_norm:.3e}")
    print(f"nfev: {result.evaluations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
