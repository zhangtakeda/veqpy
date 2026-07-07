"""Minimal external-user demo for one VEQPy Kernel case."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import veqpy as veq


def pf_profiles(psin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return smooth PF source profiles sampled on normalized flux."""

    beta0 = 0.75
    alpha_p = 5.0
    alpha_f = 3.32
    exp_ap = np.exp(alpha_p)
    exp_af = np.exp(alpha_f)
    den_p = 1.0 + exp_ap * (alpha_p - 1.0)
    den_f = 1.0 + exp_af * (alpha_f - 1.0)
    heat_profile = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    current_profile = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    return heat_profile.astype(np.float64), current_profile.astype(np.float64)


def main() -> None:
    output_dir = Path(os.environ.get("VEQPY_OUTPUT_DIR", ".")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Topology is fixed for one reusable Kernel handle.
    topology = veq.KernelTopology(
        h_count=3,
        v_count=0,
        kappa_count=6,
        psin_count=6,
        F_count=0,
        c_counts=(),
        s_counts=(3,),
        Nr=16,
        Nt=16,
        route="PF",
        coordinate="psin",
        nodes="uniform",
        ip_constraint=True,
        sample_count=51,
    )
    kernel = veq.build(
        topology=topology,
        recipe=veq.KernelRecipe(backend="numba"),
        config=veq.KernelConfig(initial="cold"),
    )

    # Boundary and source are runtime inputs for this particular case.
    boundary = veq.KernelBoundary(
        a=1.05 / 1.85,
        R0=1.05,
        Z0=0.0,
        B0=3.0,
        ka=2.2,
        s_offsets=(float(np.arcsin(0.5)),),
    )
    source_axis = np.linspace(0.0, 1.0, topology.sample_count, dtype=np.float64)
    heat_profile, current_profile = pf_profiles(source_axis)
    source = veq.KernelSource(
        heat_profile=heat_profile,
        current_profile=current_profile,
        Ip=3.0e6,
    )

    try:
        result = kernel.solve(boundary=boundary, source=source)
        if not result.success:
            raise RuntimeError(f"Kernel solve failed with residual {result.raw_norm:.3e}")

        initial = kernel.build_equilibrium(x=np.zeros(kernel.x_size))
        equilibrium = kernel.build_equilibrium()
        initial.plot(outpath=str(output_dir / "demo_init.png"))
        equilibrium.plot(outpath=str(output_dir / "demo_result.png"))
        equilibrium.write(str(output_dir / "demo_equilibrium.json"))
    finally:
        kernel.close()

    print("VEQPy minimal Kernel demo")
    print(f"success: {result.success}")
    print(f"residual: {result.raw_norm:.3e}")
    print(f"nfev: {result.nfev}")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
