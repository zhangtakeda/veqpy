"""Minimal Python-side VEQPy Kernel build + solve demo.

Run directly:

    python examples/kernel_build_solve.py

The example is intentionally notebook-like: each section is one step in the
runtime workflow.

The first run may spend a moment compiling Numba kernels. ``VEQPY_OUTPUT_DIR`` is
accepted for consistency with the plotting examples, although this script only
prints the solved Kernel result.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from veqpy.facade import (  # noqa: E402
    Kernel,
    KernelBoundary,
    KernelConfig,
    KernelSource,
    KernelTopology,
    SolveResult,
)

MU0 = 4.0e-7 * np.pi


def ensure_output_dir() -> Path:
    env_out = os.environ.get("VEQPY_OUTPUT_DIR")
    outdir = Path(env_out) if env_out else Path.cwd() / "outputs" / "kernel_build_solve"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def pf_reference_profiles(psin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75
    alpha_p = 5.0
    alpha_f = 3.32
    exp_ap = np.exp(alpha_p)
    exp_af = np.exp(alpha_f)
    den_p = 1.0 + exp_ap * (alpha_p - 1.0)
    den_f = 1.0 + exp_af * (alpha_f - 1.0)
    current = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    heat = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    return current.astype(np.float64), heat.astype(np.float64)


def build_boundary() -> KernelBoundary:
    return KernelBoundary(
        a=0.5,
        R0=1.0,
        Z0=0.0,
        B0=3.0,
        ka=1.7,
        s_offsets=np.array([0.0, float(np.arcsin(0.2))], dtype=np.float64),
    )


def build_source() -> KernelSource:
    psin = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    current_profile, pressure_gradient = pf_reference_profiles(psin)
    return KernelSource(
        heat_profile=pressure_gradient / MU0,
        current_profile=current_profile,
        Ip=3.0e6,
        case_name="kernel-build-solve-demo",
    )


def main() -> None:
    outdir = ensure_output_dir()

    # 1. Choose the fixed topology for this tiny Kernel handle.
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
        sample_count=9,
    )

    # 2. Create the reusable handle and set the default runtime config.
    kernel_config = KernelConfig(
        method="levenberg-marquardt",
        initial="cold-zeros",
        norm="none",
        max_evaluations=16,
    )
    kernel = Kernel(
        topology=topology,
        config=kernel_config,
    )
    kernel.prepare()

    # 3. Prepare typed raw runtime source data; facade materialization applies
    #    route-dependent mu0 scaling before backend entry.
    kernel_boundary = build_boundary()
    kernel_source = build_source()

    # 4. Solve. Kernel.solve() uses the handle default config unless a
    #    per-call config or field override (for example method=...) is supplied.
    result = kernel.solve(kernel_boundary, kernel_source)
    assert isinstance(result, SolveResult)

    print("VEQPy Kernel build + solve demo")
    print(f"  output dir: {outdir}")
    print(f"  topology key: {topology.key}")
    print(f"  x_size: {kernel.x_size}")
    print(f"  success: {result.success}")
    print(f"  info: {result.info}, nfev: {result.nfev}")
    print(f"  raw_norm: {result.raw_norm:.6e}")
    print(f"  scaled_norm: {result.scaled_norm:.6e}")
    print(f"  history length: {len(kernel.history)}")
    print(f"  first coefficients: {np.array2string(result.x[:4], precision=4)}")

    kernel.close()


if __name__ == "__main__":
    main()
