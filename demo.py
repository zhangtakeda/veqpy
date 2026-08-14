"""Minimal external-user demo for one VEQPy Kernel case."""

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
    P_psin = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    FF_psin = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    return P_psin.astype(np.float64), FF_psin.astype(np.float64)


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
    constraint="ip",
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
P_psin, FF_psin = pf_profiles(source_axis)
source = veq.KernelSource(
    P_psin=P_psin,
    FF_psin=FF_psin,
    Ip=3.0e6,
)

result = kernel.solve(boundary=boundary, source=source)

initial = kernel.build_equilibrium(x=np.zeros(kernel.x_size))
initial.plot("demo_init.png")

equilibrium = kernel.build_equilibrium()
equilibrium.plot("demo_result.png")
equilibrium.write("demo_equilibrium.json")

print("VEQPy minimal Kernel demo")
print(f"success: {result.success}")
print(f"residual: {result.raw_norm:.3e}")
print(f"nfev: {result.nfev}")
