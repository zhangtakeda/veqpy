import numpy as np

import veqpy as veq

# Compile-time kernel topology.
setup_topology = veq.KernelTopology(
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

setup_recipe = veq.KernelRecipe(backend="numba")
setup_config = veq.KernelConfig(initial="cold-zeros")

# Build and warm the selected backend.
kernel = veq.build(
    topology=setup_topology,
    recipe=setup_recipe,
    config=setup_config,
)

# Runtime boundary and source inputs.
runtime_boundary = veq.KernelBoundary(
    a=1.05 / 1.85,
    R0=1.05,
    Z0=0.0,
    B0=3.0,
    ka=2.2,
    s_offsets=(float(np.arcsin(0.5)),),
)


def pf_reference_profiles(psin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75
    alpha_p = 5.0
    alpha_f = 3.32
    exp_ap = np.exp(alpha_p)
    exp_af = np.exp(alpha_f)
    den_p = 1.0 + exp_ap * (alpha_p - 1.0)
    den_f = 1.0 + exp_af * (alpha_f - 1.0)
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    return heat_input.astype(np.float64), current_input.astype(np.float64)


pprime, ffprime = pf_reference_profiles(np.linspace(0.0, 1.0, setup_topology.sample_count))
runtime_source = veq.KernelSource(heat_profile=pprime, current_profile=ffprime, Ip=3.0e6)

# Solve, then materialize the initial and final equilibrium snapshots.
result = kernel.solve(
    boundary=runtime_boundary,
    source=runtime_source,
)

eq0 = kernel.build_equilibrium(x=np.zeros(kernel.x_size))
eq0.plot(outpath="demo_init.png")

eq1 = kernel.build_equilibrium()
eq1.plot(outpath="demo_result.png")
