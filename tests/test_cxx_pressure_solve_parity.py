from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from veqpy import (
    Geqdsk,
    Kernel,
    KernelBoundary,
    KernelConfig,
    KernelRecipe,
    KernelSource,
    KernelTopology,
)
from veqpy.kernels.abi.source_semantics import MU0
from veqpy.numerics import make_quadrature

pytestmark = pytest.mark.slow

_PRESSURE = 6410.0
_BETA = 0.02


def _circular_pq_topology(constraint: str) -> KernelTopology:
    return KernelTopology(
        h_count=3,
        v_count=0,
        kappa_count=6,
        psin_count=0,
        F_count=0,
        c_counts=(),
        s_counts=(3,),
        Nr=32,
        Nt=32,
        route="PQ",
        coordinate="rho",
        nodes="grid",
        constraint=constraint,
        sample_count=32,
    )


def _circular_boundary() -> KernelBoundary:
    return KernelBoundary(
        a=1.0,
        R0=10.0,
        Z0=0.0,
        B0=3.0,
        ka=1.0,
        s_offsets=(0.0,),
    )


def _constant_pressure_sources(
    topology: KernelTopology,
) -> tuple[KernelSource, KernelSource, np.ndarray]:
    rho, _ = make_quadrature(topology.Nr, scheme=topology.quadrature)
    q = 1.71 + 0.16 * rho * rho
    constraints: dict[str, float] = {}
    if topology.source_uses_beta_constraint:
        constraints["beta"] = _BETA
    return (
        KernelSource(
            p=np.full(topology.Nr, _PRESSURE, dtype=np.float64),
            q=q,
            **constraints,
        ),
        KernelSource(
            pprime=np.zeros(topology.Nr, dtype=np.float64),
            p0=_PRESSURE,
            q=q,
            **constraints,
        ),
        q,
    )


def _solve_config() -> KernelConfig:
    return KernelConfig(
        method="powell",
        initial="cold",
        continuation="cold",
        norm="fast",
        max_residual=1.0e-8,
        max_evaluations=3000,
    )


def _cxx_source_state(kernel: Kernel) -> np.ndarray:
    state = dict(kernel._impl._cxx_solver().source_state())
    return np.array(
        [
            state["alpha1"],
            state["alpha2"],
            state["scaled_effective_p0"],
            state["pressure_multiplier"],
        ],
        dtype=np.float64,
    )


@pytest.fixture(scope="module")
def cxx_cache_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cxx-pressure-solve-cache")


@pytest.mark.parametrize("constraint", ["none", "beta"])
def test_circular_constant_pressure_solve_is_backend_and_input_mode_invariant(
    constraint: str,
    cxx_cache_root: Path,
    tmp_path: Path,
) -> None:
    topology = _circular_pq_topology(constraint)
    source_from_p, source_from_pprime, target_q = _constant_pressure_sources(topology)
    boundary = _circular_boundary()
    config = _solve_config()
    kernels = {
        "numba": Kernel(
            topology=topology,
            recipe=KernelRecipe(backend="numba"),
        ),
        "cxx": Kernel(
            topology=topology,
            recipe=KernelRecipe(backend="cxx"),
            cache_root=cxx_cache_root,
        ),
    }
    results = {}
    equilibria = {}
    cxx_states = {}
    try:
        for backend, kernel in kernels.items():
            for pressure_mode, source in (
                ("p", source_from_p),
                ("pprime", source_from_pprime),
            ):
                result = kernel.solve(boundary, source, config=config)
                results[backend, pressure_mode] = result
                equilibria[backend, pressure_mode] = kernel.build_equilibrium()
                if backend == "cxx":
                    cxx_states[pressure_mode] = _cxx_source_state(kernel)
    finally:
        for kernel in kernels.values():
            kernel.close()

    for result in results.values():
        assert result.success
        assert result.raw_norm < config.max_residual
        assert np.all(np.isfinite(result.x))
        assert np.all(np.isfinite(result.alpha))

    for backend in ("numba", "cxx"):
        assert_allclose(
            results[backend, "p"].x,
            results[backend, "pprime"].x,
            rtol=0.0,
            atol=0.0,
        )
        assert_allclose(
            results[backend, "p"].alpha,
            results[backend, "pprime"].alpha,
            rtol=0.0,
            atol=0.0,
        )
    assert_allclose(
        results["cxx", "p"].x,
        results["numba", "p"].x,
        rtol=2.0e-8,
        atol=5.0e-11,
    )
    assert_allclose(
        results["cxx", "p"].alpha,
        results["numba", "p"].alpha,
        rtol=2.0e-9,
        atol=5.0e-12,
    )

    expected_pressure = (
        _PRESSURE
        if constraint == "none"
        else 0.5 * _BETA * boundary.B0 * boundary.B0 / MU0
    )
    expected_beta = 2.0 * MU0 * expected_pressure / (boundary.B0 * boundary.B0)
    cxx_result = results["cxx", "p"]
    cxx_equilibrium = equilibria["cxx", "p"]
    cxx_state = cxx_states["p"]
    assert_allclose(cxx_state[:2], cxx_result.alpha, rtol=2.0e-13, atol=2.0e-13)
    assert cxx_state[2] / MU0 == pytest.approx(expected_pressure)
    assert cxx_state[3] == pytest.approx(expected_pressure / _PRESSURE)
    assert abs(cxx_result.alpha[0] * cxx_result.alpha[1]) == pytest.approx(
        MU0 * expected_pressure
    )
    assert cxx_equilibrium.p0 == pytest.approx(expected_pressure)
    assert_allclose(cxx_equilibrium.P, expected_pressure, rtol=0.0, atol=1.0e-10)
    assert_allclose(cxx_equilibrium.P_r, 0.0, rtol=0.0, atol=1.0e-12)
    assert cxx_equilibrium.beta_t == pytest.approx(expected_beta)
    assert_allclose(
        cxx_equilibrium.q,
        target_q,
        rtol=2.0e-7,
        atol=5.0e-9,
    )

    numba_equilibrium = equilibria["numba", "p"]
    assert_allclose(cxx_equilibrium.psin, numba_equilibrium.psin, rtol=2.0e-9, atol=5.0e-11)
    assert_allclose(cxx_equilibrium.R, numba_equilibrium.R, rtol=2.0e-9, atol=5.0e-11)
    assert_allclose(cxx_equilibrium.Z, numba_equilibrium.Z, rtol=2.0e-9, atol=5.0e-11)
    assert cxx_equilibrium.Ip == pytest.approx(numba_equilibrium.Ip, rel=2.0e-9)

    if constraint == "none":
        _assert_geqdsk_roundtrip(cxx_equilibrium, tmp_path)


def _assert_geqdsk_roundtrip(equilibrium, tmp_path: Path) -> None:
    output = tmp_path / "circular-pq-cxx-129x129.geqdsk"
    geqdsk = equilibrium.to_geqdsk(
        R_range=(8.5, 11.5),
        Z_range=(-1.5, 1.5),
        NR=129,
        NZ=129,
        header="Circular PQ equilibrium solved by VEQPy Cxx",
    )
    geqdsk.check()
    geqdsk.write(output)
    restored = Geqdsk(output)
    restored.check()

    assert (restored.NR, restored.NZ) == (129, 129)
    assert restored.psi.shape == (129, 129)
    assert restored.boundary.shape == (65, 2)
    assert_allclose(restored.boundary[-1], restored.boundary[0], rtol=0.0, atol=0.0)
    assert restored.Rmin == pytest.approx(8.5)
    assert restored.Rmax == pytest.approx(11.5)
    assert restored.Zmin == pytest.approx(-1.5)
    assert restored.Zmax == pytest.approx(1.5)
    assert_allclose(restored.P, _PRESSURE, rtol=2.0e-9, atol=2.0e-8)
    assert_allclose(restored.P_psi, 0.0, rtol=0.0, atol=1.0e-12)
