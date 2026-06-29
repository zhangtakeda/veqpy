from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import veqlib.facade as facade
import veqlib.facade.affinity as affinity
from veqlib.facade import (
    INITIAL_POLICY_COLD,
    RESIDUAL_NORMALIZATION_FAST,
    SOLVER_METHOD_POWELL,
    Kernel,
    KernelBoundary,
    KernelBuild,
    KernelInput,
    KernelResult,
    KernelSolve,
    KernelTopology,
)
from veqlib.facade.affinity import pinned_cpu
from veqlib.facade.registry import ThreadOwnedKernelSolver

MU0 = 4.0e-7 * np.pi
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_kernel_topology(**overrides: object) -> KernelTopology:
    params: dict[str, object] = {
        "h_count": 2,
        "v_count": 0,
        "kappa_count": 2,
        "psin_count": 3,
        "F_count": 0,
        "c_counts": (),
        "s_counts": (2,),
        "Nr": 8,
        "Nt": 8,
        "route": "pf",
        "coordinate": "PSIN",
        "constraint": "ip",
        "nodes": "uniform",
        "sample_count": 9,
    }
    params.update(overrides)
    return KernelTopology(**params)  # type: ignore[arg-type]


def tiny_kernel_boundary() -> KernelBoundary:
    return KernelBoundary(
        a=0.5,
        R0=1.0,
        Z0=0.0,
        B0=3.0,
        ka=1.7,
        s_offsets=np.array([0.0, np.arcsin(0.2)], dtype=np.float64),
    )


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


def tiny_kernel_input(*, case_name: str | None = None) -> KernelInput:
    psin = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    scaled_current, scaled_heat = pf_reference_profiles(psin)
    return KernelInput(
        boundary=tiny_kernel_boundary(),
        scaled_heat=scaled_heat,
        scaled_current=scaled_current,
        scaled_Ip=3.0e6 * MU0,
        case_name=case_name,
    )


def test_facade_public_exports_are_stable() -> None:
    assert facade.__all__ == [
        "INITIAL_POLICY_COLD",
        "INITIAL_POLICY_COLD_GEOMETRIC",
        "INITIAL_POLICY_COLD_ZEROS",
        "INITIAL_POLICY_WARM_CLONE",
        "Kernel",
        "KernelArtifact",
        "KernelBoundary",
        "KernelBuild",
        "KernelBuildError",
        "KernelInput",
        "KernelLoadError",
        "KernelRegistry",
        "KernelResult",
        "KernelSolve",
        "KernelTopology",
        "LoadedKernel",
        "PayloadSequenceStep",
        "RESIDUAL_NORMALIZATION_BALANCED",
        "RESIDUAL_NORMALIZATION_FAST",
        "RESIDUAL_NORMALIZATION_NONE",
        "RESIDUAL_NORMALIZATION_SAFE",
        "SOLVER_METHOD_LEVENBERG_MARQUARDT",
        "SOLVER_METHOD_NEWTON_KRYLOV",
        "SOLVER_METHOD_NEWTON_RAPHSON",
        "SOLVER_METHOD_POWELL",
        "SolverThreadError",
        "TopologyError",
        "VEQlibSolver",
        "build",
        "build_kernel",
        "current_cpu_affinity",
        "default_kernel_cache_root",
        "initial_policy_code",
        "payload_json_with_initial_policy",
        "pinned_cpu",
        "residual_normalization_code",
        "solve_payload_sequence",
        "solver_method_code",
    ]


def test_facade_user_signatures_hide_compiler_and_type_alias() -> None:
    for api in (
        Kernel,
        facade.build,
        facade.build_kernel,
        facade.KernelRegistry,
        facade.VEQlibSolver,
    ):
        assert "cxx" not in inspect.signature(api).parameters
    assert not hasattr(facade, "CpuPinning")
    assert not hasattr(affinity, "CpuPinning")


def test_veqlib_facade_import_does_not_import_veqpy() -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    source_paths = str(PROJECT_ROOT)
    env["PYTHONPATH"] = (
        source_paths
        if not existing_pythonpath
        else os.pathsep.join((source_paths, existing_pythonpath))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import veqlib.facade; "
            "print(any(name == 'veqpy' or name.startswith('veqpy.') for name in sys.modules))",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


def test_kernel_topology_canonicalizes_without_veqpy_model() -> None:
    topology = make_kernel_topology(c_counts=(0, 0), s_counts=(2, 0, 0), K_max=None)
    same_shape = make_kernel_topology(c_counts=(), s_counts=(2,), L_max=2, M_max=1, K_max=2)
    family_build = KernelBuild(layout="profile-first", build="release", fp_mode="FMA")
    family_topology = topology.with_build(family_build)

    assert topology.to_canonical_dict() == same_shape.to_canonical_dict()
    assert topology.key == same_shape.key
    assert family_topology.layout == "family"
    assert family_topology.build == "release"
    assert family_topology.fp_mode == "FMA"
    assert family_topology.route == "PF"
    assert family_topology.coordinate == "psin"
    assert family_topology.constraint == "Ip"
    assert family_topology.sample_count == 9
    assert topology.packed_size() == 9


def test_kernel_build_lowers_options_into_topology() -> None:
    topology = make_kernel_topology()
    build = KernelBuild(
        layout="family",
        build="fastmath-enzyme",
        enable_thin_lto=False,
        analysis=True,
        enzyme_jacobian_batch_width=8,
    )

    lowered = topology.with_build(build)

    assert lowered.layout == "family"
    assert lowered.build_options_dict() == {
        "preset": "fastmath-enzyme",
        "cmake_build_type": "Release",
        "fp_mode": "RELAXED",
        "enable_enzyme": True,
        "enable_native_optimizations": True,
        "enable_thin_lto": False,
        "analysis": True,
        "enzyme_jacobian_batch_width": 8,
    }


def test_kernel_input_is_typed_scaled_runtime_payload() -> None:
    psin = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    scaled_current, scaled_heat = pf_reference_profiles(psin)
    input_data = KernelInput(
        boundary=tiny_kernel_boundary(),
        scaled_heat=scaled_heat,
        scaled_current=scaled_current,
        scaled_Ip=3.0e6 * MU0,
        fix_rho=0.125,
        case_name="tiny",
    )
    payload = input_data.to_payload_dict()

    assert payload["case_name"] == "tiny"
    assert payload["boundary"]["a"] == 0.5
    assert payload["constraints"]["fix_rho"] == 0.125
    assert_allclose(payload["source"]["scaled_heat"], scaled_heat)
    assert_allclose(payload["source"]["scaled_current"], scaled_current)
    assert_allclose(payload["constraints"]["scaled_Ip"], 3.0e6 * MU0)
    assert "beta" not in payload["constraints"]
    assert not input_data.scaled_heat.flags.writeable
    assert not input_data.scaled_current.flags.writeable


def test_kernel_input_direct_constructor_validates_shape() -> None:
    with pytest.raises(ValueError, match="scaled_heat must be 1D"):
        KernelInput(
            boundary=tiny_kernel_boundary(),
            scaled_heat=np.ones((2, 1), dtype=np.float64),
            scaled_current=np.ones(2, dtype=np.float64),
        )


def test_kernel_solve_payload_uses_codes_and_x_size_default_budget() -> None:
    solve = KernelSolve(method="powell", initial="cold", norm="fast")

    payload = solve.to_payload_dict(x_size=7)

    assert payload["method_code"] == SOLVER_METHOD_POWELL
    assert payload["initial_policy_code"] == INITIAL_POLICY_COLD
    assert payload["residual_normalization_code"] == RESIDUAL_NORMALIZATION_FAST
    assert payload["max_evaluations"] == 49
    assert solve.runtime_args(x_size=7) == (
        SOLVER_METHOD_POWELL,
        1.0e-6,
        49,
        10.0,
        1.0e-5,
        INITIAL_POLICY_COLD,
        RESIDUAL_NORMALIZATION_FAST,
        1.0,
        1.0e6,
        3.0,
        4,
        1.0e-6,
        0.5,
    )


def test_kernel_handle_build_dry_run_and_payload_json(tmp_path) -> None:
    topology = make_kernel_topology()
    handle = facade.build(topology, cache_root=tmp_path, dry_run=True)
    payload = json.loads(handle.payload_json(tiny_kernel_input(case_name="payload-smoke")))

    assert handle.x_size == 9
    assert handle.build_topology.key == topology.key
    assert payload["case_name"] == "payload-smoke"
    assert payload["solver"]["max_evaluations"] == 81
    assert payload["source"]["scaled_heat"]


@pytest.mark.slow
def test_kernel_python_build_and_solve_native_flow(tmp_path) -> None:
    """Document the Python-side VEQlib flow: topology -> build -> solve -> snapshot."""

    topology = make_kernel_topology()
    build_config = KernelBuild(build="fastmath")
    handle = Kernel(topology, build=build_config, cache_root=tmp_path)

    artifact = handle.build()
    assert artifact.built is True
    assert artifact.shared_library_path.exists()
    assert handle.x_size == 9

    runtime_input = tiny_kernel_input(case_name="python-build-solve-flow")
    solve_config = KernelSolve(method="powell", initial="cold", norm="fast")

    result = handle.solve(runtime_input, solve=solve_config)

    assert isinstance(result, KernelResult)
    assert result.success is True
    assert result.x.shape == (handle.x_size,)
    assert result.raw.shape == (handle.x_size,)
    assert result.scaled.shape == (handle.x_size,)
    assert result.alpha.shape == (2,)
    assert result.x.flags.owndata
    assert result.raw.flags.owndata
    assert result.scaled.flags.owndata
    assert result.alpha.flags.owndata
    assert result.raw_norm >= 0.0
    assert result.scaled_norm >= 0.0
    assert handle.result is result
    assert handle.history == [result]
    assert "python-build-solve-flow" in handle.metadata_json()

    bad_source = KernelInput(
        boundary=tiny_kernel_boundary(),
        scaled_heat=np.ones(8, dtype=np.float64),
        scaled_current=np.ones(8, dtype=np.float64),
        scaled_Ip=3.0e6 * MU0,
    )
    with pytest.raises(RuntimeError, match="scaled_heat length mismatch"):
        handle.solve(bad_source, solve=solve_config)

    bad_offsets = KernelInput(
        boundary=KernelBoundary(
            a=0.5,
            R0=1.0,
            Z0=0.0,
            B0=3.0,
            ka=1.7,
            c_offsets=np.ones(3, dtype=np.float64),
            s_offsets=np.array([0.0, np.arcsin(0.2)], dtype=np.float64),
        ),
        scaled_heat=runtime_input.scaled_heat,
        scaled_current=runtime_input.scaled_current,
        scaled_Ip=3.0e6 * MU0,
    )
    with pytest.raises(RuntimeError, match="c_offsets length mismatch"):
        handle.solve(bad_offsets, solve=solve_config)

    handle.close()
    assert handle._solver is None


class _FakeVEQlibSolver:
    def __init__(self, *, x_size: int = 9) -> None:
        self.x_size = x_size
        self.runtime_calls: list[tuple[object, ...]] = []
        self.json_payloads: list[str] = []
        self.solve_count = 0
        self._x = np.zeros(x_size, dtype=np.float64)
        self._raw = np.zeros(x_size, dtype=np.float64)
        self._scaled = np.zeros(x_size, dtype=np.float64)
        self._alpha = np.zeros(2, dtype=np.float64)

    def build(self, *, force: bool = False, dry_run: bool = False):  # pragma: no cover - not used
        raise AssertionError("build should not be called")

    def set_kernel_runtime(self, *args: object) -> None:
        self.runtime_calls.append(args)

    def set_case_json(self, payload: str) -> None:
        self.json_payloads.append(payload)

    def solve_direct(self):
        self.solve_count += 1
        self._x[:] = self.solve_count
        self._raw[:] = self.solve_count + 10
        self._scaled[:] = self.solve_count + 20
        self._alpha[:] = (self.solve_count, self.solve_count + 0.5)
        return (
            0.25,
            True,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8.0,
            9.0,
            self._x,
            self._raw,
            self._scaled,
            self._alpha,
        )


class _FakeLegacyVEQlibSolver(_FakeVEQlibSolver):
    def __getattribute__(self, name: str):
        if name == "set_kernel_runtime":
            raise AttributeError(name)
        return super().__getattribute__(name)


def test_kernel_result_copies_solve_direct_views() -> None:
    raw = _FakeVEQlibSolver(x_size=3).solve_direct()

    result = KernelResult.from_solve_direct(raw)
    raw[11][:] = 99.0

    assert result.elapsed_ms == 0.25
    assert result.success is True
    assert result.info == 1
    assert_allclose(result.x, np.ones(3))
    assert result.x.flags.owndata
    assert result.raw.flags.owndata
    assert result.scaled.flags.owndata
    assert result.alpha.flags.owndata


def test_kernel_solve_uses_typed_runtime_and_owned_history() -> None:
    topology = make_kernel_topology()
    handle = Kernel(topology)
    fake = _FakeVEQlibSolver(x_size=handle.x_size)
    handle._solver = fake  # exercise Kernel.solve without building a native artifact

    first = handle.solve(tiny_kernel_input(case_name="typed"))
    second = handle.solve(tiny_kernel_input(case_name="typed-2"))

    assert fake.json_payloads == []
    assert len(fake.runtime_calls) == 2
    assert fake.runtime_calls[0][0] == "typed"
    assert fake.runtime_calls[1][0] == "typed-2"
    assert first is handle.history[0]
    assert second is handle.result
    assert handle.history == [first, second]
    assert_allclose(first.x, np.ones(handle.x_size))
    assert_allclose(second.x, np.full(handle.x_size, 2.0))


def test_kernel_solve_falls_back_to_json_for_legacy_solver() -> None:
    topology = make_kernel_topology()
    handle = Kernel(topology)
    fake = _FakeLegacyVEQlibSolver(x_size=handle.x_size)
    handle._solver = fake

    result = handle.solve(tiny_kernel_input(case_name="legacy"))

    assert isinstance(result, KernelResult)
    assert len(fake.json_payloads) == 1
    payload = json.loads(fake.json_payloads[0])
    assert payload["case_name"] == "legacy"
    assert payload["solver"]["max_evaluations"] == handle.x_size * handle.x_size


def test_kernel_clear_and_close_lifecycle() -> None:
    topology = make_kernel_topology()
    handle = Kernel(topology)
    fake = _FakeVEQlibSolver(x_size=handle.x_size)
    handle._solver = fake
    handle.solve(tiny_kernel_input())

    handle.clear()
    assert handle.history == []
    assert handle.result is None
    assert handle._solver is fake

    handle.close()
    assert handle._solver is None


def test_pinned_cpu_defaults_to_min_allowed_and_restores(monkeypatch) -> None:
    current = {2, 4, 8}
    calls: list[tuple[int, tuple[int, ...]]] = []

    def fake_getaffinity(pid: int) -> set[int]:
        assert pid == 0
        return set(current)

    def fake_setaffinity(pid: int, cpus: set[int]) -> None:
        assert pid == 0
        calls.append((pid, tuple(sorted(cpus))))
        current.clear()
        current.update(cpus)

    monkeypatch.setattr("veqlib.facade.affinity.os.sched_getaffinity", fake_getaffinity)
    monkeypatch.setattr("veqlib.facade.affinity.os.sched_setaffinity", fake_setaffinity)
    monkeypatch.delenv("VEQLIB_PIN_CPU", raising=False)
    monkeypatch.delenv("VEQLIB_PIN_CPU_ID", raising=False)

    with pinned_cpu():
        assert current == {2}

    assert current == {2, 4, 8}
    assert calls == [(0, (2,)), (0, (2, 4, 8))]


def test_pinned_cpu_can_be_disabled_by_env(monkeypatch) -> None:
    current = {3, 5}
    calls: list[tuple[int, tuple[int, ...]]] = []

    monkeypatch.setattr(
        "veqlib.facade.affinity.os.sched_getaffinity",
        lambda pid: set(current),
    )
    monkeypatch.setattr(
        "veqlib.facade.affinity.os.sched_setaffinity",
        lambda pid, cpus: calls.append((pid, tuple(sorted(cpus)))),
    )
    monkeypatch.setenv("VEQLIB_PIN_CPU", "0")
    monkeypatch.delenv("VEQLIB_PIN_CPU_ID", raising=False)

    with pinned_cpu():
        assert current == {3, 5}

    assert calls == []


def test_pinned_cpu_nested_scope_reuses_outer_affinity(monkeypatch) -> None:
    current = {6, 7}
    calls: list[tuple[int, tuple[int, ...]]] = []

    def fake_getaffinity(pid: int) -> set[int]:
        assert pid == 0
        return set(current)

    def fake_setaffinity(pid: int, cpus: set[int]) -> None:
        assert pid == 0
        calls.append((pid, tuple(sorted(cpus))))
        current.clear()
        current.update(cpus)

    monkeypatch.setattr("veqlib.facade.affinity.os.sched_getaffinity", fake_getaffinity)
    monkeypatch.setattr("veqlib.facade.affinity.os.sched_setaffinity", fake_setaffinity)
    monkeypatch.delenv("VEQLIB_PIN_CPU", raising=False)
    monkeypatch.delenv("VEQLIB_PIN_CPU_ID", raising=False)

    with pinned_cpu():
        assert current == {6}
        with pinned_cpu():
            assert current == {6}
        with pinned_cpu():
            assert current == {6}

    assert current == {6, 7}
    assert calls == [(0, (6,)), (0, (6, 7))]


def test_kernel_pinned_uses_handle_policy(monkeypatch) -> None:
    events: list[tuple[str, object]] = []

    class FakePin:
        def __init__(self, policy: object) -> None:
            self.policy = policy

        def __enter__(self) -> None:
            events.append(("enter", self.policy))

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            events.append(("exit", self.policy))

    monkeypatch.setattr("veqlib.facade.kernel.pinned_cpu", FakePin)

    handle = Kernel(make_kernel_topology(), pin_cpu=9)
    with handle.pinned():
        events.append(("body", None))

    assert events == [("enter", 9), ("body", None), ("exit", 9)]


def test_thread_owned_solver_wraps_native_calls_in_pin_context(monkeypatch) -> None:
    events: list[tuple[str, object]] = []

    class FakePin:
        def __init__(self, policy: object) -> None:
            self.policy = policy

        def __enter__(self) -> None:
            events.append(("enter", self.policy))

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            events.append(("exit", self.policy))

    class RawSolver:
        def solve_direct(self) -> str:
            events.append(("solve", None))
            return "ok"

    monkeypatch.setattr("veqlib.facade.registry.pinned_cpu", FakePin)

    solver = ThreadOwnedKernelSolver(RawSolver(), pin_cpu=4)

    assert solver.solve_direct() == "ok"
    assert events == [("enter", 4), ("solve", None), ("exit", 4)]


def test_thread_owned_solver_skips_pin_context_inside_outer_scope(monkeypatch) -> None:
    events: list[str] = []

    class RawSolver:
        def solve_direct(self) -> str:
            events.append("solve")
            return "ok"

    def fail_pin(policy: object) -> object:
        raise AssertionError(f"unexpected inner pin context: {policy!r}")

    monkeypatch.setattr("veqlib.facade.registry.cpu_pin_scope_active", lambda: True)
    monkeypatch.setattr("veqlib.facade.registry.pinned_cpu", fail_pin)

    solver = ThreadOwnedKernelSolver(RawSolver(), pin_cpu=4)

    assert solver.solve_direct() == "ok"
    assert events == ["solve"]
