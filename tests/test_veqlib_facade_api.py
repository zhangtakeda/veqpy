from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import veqlib.facade as facade
from veqlib.facade import (
    Kernel,
    KernelBoundary,
    KernelBuild,
    KernelConfig,
    KernelInput,
    KernelResult,
    KernelTopology,
)

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
        scaled_heat=scaled_heat,
        scaled_current=scaled_current,
        scaled_Ip=3.0e6 * MU0,
        case_name=case_name,
    )


class RecordingSolver:
    def __init__(self) -> None:
        self.runtime_args: tuple[object, ...] | None = None

    def set_kernel_runtime(self, *args: object) -> None:
        self.runtime_args = args


def test_veqlib_facade_imports_without_importing_veqpy() -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_pythonpath
        else os.pathsep.join((str(PROJECT_ROOT), existing_pythonpath))
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


def test_veqlib_facade_root_exports_semantic_surface() -> None:
    assert facade.__all__ == [
        "Kernel",
        "KernelArtifact",
        "KernelBoundary",
        "KernelBuild",
        "KernelBuildError",
        "KernelCleanResult",
        "KernelConfig",
        "KernelInput",
        "KernelLoadError",
        "KernelRegistry",
        "KernelResult",
        "KernelTopology",
        "LoadedKernel",
        "SolverThreadError",
        "TopologyError",
        "VEQlibSolver",
        "build",
        "build_artifact",
        "clean",
        "solve",
    ]
    for helper in (
        "build_kernel",
        "default_kernel_cache_root",
        "pinned_cpu",
        "current_cpu_affinity",
        "solver_method_code",
        "SOLVER_METHOD_POWELL",
    ):
        assert not hasattr(facade, helper)


def test_kernel_topology_and_runtime_inputs_are_user_facing_contracts() -> None:
    topology = make_kernel_topology(c_counts=(0, 0), s_counts=(2, 0, 0), K_max=None)
    same_shape = make_kernel_topology(c_counts=(), s_counts=(2,), L_max=2, M_max=1, K_max=2)
    family_topology = topology.with_build(KernelBuild(layout="profile-first", build="release"))
    kernel_input = tiny_kernel_input(case_name="tiny")
    kernel_boundary = tiny_kernel_boundary()

    assert topology.to_canonical_dict() == same_shape.to_canonical_dict()
    assert topology.key == same_shape.key
    assert family_topology.layout == "family"
    assert topology.route == "PF"
    assert topology.coordinate == "psin"
    assert topology.constraint == "Ip"
    assert topology.sample_count == 9
    assert topology.packed_size() == 9

    assert kernel_input.case_name == "tiny"
    assert kernel_boundary.a == 0.5
    assert_allclose(kernel_input.scaled_heat, tiny_kernel_input().scaled_heat)
    assert kernel_input.scaled_Ip == 3.0e6 * MU0
    assert kernel_boundary.c_offsets.flags.c_contiguous
    assert kernel_boundary.s_offsets.flags.c_contiguous
    assert kernel_input.scaled_heat.flags.c_contiguous
    assert kernel_input.scaled_current.flags.c_contiguous
    assert not kernel_boundary.c_offsets.flags.writeable
    assert not kernel_boundary.s_offsets.flags.writeable
    assert not kernel_input.scaled_heat.flags.writeable
    assert not kernel_input.scaled_current.flags.writeable

    with pytest.raises(ValueError, match="scaled_heat must be 1D"):
        KernelInput(
            scaled_heat=np.ones((2, 1), dtype=np.float64),
            scaled_current=np.ones(2, dtype=np.float64),
        )


def test_kernel_runtime_case_must_match_topology_before_native() -> None:
    topology = make_kernel_topology()
    handle = Kernel(topology)
    recorder = RecordingSolver()
    handle._solver = recorder  # type: ignore[assignment]

    bad_source_length = KernelInput(
        scaled_heat=np.ones(topology.sample_count - 1, dtype=np.float64),
        scaled_current=np.ones(topology.sample_count - 1, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="case does not match kernel topology: scaled_heat"):
        handle._set_runtime(
            tiny_kernel_boundary(),
            bad_source_length,
            KernelConfig(),
            case_name=None,
        )
    assert recorder.runtime_args is None

    too_many_c_offsets = KernelBoundary(
        a=0.5,
        R0=1.0,
        Z0=0.0,
        B0=3.0,
        c_offsets=np.zeros(topology.M_max + 2, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="case does not match kernel topology: c_offsets"):
        handle._set_runtime(
            too_many_c_offsets,
            tiny_kernel_input(),
            KernelConfig(),
            case_name=None,
        )
    assert recorder.runtime_args is None

    too_many_s_offsets = KernelBoundary(
        a=0.5,
        R0=1.0,
        Z0=0.0,
        B0=3.0,
        s_offsets=np.zeros(topology.M_max + 2, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="case does not match kernel topology: s_offsets"):
        handle._set_runtime(
            too_many_s_offsets,
            tiny_kernel_input(),
            KernelConfig(),
            case_name=None,
        )
    assert recorder.runtime_args is None

    handle._set_runtime(
        tiny_kernel_boundary(),
        tiny_kernel_input(),
        KernelConfig(),
        case_name="override",
    )
    assert recorder.runtime_args is not None
    assert recorder.runtime_args[0] == "override"


def test_kernel_dry_run_and_python_owned_result_snapshot(tmp_path: Path) -> None:
    topology = make_kernel_topology()
    handle = facade.build(topology, cache_root=tmp_path, dry_run=True)

    assert isinstance(handle, Kernel)
    assert handle.x_size == 9

    raw_x = np.ones(3, dtype=np.float64)
    raw = np.full(3, 2.0, dtype=np.float64)
    scaled = np.full(3, 3.0, dtype=np.float64)
    alpha = np.array([4.0, 5.0], dtype=np.float64)
    result = KernelResult.from_solve_direct(
        (0.25, True, 1, 2, 3, 4, 5, 6, 7, 8.0, 9.0, raw_x, raw, scaled, alpha)
    )
    raw_x[:] = 99.0

    assert result.elapsed_ms == 0.25
    assert result.success is True
    assert result.x.tolist() == [1.0, 1.0, 1.0]
    assert result.x.flags.owndata
    assert result.raw.flags.owndata
    assert result.scaled.flags.owndata
    assert result.alpha.flags.owndata


@pytest.mark.slow
def test_kernel_python_build_and_solve_native_flow(tmp_path: Path) -> None:
    topology = make_kernel_topology()
    handle = Kernel(topology, build=KernelBuild(build="fastmath"), cache_root=tmp_path)

    artifact = handle.build()
    assert artifact.built is True
    assert artifact.shared_library_path.exists()

    kernel_boundary = tiny_kernel_boundary()
    kernel_input = tiny_kernel_input()
    result = handle.solve(
        kernel_boundary,
        kernel_input,
        config=KernelConfig(method="powell", initial="cold"),
    )
    assert result.success is True
    assert result.x.shape == (handle.x_size,)
    assert result.raw.shape == (handle.x_size,)
    assert result.scaled.shape == (handle.x_size,)
    assert_allclose(handle.residual(result.x, kernel_boundary, kernel_input), result.raw)

    residual_out = np.empty(handle.x_size, dtype=np.float64)
    handle.residual_into(residual_out, result.x, kernel_boundary, kernel_input)
    assert_allclose(residual_out, result.raw)

    jvp_out = np.empty(handle.x_size, dtype=np.float64)
    handle.jvp_into(
        jvp_out,
        result.x,
        np.ones(handle.x_size, dtype=np.float64),
        kernel_boundary,
        kernel_input,
    )
    assert jvp_out.shape == (handle.x_size,)

    jacobian_out = np.empty((handle.x_size, handle.x_size), dtype=np.float64)
    handle.jacobian_into(jacobian_out, result.x, kernel_boundary, kernel_input)
    assert jacobian_out.shape == (handle.x_size, handle.x_size)

    assert handle.jvp(
        result.x,
        np.ones(handle.x_size, dtype=np.float64),
        kernel_boundary,
        kernel_input,
    ).shape == (handle.x_size,)
    assert handle.jacobian(result.x, kernel_boundary, kernel_input).shape == (
        handle.x_size,
        handle.x_size,
    )

    handle.clear()
    assert handle.history == []
    assert handle.result is None
    handle.close()
