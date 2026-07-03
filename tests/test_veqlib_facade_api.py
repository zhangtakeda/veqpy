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
    KernelConfig,
    KernelRecipe,
    KernelSource,
    KernelTopology,
)
from veqlib.facade.abi import solve_result_from_native
from veqlib.facade.identity import recipe_identity_payload, topology_identity_payload
from veqlib.facade.options import (
    RESIDUAL_NORMALIZATION_BALANCED,
    SOLVER_METHOD_LEVENBERG_MARQUARDT,
    SOLVER_METHOD_POWELL,
)
from veqlib.facade.source_semantics import materialize_kernel_source

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
        "nodes": "uniform",
        "ip_constraint": True,
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


def tiny_kernel_source(*, case_name: str | None = None) -> KernelSource:
    psin = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    current_profile, scaled_heat = pf_reference_profiles(psin)
    return KernelSource(
        heat_profile=scaled_heat / MU0,
        current_profile=current_profile,
        Ip=3.0e6,
        case_name=case_name,
    )


class RecordingSolver:
    def __init__(self, *, x_size: int = 9) -> None:
        self.x_size = x_size
        self.runtime_args: tuple[object, ...] | None = None
        self.runtime_calls: list[tuple[object, ...]] = []

    def set_kernel_runtime(self, *args: object) -> None:
        self.runtime_args = args
        self.runtime_calls.append(args)

    def solve_direct(self) -> tuple[object, ...]:
        x = np.zeros(self.x_size, dtype=np.float64)
        alpha = np.zeros(2, dtype=np.float64)
        return (0.0, True, 1, 2, 3, 4, 5, 6, 7, 8.0, 9.0, x, x, x, alpha)


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
        "PrepareResult",
        "KernelBoundary",
        "KernelRecipe",
        "PrepareError",
        "CleanResult",
        "KernelConfig",
        "KernelSource",
        "KernelLoadError",
        "KernelRegistry",
        "SolveResult",
        "KernelTopology",
        "LoadedKernel",
        "SolverThreadError",
        "TopologyError",
        "VEQlibSolver",
        "build",
        "prepare",
        "clean",
        "solve",
        "materialize_kernel_source",
    ]


def test_kernel_topology_and_runtime_source_is_user_facing_contract() -> None:
    topology = make_kernel_topology(c_counts=(0, 0), s_counts=(2, 0, 0), K_max=None)
    same_shape = make_kernel_topology(c_counts=(), s_counts=(2,), L_max=2, M_max=1, K_max=2)
    family_recipe = KernelRecipe(layout="family", build="release")
    numba_recipe = KernelRecipe(backend="numba", layout="degree")
    kernel_source = tiny_kernel_source(case_name="tiny")
    materialized_source = materialize_kernel_source(topology, kernel_source)
    kernel_boundary = tiny_kernel_boundary()

    assert topology_identity_payload(topology) == topology_identity_payload(same_shape)
    assert topology.key == same_shape.key
    assert family_recipe.backend == "cxx"
    assert family_recipe.layout == "family"
    assert family_recipe.layout_profile_first is True
    assert numba_recipe.backend == "numba"
    assert numba_recipe.layout == "degree"
    assert recipe_identity_payload(family_recipe)["preset"] == "release"
    assert topology.route == "PF"
    assert topology.coordinate == "psin"
    assert topology.ip_constraint is True
    assert topology.sample_count == 9
    assert topology.x_size == 9

    assert kernel_source.case_name == "tiny"
    assert kernel_boundary.a == 0.5
    assert_allclose(kernel_source.heat_profile, tiny_kernel_source().heat_profile)
    assert kernel_source.Ip == 3.0e6
    assert not hasattr(kernel_source, "scaled_heat")
    assert not hasattr(kernel_source, "scaled_current")
    assert not hasattr(kernel_source, "scaled_Ip")
    assert_allclose(materialized_source.scaled_heat, tiny_kernel_source().heat_profile * MU0)
    assert_allclose(materialized_source.scaled_current, tiny_kernel_source().current_profile)
    assert materialized_source.scaled_Ip == 3.0e6 * MU0
    assert kernel_boundary.c_offsets.flags.c_contiguous
    assert kernel_boundary.s_offsets.flags.c_contiguous
    assert kernel_source.heat_profile.flags.c_contiguous
    assert kernel_source.current_profile.flags.c_contiguous
    assert not kernel_boundary.c_offsets.flags.writeable
    assert not kernel_boundary.s_offsets.flags.writeable
    assert not kernel_source.heat_profile.flags.writeable
    assert not kernel_source.current_profile.flags.writeable

    with pytest.raises(ValueError, match="heat_profile must be 1D"):
        KernelSource(
            heat_profile=np.ones((2, 1), dtype=np.float64),
            current_profile=np.ones(2, dtype=np.float64),
        )

    with pytest.raises(ValueError, match="veqlib.facade.Kernel only supports"):
        Kernel(topology=topology, recipe=numba_recipe)


@pytest.mark.parametrize(
    ("route", "current_profile", "expected_current"),
    [
        ("PF", np.array([1.0, 2.0, 3.0], dtype=np.float64), np.array([1.0, 2.0, 3.0])),
        ("PP", np.array([1.0, 2.0, 3.0], dtype=np.float64), np.array([1.0, 2.0, 3.0])),
        ("PI", np.array([1.0e6, 2.0e6, 3.0e6]), np.array([1.0e6, 2.0e6, 3.0e6]) * MU0),
        ("PJ1", np.array([1.0e6, 2.0e6, 3.0e6]), np.array([1.0e6, 2.0e6, 3.0e6]) * MU0),
        ("PJ2", np.array([1.0e6, 2.0e6, 3.0e6]), np.array([1.0e6, 2.0e6, 3.0e6]) * MU0),
        ("PQ", np.array([1.0, 2.0, 3.0], dtype=np.float64), np.array([1.0, 2.0, 3.0])),
    ],
)
def test_kernel_source_materialization_locks_route_scaling(
    route: str,
    current_profile: np.ndarray,
    expected_current: np.ndarray,
) -> None:
    topology = make_kernel_topology(
        route=route,
        coordinate="rho",
        nodes="uniform",
        sample_count=3,
        ip_constraint=False,
        psin_count=0,
        F_count=1 if route == "PJ2" else 0,
        h_count=1,
        kappa_count=0,
        s_counts=(),
    )
    heat_profile = np.array([1.0e6, 1.2e6, 1.4e6], dtype=np.float64)
    source = KernelSource(
        heat_profile=heat_profile,
        current_profile=current_profile,
        Ip=3.0e6,
    )

    materialized = materialize_kernel_source(topology, source)

    assert_allclose(materialized.scaled_heat, heat_profile * MU0)
    assert_allclose(materialized.scaled_current, expected_current)
    assert materialized.scaled_Ip == 3.0e6 * MU0
    assert not materialized.scaled_heat.flags.writeable
    assert not materialized.scaled_current.flags.writeable


def test_kernel_source_materialization_errors_use_raw_field_names() -> None:
    topology = make_kernel_topology(coordinate="rho", psin_count=0, sample_count=3)
    source = KernelSource(
        heat_profile=np.array([1.0e6, 1.1e6], dtype=np.float64),
        current_profile=np.ones(2, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="heat_profile and current_profile"):
        materialize_kernel_source(topology, source)

    prescaled_ip_source = KernelSource(
        heat_profile=np.full(3, 1.0e6, dtype=np.float64),
        current_profile=np.ones(3, dtype=np.float64),
        Ip=3.0e6 * MU0,
    )
    with pytest.warns(RuntimeWarning, match="Pass raw case values"):
        with pytest.raises(ValueError, match="Ip abs"):
            materialize_kernel_source(topology, prescaled_ip_source)


def test_kernel_runtime_case_must_match_topology_before_native() -> None:
    topology = make_kernel_topology()
    handle = Kernel(topology=topology)
    recorder = RecordingSolver()
    handle._solver = recorder  # type: ignore[assignment]

    bad_source_length = KernelSource(
        heat_profile=np.ones(topology.sample_count - 1, dtype=np.float64),
        current_profile=np.ones(topology.sample_count - 1, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="case does not match kernel topology: heat_profile"):
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
            tiny_kernel_source(),
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
            tiny_kernel_source(),
            KernelConfig(),
            case_name=None,
        )
    assert recorder.runtime_args is None

    handle._set_runtime(
        tiny_kernel_boundary(),
        tiny_kernel_source(),
        KernelConfig(),
        case_name="override",
    )
    assert recorder.runtime_args is not None
    assert recorder.runtime_args[0] == "override"
    assert_allclose(recorder.runtime_args[8], tiny_kernel_source().heat_profile * MU0)
    assert_allclose(recorder.runtime_args[9], tiny_kernel_source().current_profile)
    assert recorder.runtime_args[10] == 3.0e6 * MU0


def test_kernel_solve_uses_handle_default_config_with_per_call_overrides() -> None:
    topology = make_kernel_topology()
    default_config = KernelConfig(
        method="levenberg-marquardt",
        max_residual=2.0e-6,
        max_evaluations=123,
        norm="balanced",
    )
    handle = Kernel(topology=topology, config=default_config)
    recorder = RecordingSolver(x_size=handle.x_size)
    handle._solver = recorder  # type: ignore[assignment]

    handle.solve(tiny_kernel_boundary(), source=tiny_kernel_source(), case_name="default")
    assert recorder.runtime_args is not None
    assert recorder.runtime_args[0] == "default"
    assert recorder.runtime_args[12] == SOLVER_METHOD_LEVENBERG_MARQUARDT
    assert recorder.runtime_args[13] == default_config.max_residual
    assert recorder.runtime_args[14] == default_config.max_evaluations
    assert recorder.runtime_args[19] == RESIDUAL_NORMALIZATION_BALANCED

    handle.solve(
        tiny_kernel_boundary(),
        tiny_kernel_source(),
        method="powell",
        max_residual=3.0e-6,
        max_evaluations=None,
    )
    assert recorder.runtime_args[12] == SOLVER_METHOD_POWELL
    assert recorder.runtime_args[13] == 3.0e-6
    assert recorder.runtime_args[14] == handle.x_size * handle.x_size
    assert handle.config is default_config
    assert handle.config.method == "levenberg-marquardt"
    assert handle.config.max_evaluations == 123

    temporary_config = KernelConfig(method="powell", max_evaluations=5)
    handle.solve(
        tiny_kernel_boundary(),
        tiny_kernel_source(),
        config=temporary_config,
        method="levenberg-marquardt",
    )
    assert recorder.runtime_args[12] == SOLVER_METHOD_LEVENBERG_MARQUARDT
    assert recorder.runtime_args[14] == 5
    assert temporary_config.method == "powell"


def test_kernel_dry_run_and_python_owned_result_snapshot(tmp_path: Path) -> None:
    topology = make_kernel_topology()
    recipe = KernelRecipe(build="release", layout="family")
    kernel_config = KernelConfig(max_residual=4.0e-6)
    handle = facade.build(
        topology=topology,
        recipe=recipe,
        config=kernel_config,
        cache_root=tmp_path,
        dry_run=True,
    )
    artifact = facade.prepare(topology, recipe=recipe, cache_root=tmp_path, dry_run=True)
    default_artifact = facade.prepare(
        topology,
        recipe=KernelRecipe(build="fastmath", layout="degree"),
        cache_root=tmp_path,
        dry_run=True,
    )

    assert isinstance(handle, Kernel)
    assert handle.recipe is recipe
    assert handle.config is kernel_config
    assert handle.x_size == 9
    assert artifact.recipe is recipe
    assert artifact.topology.key == default_artifact.topology.key == topology.key
    assert artifact.artifact_id != default_artifact.artifact_id
    assert artifact.metadata["topology"] == topology_identity_payload(topology)
    assert artifact.metadata["recipe"] == recipe_identity_payload(recipe)
    assert artifact.metadata["build"]["build"] == "release"
    assert "-DVEQ_LAYOUT_PROFILE_FIRST=1" in artifact.metadata["build"]["cmake_configure"]

    raw_x = np.ones(3, dtype=np.float64)
    raw = np.full(3, 2.0, dtype=np.float64)
    scaled = np.full(3, 3.0, dtype=np.float64)
    alpha = np.array([4.0, 5.0], dtype=np.float64)
    result = solve_result_from_native(
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


def test_kernel_artifact_identity_excludes_runtime_case_and_solver_config(tmp_path: Path) -> None:
    topology = make_kernel_topology()
    recipe = KernelRecipe(build="release", layout="degree")
    first = Kernel(
        topology=topology,
        recipe=recipe,
        config=KernelConfig(max_residual=1.0e-5, max_evaluations=3),
        cache_root=tmp_path,
    )
    second = Kernel(
        topology=topology,
        recipe=recipe,
        config=KernelConfig(max_residual=9.0e-7, max_evaluations=99),
        cache_root=tmp_path,
    )
    materialize_kernel_source(topology, tiny_kernel_source(case_name="one"))
    materialize_kernel_source(
        topology,
        KernelSource(
            heat_profile=tiny_kernel_source().heat_profile * 1.1,
            current_profile=tiny_kernel_source().current_profile,
            Ip=4.0e6,
            case_name="two",
        ),
    )

    first_artifact = first.prepare(dry_run=True)
    second_artifact = second.prepare(dry_run=True)

    assert first_artifact.artifact_id == second_artifact.artifact_id
    assert first_artifact.metadata["topology"] == topology_identity_payload(topology)
    assert first_artifact.metadata["recipe"] == recipe_identity_payload(recipe)
    assert "config" not in first_artifact.metadata
    assert "source" not in first_artifact.metadata
    assert "x0" not in first_artifact.metadata


@pytest.mark.slow
def test_kernel_python_build_and_solve_native_flow(tmp_path: Path) -> None:
    topology = make_kernel_topology()
    handle = Kernel(
        topology=topology,
        recipe=KernelRecipe(build="fastmath"),
        cache_root=tmp_path,
    )

    artifact = handle.prepare()
    assert artifact.built is True
    assert artifact.shared_library_path.exists()

    kernel_boundary = tiny_kernel_boundary()
    kernel_source = tiny_kernel_source()
    result = handle.solve(
        kernel_boundary,
        kernel_source,
        config=KernelConfig(method="powell", initial="cold"),
    )
    assert result.success is True
    assert result.x.shape == (handle.x_size,)
    assert result.raw.shape == (handle.x_size,)
    assert result.scaled.shape == (handle.x_size,)
    assert_allclose(handle.residual(result.x, kernel_boundary, kernel_source), result.raw)

    residual_out = np.empty(handle.x_size, dtype=np.float64)
    handle.residual_into(residual_out, result.x, kernel_boundary, kernel_source)
    assert_allclose(residual_out, result.raw)

    jvp_out = np.empty(handle.x_size, dtype=np.float64)
    handle.jvp_into(
        jvp_out,
        result.x,
        np.ones(handle.x_size, dtype=np.float64),
        kernel_boundary,
        kernel_source,
    )
    assert jvp_out.shape == (handle.x_size,)

    jacobian_out = np.empty((handle.x_size, handle.x_size), dtype=np.float64)
    handle.jacobian_into(jacobian_out, result.x, kernel_boundary, kernel_source)
    assert jacobian_out.shape == (handle.x_size, handle.x_size)

    assert handle.jvp(
        result.x,
        np.ones(handle.x_size, dtype=np.float64),
        kernel_boundary,
        kernel_source,
    ).shape == (handle.x_size,)
    assert handle.jacobian(result.x, kernel_boundary, kernel_source).shape == (
        handle.x_size,
        handle.x_size,
    )

    handle.clear()
    assert handle.history == []
    assert handle.result is None
    handle.close()
