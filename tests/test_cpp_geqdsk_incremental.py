from __future__ import annotations

import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from veqpy.cpp import (
    INITIAL_POLICY_COLD,
    RESIDUAL_NORMALIZATION_FAST,
    SOLVER_METHOD_POWELL,
    KernelRegistry,
    VEQlibSolver,
)
from veqpy.operator import Operator
from veqpy.topology import Topology

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
FIGURE06_PATH = SCRIPT_DIR / "06-high-order-reconstructions.py"


@lru_cache(maxsize=1)
def _figure06_module() -> ModuleType:
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("veqpy_figure06_incremental", FIGURE06_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load Figure 06 script from {FIGURE06_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _case_specs() -> tuple[Any, ...]:
    return tuple(_figure06_module().CASE_SPECS)


def _profile_count(profile_coeffs: dict[str, list[float]], name: str) -> int:
    values = profile_coeffs.get(name)
    return 0 if values is None else int(np.asarray(values, dtype=np.float64).size)


def _family_counts(
    profile_coeffs: dict[str, list[float]],
    prefix: str,
    first: int,
) -> tuple[int, ...]:
    orders = [
        int(name[1:])
        for name in profile_coeffs
        if len(name) > 1 and name[0] == prefix and name[1:].isdigit()
    ]
    if not orders:
        return ()
    return tuple(
        _profile_count(profile_coeffs, f"{prefix}{order}")
        for order in range(first, max(orders) + 1)
    )


def _topology_from_case(case_spec: Any, geqdsk: Any) -> Topology:
    profile_coeffs = case_spec.profile_coeffs
    return Topology(
        h_count=_profile_count(profile_coeffs, "h"),
        v_count=_profile_count(profile_coeffs, "v"),
        kappa_count=_profile_count(profile_coeffs, "k"),
        psin_count=_profile_count(profile_coeffs, "psin"),
        F_count=_profile_count(profile_coeffs, "F"),
        c_counts=_family_counts(profile_coeffs, "c", 0),
        s_counts=_family_counts(profile_coeffs, "s", 1),
        Nr=int(case_spec.solve_nr),
        Nt=int(case_spec.solve_nt),
        route="PF",
        coordinate="psin",
        constraint="Ip",
        nodes="uniform",
        sample_count=int(np.asarray(geqdsk.P_psi, dtype=np.float64).size),
        M_max=int(case_spec.boundary_fit_m),
        K_max=max(2, int(case_spec.boundary_fit_m)),
    )


def _case_bundle(case_spec: Any) -> tuple[Any, Any, Any, Any, Topology, dict[str, Any]]:
    return _case_bundle_by_key(str(case_spec.case_key))


@lru_cache(maxsize=None)
def _case_bundle_by_key(case_key: str) -> tuple[Any, Any, Any, Any, Topology, dict[str, Any]]:
    fig06 = _figure06_module()
    case_spec = next(spec for spec in fig06.CASE_SPECS if spec.case_key == case_key)
    geqdsk = fig06.read_geqdsk(case_spec.gfile_path)
    boundary, fit = fig06.build_boundary(
        geqdsk,
        fit_m=int(case_spec.boundary_fit_m),
        fit_n=int(case_spec.boundary_fit_n),
    )
    case = fig06.build_solver_case(boundary, geqdsk, profile_coeffs=case_spec.profile_coeffs)
    topology = _topology_from_case(case_spec, geqdsk)
    payload = _case_payload(case_spec, geqdsk, case, boundary, fit, topology)
    return fig06, geqdsk, case, boundary, topology, payload


def _case_payload(
    case_spec: Any,
    geqdsk: Any,
    case: Any,
    boundary: Any,
    fit: dict[str, Any],
    topology: Topology,
) -> dict[str, Any]:
    c_offsets = np.asarray(boundary.c_offsets, dtype=np.float64)
    s_offsets = np.asarray(boundary.s_offsets, dtype=np.float64)
    return {
        "schema": "veqpy.cpp.geqdsk_case_payload.v1",
        "case_key": str(case_spec.case_key),
        "topology_key": topology.key,
        "topology": topology.to_canonical_dict(),
        "grid": {"Nr": int(case_spec.solve_nr), "Nt": int(case_spec.solve_nt)},
        "profiles": {
            "active_profiles": {
                str(name): int(count) for name, count in case.active_profiles.items()
            },
            "initial_coefficients": {
                str(name): np.asarray(values, dtype=np.float64).tolist()
                for name, values in case_spec.profile_coeffs.items()
            },
        },
        "boundary": {
            "a": float(boundary.a),
            "R0": float(boundary.R0),
            "Z0": float(boundary.Z0),
            "B0": float(boundary.B0),
            "ka": float(boundary.ka),
            "c_offsets": c_offsets.tolist(),
            "s_offsets": s_offsets.tolist(),
            "fit_rms": float(fit["rms"]),
        },
        "source": {
            "heat_input": np.asarray(case.heat_input, dtype=np.float64).tolist(),
            "current_input": np.asarray(case.current_input, dtype=np.float64).tolist(),
            "sample_count": int(np.asarray(case.heat_input, dtype=np.float64).size),
            "Ip": float(case.Ip),
            "geqdsk_shape": {"NR": int(geqdsk.NR), "NZ": int(geqdsk.NZ)},
        },
    }


def _kernel_payload(
    fig06: ModuleType,
    case_spec: Any,
    case: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    solve_grid = fig06.Grid(Nr=int(case_spec.solve_nr), Nt=int(case_spec.solve_nt))
    operator = Operator(solve_grid, case)
    coefficients = {
        str(name): np.asarray(values, dtype=np.float64)
        for name, values in case_spec.profile_coeffs.items()
    }
    x0 = operator.pack_coefficients(coefficients)
    source_plan = operator.plan.source_plan
    return {
        "case_name": str(case_spec.case_key),
        "boundary": payload["boundary"],
        "source": {
            "scaled_heat": source_plan.scaled_heat.tolist(),
            "scaled_current": source_plan.scaled_current.tolist(),
        },
        "constraints": {
            "scaled_Ip": float(source_plan.scaled_Ip),
            "fix_rho": float(operator.fix_rho),
        },
        "solver": {
            "method_code": SOLVER_METHOD_POWELL,
            "max_residual": 1.0e-6,
            "max_evaluations": int(x0.size) ** 2,
            "accepted_residual_factor": 10.0,
            "accepted_residual_floor": 1.0e-5,
            "initial_policy_code": INITIAL_POLICY_COLD,
            "residual_normalization_code": RESIDUAL_NORMALIZATION_FAST,
            "residual_normalization_floor": 1.0,
            "residual_normalization_max_ratio": 1.0e6,
            "residual_normalization_huber_tau": 3.0,
            "residual_normalization_probe_count": 4,
            "residual_normalization_probe_step": 1.0e-6,
            "residual_normalization_sensitivity_lambda": 0.5,
        },
    }


@pytest.mark.parametrize("case_spec", _case_specs(), ids=lambda spec: spec.case_key)
def test_figure06_geqdsk_cases_define_cpp_topology_and_payloads(case_spec: Any) -> None:
    _fig06, _geqdsk, case, _boundary, topology, payload = _case_bundle(case_spec)

    assert topology.build == "fastmath"
    assert topology.M_max == case_spec.boundary_fit_m
    assert topology.key == payload["topology_key"]
    assert payload["source"]["sample_count"] == topology.sample_count
    assert len(payload["source"]["heat_input"]) == topology.sample_count
    assert len(payload["source"]["current_input"]) == topology.sample_count
    assert payload["profiles"]["active_profiles"] == case.active_profiles
    topology.validate_supported_for_veqlib_mvp()


@pytest.mark.slow
def test_figure06_cpp_kernels_load_multiple_topologies_in_one_process(tmp_path: Path) -> None:
    registry = KernelRegistry(cache_root=tmp_path)
    cxx_results: dict[str, Any] = {}

    for case_spec in _case_specs():
        fig06, _geqdsk, case, _boundary, topology, payload = _case_bundle(case_spec)
        solver = VEQlibSolver(topology, registry=registry)
        solver.set_case_json(
            json.dumps(
                _kernel_payload(fig06, case_spec, case, payload),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        metadata = solver.metadata()

        assert metadata["grid"]["M_max"] == int(case_spec.boundary_fit_m)
        assert metadata["grid"]["Nr"] == int(case_spec.solve_nr)
        assert metadata["grid"]["Nt"] == int(case_spec.solve_nt)
        assert metadata["source"]["sample_count"] == topology.sample_count
        cxx_results[str(case_spec.case_key)] = solver.solve_direct()

    solovev_spec = next(spec for spec in _case_specs() if spec.case_key == "solovev")
    fig06, _geqdsk, solovev_case, _boundary, _topology, _payload = _case_bundle(solovev_spec)
    py_solver = fig06.build_solver(
        solovev_case,
        fig06.Grid(Nr=int(solovev_spec.solve_nr), Nt=int(solovev_spec.solve_nt)),
    )
    py_solver, _elapsed_ms, _wall_ms = fig06.solve_existing_solver_once(py_solver)
    assert py_solver.result is not None
    assert py_solver.result.success is True

    solovev_cxx = cxx_results["solovev"]
    assert bool(solovev_cxx[1]) is True
    np.testing.assert_allclose(
        np.asarray(solovev_cxx[11], dtype=np.float64),
        np.asarray(py_solver.result.x, dtype=np.float64),
        rtol=1.0e-7,
        atol=5.0e-8,
    )

    # These two are intentionally not used as parity gates yet: they exercise
    # the multi-topology nanobind architecture, while solver-strategy parity is
    # tracked separately.
    assert len(cxx_results["chease"]) == 15
    assert len(cxx_results["efit"]) == 15
