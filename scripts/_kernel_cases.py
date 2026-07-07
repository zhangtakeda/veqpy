"""Kernel and GEQDSK case helpers for manuscript scripts."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from _cases import (
    BOUNDARY_MAXTOL,
    CASE_BOUNDARY_FIT_M,
    CASE_BOUNDARY_FIT_N,
    CASE_LABELS,
    CASE_REFERENCE_EQUILIBRIUM_JSONS,
    CASE_REFERENCE_GFILES,
    CASE_REFERENCE_PROFILE_LENGTHS,
    MU0,
    REDUCED_EQUILIBRIUM_JSON_TEMPLATE,
    REDUCED_EQUILIBRIUM_MANIFEST_PATH,
    REFERENCE_EQUILIBRIUM_MANIFEST_PATH,
    REFERENCE_LAYOUT_NR,
    REFERENCE_LAYOUT_NT,
    REFERENCE_SOLVER_MAXFEV,
)


@dataclass(frozen=True)
class PreparedInterpAxis:
    unique_axis: np.ndarray
    order: np.ndarray
    unique_index: np.ndarray


@dataclass(frozen=True)
class PfReferenceCase:
    case_key: str
    boundary: object
    geqdsk: object
    equilibrium: object
    ref_profiles: dict[str, np.ndarray | float]
    psin_interp_axis: PreparedInterpAxis


@dataclass(frozen=True)
class ScriptKernelCase:
    name: str
    topology: object
    boundary: object
    source: object
    config: object
    active_profiles: dict[str, int] | None = None


def demo_psin_reference_profiles(psin):
    """Return the PF source profiles shared by Figures 03 and 04."""
    import numpy as np

    psin = np.asarray(psin, dtype=np.float64)
    beta0 = 0.75
    alpha_p, alpha_f = 5.0, 3.32
    exp_ap = np.exp(alpha_p)
    exp_af = np.exp(alpha_f)
    den_p = 1.0 + exp_ap * (alpha_p - 1.0)
    den_f = 1.0 + exp_af * (alpha_f - 1.0)

    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    return current_input, heat_input


@lru_cache(maxsize=1)
def load_veqpy_components() -> dict[str, object]:
    from veqpy import (
        Kernel,
        KernelBoundary,
        KernelConfig,
        KernelRecipe,
        KernelSource,
        KernelTopology,
    )
    from veqpy.kernels.boundary_materialization import (
        materialize_kernel_boundary,
        materialized_boundary_fit_payload,
    )
    from veqpy.kernels.numba_kernel.packed_layout import (
        build_profile_index,
        build_profile_layout,
        build_profile_names,
        build_shape_profile_names,
    )
    from veqpy.kernels.types import kernel_boundary_shape_orders
    from veqpy.model import Equilibrium, Geqdsk, Grid

    return {
        "Equilibrium": Equilibrium,
        "Geqdsk": Geqdsk,
        "Grid": Grid,
        "Kernel": Kernel,
        "KernelBoundary": KernelBoundary,
        "KernelConfig": KernelConfig,
        "KernelRecipe": KernelRecipe,
        "KernelSource": KernelSource,
        "KernelTopology": KernelTopology,
        "build_profile_index": build_profile_index,
        "build_profile_layout": build_profile_layout,
        "build_profile_names": build_profile_names,
        "build_shape_profile_names": build_shape_profile_names,
        "kernel_boundary_shape_orders": kernel_boundary_shape_orders,
        "materialize_kernel_boundary": materialize_kernel_boundary,
        "materialized_boundary_fit_payload": materialized_boundary_fit_payload,
    }


def active_profiles_from_coeffs(profile_coeffs: Mapping[str, object]) -> dict[str, int]:
    active_profiles: dict[str, int] = {}
    for name, coeff in profile_coeffs.items():
        if coeff is None:
            continue
        if isinstance(coeff, (int, np.integer)):
            length = int(coeff)
        else:
            length = int(np.asarray(coeff, dtype=np.float64).size)
        if length > 0:
            active_profiles[str(name)] = length
    return active_profiles


def coefficients_from_profile_coeffs(profile_coeffs: Mapping[str, object]) -> dict[str, np.ndarray]:
    coefficients: dict[str, np.ndarray] = {}
    for name, coeff in profile_coeffs.items():
        if coeff is None:
            continue
        if isinstance(coeff, (int, np.integer)):
            length = int(coeff)
            if length <= 0:
                continue
            coeff_array = np.zeros(length, dtype=np.float64)
        else:
            coeff_array = np.asarray(coeff, dtype=np.float64)
            if coeff_array.size <= 0:
                continue
        coefficients[str(name)] = coeff_array
    return coefficients


def read_geqdsk(path: str):
    geqdsk = load_veqpy_components()["Geqdsk"]()
    geqdsk.read_geqdsk(str(path))
    return geqdsk


def load_equilibrium_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing equilibrium JSON: {path}")
    return load_veqpy_components()["Equilibrium"].load(path)


def as_float64_array(values, *, copy: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr.copy() if copy else arr


def prepare_interp_axis(axis: np.ndarray) -> PreparedInterpAxis:
    axis_f64 = as_float64_array(axis)
    order = np.argsort(axis_f64)
    axis_sorted = axis_f64[order]
    unique_axis, unique_index = np.unique(axis_sorted, return_index=True)
    return PreparedInterpAxis(unique_axis=unique_axis, order=order, unique_index=unique_index)


def prepare_interp_values(values: np.ndarray, prepared_axis: PreparedInterpAxis) -> np.ndarray:
    values_f64 = as_float64_array(values)
    return values_f64[prepared_axis.order][prepared_axis.unique_index]


def profile_interp(
    axis: np.ndarray | PreparedInterpAxis, values: np.ndarray, x_new: np.ndarray
) -> np.ndarray:
    from scipy.interpolate import PchipInterpolator

    prepared_axis = axis if isinstance(axis, PreparedInterpAxis) else prepare_interp_axis(axis)
    unique_axis = prepared_axis.unique_axis
    unique_values = prepare_interp_values(values, prepared_axis)
    x_new = as_float64_array(x_new)
    if unique_axis.size < 2:
        fill_value = float(unique_values[0] if unique_values.size else 0.0)
        return np.full_like(x_new, fill_value, dtype=np.float64)
    if unique_axis.size < 3:
        return np.interp(x_new, unique_axis, unique_values).astype(np.float64, copy=False)
    return as_float64_array(PchipInterpolator(unique_axis, unique_values, extrapolate=True)(x_new))


def build_pf_reference_profiles(equilibrium) -> dict[str, np.ndarray | float]:
    psin_r = as_float64_array(equilibrium.psin_r, copy=True)
    psin_r_safe = np.where(np.abs(psin_r) > 1.0e-14, psin_r, 1.0e-14)
    pn_psin = as_float64_array(equilibrium.Pn_r, copy=True) / psin_r_safe
    return {
        "psin": as_float64_array(equilibrium.psin, copy=True),
        "FFn_psin": as_float64_array(equilibrium.FFn_r, copy=True) / psin_r_safe,
        "Pn_psin": pn_psin,
        "setup_Pn_psin": pn_psin / MU0,
    }


def _family_counts(signature: Mapping[str, int], prefix: str, *, start: int) -> tuple[int, ...]:
    values: list[int] = []
    order = start
    while True:
        key = f"{prefix}{order}"
        if key not in signature:
            break
        values.append(int(signature[key]))
        order += 1
    while values and values[-1] == 0:
        values.pop()
    return tuple(values)


def profile_counts_from_signature(
    signature: Mapping[str, int],
    *,
    route: str = "PF",
    coordinate: str = "psin",
    nodes: str = "uniform",
    pj2_f_count: int = 6,
) -> dict[str, object]:
    route_key = str(route).upper()
    coordinate_key = str(coordinate).lower()
    nodes_key = str(nodes).lower()
    psin_count = int(signature.get("psin", 0))
    f_count = int(signature.get("F", 0))
    if route_key == "PJ2" and f_count <= 0:
        f_count = int(pj2_f_count)
    if route_key == "PJ2":
        psin_count = 0
    elif not (coordinate_key == "psin" and nodes_key == "uniform"):
        psin_count = 0
    return {
        "h_count": int(signature.get("h", 0)),
        "v_count": int(signature.get("v", 0)),
        "kappa_count": int(signature.get("k", 0)),
        "psin_count": psin_count,
        "F_count": f_count,
        "c_counts": _family_counts(signature, "c", start=0),
        "s_counts": _family_counts(signature, "s", start=1),
    }


def build_kernel_topology(
    signature: Mapping[str, int],
    *,
    nr: int,
    nt: int,
    route: str = "PF",
    coordinate: str = "psin",
    nodes: str = "uniform",
    ip_constraint: bool = True,
    beta_constraint: bool = False,
    sample_count: int | None = None,
    m_max: int | None = None,
    k_max: int | None = None,
) -> object:
    components = load_veqpy_components()
    return components["KernelTopology"](
        **profile_counts_from_signature(
            signature,
            route=route,
            coordinate=coordinate,
            nodes=nodes,
        ),
        Nr=int(nr),
        Nt=int(nt),
        route=str(route),
        coordinate=str(coordinate),
        nodes=str(nodes),
        ip_constraint=bool(ip_constraint),
        beta_constraint=bool(beta_constraint),
        sample_count=sample_count,
        M_max=m_max,
        K_max=k_max,
    )


def kernel_config(
    *,
    method: str = "powell",
    max_residual: float = 1.0e-6,
    max_evaluations: int | None = None,
    initial: str = "cold",
    continuation: str = "cold",
    norm: str = "fast",
) -> object:
    components = load_veqpy_components()
    return components["KernelConfig"](
        method=method,
        max_residual=float(max_residual),
        max_evaluations=max_evaluations,
        initial=initial,
        continuation=continuation,
        norm=norm,
    )


def solve_script_kernel_case(
    case: ScriptKernelCase,
    *,
    backend: str = "numba",
    layout: str = "degree",
):
    components = load_veqpy_components()
    kernel = components["Kernel"](
        topology=case.topology,
        recipe=components["KernelRecipe"](backend=backend, layout=layout),
        config=case.config,
    )
    result = kernel.solve(case.boundary, case.source)
    return result, kernel


def build_geqdsk_boundary(geqdsk, *, fit_m: int, fit_n: int, return_fit: bool = False):
    components = load_veqpy_components()
    boundary = components["KernelBoundary"](
        B0=float(geqdsk.Bt0),
        R_boundary=np.asarray(geqdsk.boundary[:, 0], dtype=np.float64),
        Z_boundary=np.asarray(geqdsk.boundary[:, 1], dtype=np.float64),
        c_order=int(fit_m),
        s_order=int(fit_n),
        fit_maxtol=BOUNDARY_MAXTOL,
    )
    materialized = components["materialize_kernel_boundary"](boundary)
    fit = components["materialized_boundary_fit_payload"](materialized)
    return (boundary, fit) if return_fit else boundary


def load_pf_benchmark(backend: str):
    os.environ["VEQPY_BACKEND"] = str(backend)
    components = load_veqpy_components()
    reference_grid = components["Grid"](
        Nr=REFERENCE_LAYOUT_NR,
        Nt=REFERENCE_LAYOUT_NT,
        quadrature_scheme="legendre",
    )
    config = kernel_config(
        method="powell",
        max_residual=1.0e-6,
        max_evaluations=REFERENCE_SOLVER_MAXFEV,
        initial="cold",
        continuation="cold",
        norm="none",
    )
    return SimpleNamespace(
        BACKEND=str(backend),
        Grid=components["Grid"],
        Kernel=components["Kernel"],
        KernelRecipe=components["KernelRecipe"],
        CONFIG=config,
        REFERENCE_GRID=reference_grid,
    )


def build_pf_reference_case(case_key: str) -> PfReferenceCase:
    equilibrium = load_equilibrium_json(CASE_REFERENCE_EQUILIBRIUM_JSONS[case_key])
    geqdsk = read_geqdsk(CASE_REFERENCE_GFILES[case_key])
    boundary = build_geqdsk_boundary(
        geqdsk,
        fit_m=CASE_BOUNDARY_FIT_M[case_key],
        fit_n=CASE_BOUNDARY_FIT_N[case_key],
    )
    return PfReferenceCase(
        case_key=case_key,
        boundary=boundary,
        geqdsk=geqdsk,
        equilibrium=equilibrium,
        ref_profiles=build_pf_reference_profiles(equilibrium),
        psin_interp_axis=prepare_interp_axis(np.asarray(equilibrium.psin, dtype=np.float64)),
    )


def make_profile_coeffs(
    signature: dict[str, int],
    *,
    max_lengths: dict[str, int],
) -> dict[str, list[float] | None]:
    profile_coeffs: dict[str, list[float] | None] = {name: None for name in max_lengths}
    for name, length in signature.items():
        coeff_length = int(length)
        if coeff_length > 0:
            profile_coeffs[name] = [0.0] * coeff_length
    return profile_coeffs


def build_pf_case(benchmark, reference: PfReferenceCase, signature: dict[str, int], grid=None):
    grid = benchmark.REFERENCE_GRID if grid is None else grid
    components = load_veqpy_components()
    normalized_signature = normalize_signature(signature)
    active_profiles = active_profiles_from_coeffs(
        make_profile_coeffs(
            normalized_signature,
            max_lengths=CASE_REFERENCE_PROFILE_LENGTHS[reference.case_key],
        )
    )
    c_order, s_order = components["kernel_boundary_shape_orders"](reference.boundary)
    boundary_m_max = max(c_order, s_order, int(grid.M_max), 1)
    topology = build_kernel_topology(
        normalized_signature,
        nr=int(grid.Nr),
        nt=int(grid.Nt),
        route="PF",
        coordinate="psin",
        nodes="uniform",
        ip_constraint=True,
        sample_count=int(np.asarray(reference.geqdsk.P_psi).size),
        m_max=boundary_m_max,
        k_max=max(2, boundary_m_max),
    )
    return ScriptKernelCase(
        name=f"{reference.case_key}-PF-psin-uniform-Ip",
        topology=topology,
        boundary=reference.boundary,
        source=components["KernelSource"](
            heat_profile=np.asarray(reference.geqdsk.P_psi, dtype=np.float64),
            current_profile=np.asarray(reference.geqdsk.FF_psi, dtype=np.float64),
            Ip=float(reference.geqdsk.Ip),
            beta=np.nan,
            case_name=f"{reference.case_key}-PF",
        ),
        config=kernel_config(
            method="powell",
            max_residual=1.0e-6,
            max_evaluations=REFERENCE_SOLVER_MAXFEV,
            initial="cold",
            continuation="cold",
            norm="none",
        ),
        active_profiles=active_profiles,
    )


def reduced_equilibrium_json_path(case_key: str, config_label: str) -> str:
    return REDUCED_EQUILIBRIUM_JSON_TEMPLATE.format(
        case_key=str(case_key),
        config_label=str(config_label).lower(),
    )


def load_reduced_equilibrium_manifest(
    path: str | None = None,
) -> dict[tuple[str, str], dict[str, object]]:
    path = REDUCED_EQUILIBRIUM_MANIFEST_PATH if path is None else str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing reduced-equilibrium manifest: {path}. "
            "Run `python scripts/07-pareto-analysis.py` first."
        )
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    manifest: dict[tuple[str, str], dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        case_key = str(entry.get("case_key", ""))
        config_label = str(entry.get("config_label", ""))
        if case_key and config_label:
            manifest[(case_key, config_label)] = entry
    return manifest


def load_reference_equilibrium_manifest(
    path: str | None = None,
) -> dict[tuple[str, str], dict[str, object]]:
    path = REFERENCE_EQUILIBRIUM_MANIFEST_PATH if path is None else str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing reference-equilibrium manifest: {path}. "
            "Run `python scripts/06-high-order-reconstructions.py` first."
        )
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    manifest: dict[tuple[str, str], dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        case_key = str(entry.get("case_key", ""))
        config_label = str(entry.get("config_label", ""))
        if case_key and config_label:
            manifest[(case_key, config_label)] = entry
    return manifest


def manifest_entry(
    manifest: dict[tuple[str, str], dict[str, object]],
    case_key: str,
    config_label: str,
) -> dict[str, object]:
    entry = manifest.get((case_key, config_label))
    if entry is None:
        raise FileNotFoundError(
            f"Missing {CASE_LABELS[case_key]} {config_label} "
            f"entry in {REDUCED_EQUILIBRIUM_MANIFEST_PATH}. "
            "Run `python scripts/07-pareto-analysis.py` first."
        )
    return entry


def reference_manifest_entry(
    manifest: dict[tuple[str, str], dict[str, object]], case_key: str
) -> dict[str, object]:
    entry = manifest.get((case_key, "Ref"))
    if entry is None:
        raise FileNotFoundError(
            f"Missing {CASE_LABELS[case_key]} Ref entry in "
            f"{REFERENCE_EQUILIBRIUM_MANIFEST_PATH}. "
            "Run `python scripts/06-high-order-reconstructions.py` first."
        )
    return entry


def metadata_float(
    entry: dict[str, object], key: str, default: float | None = None
) -> float | None:
    value = entry.get(key)
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    import math

    return parsed if math.isfinite(parsed) else default


def metadata_int(entry: dict[str, object], key: str) -> int | None:
    value = entry.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_signature(signature: dict[str, int]) -> dict[str, int]:
    return {str(name): int(length) for name, length in sorted(signature.items()) if int(length) > 0}


def signature_from_metadata(entry: dict[str, object]) -> dict[str, int]:
    signature = entry.get("signature", {})
    return normalize_signature(signature) if isinstance(signature, dict) else {}
