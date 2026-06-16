"""
Module: engine.numba_source

Role:
- Register concrete source routes.
- Validate route/coordinate/nodes triples and execute source kernels.

Public API:
- register_source_route
- validate_route
- build_source_remap_cache
- resolve_source_inputs

Notes:
- Source route routing stays here.
- The operator layer only binds one source runner and uses it as the Stage-C entrypoint.
- Each route must fill the same normalized root/source contract:
  psin, psin_r, psin_rr, Pn_psin, FFn_psin, alpha1, and alpha2.  The route
  name only changes which user source profile is treated as primitive.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numba import njit

try:
    from veqpy.base.registry import Registry
except ModuleNotFoundError as exc:
    if exc.name != "orjson":
        raise
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    _registry_path = Path(__file__).resolve().parents[1] / "base" / "registry.py"
    _registry_spec = spec_from_file_location("_veqpy_base_registry", _registry_path)
    if _registry_spec is None or _registry_spec.loader is None:
        raise
    _registry_module = module_from_spec(_registry_spec)
    _registry_spec.loader.exec_module(_registry_module)
    Registry = _registry_module.Registry
from veqpy.math.fast import (
    copy_into,
    dot,
    matvec_into,
    product_into,
    scale_into,
    scaled_product_into,
    scaled_product_ratio_into,
    scaled_ratio_into,
    weighted_dot,
)
from veqpy.math.interpolate import (
    DEFAULT_LOCAL_BARYCENTRIC_STENCIL,
    build_uniform_source_interpolation_matrix,
)
from veqpy.workspace.field_rows import (
    GEOMETRY_RADIAL_KN,
    GEOMETRY_RADIAL_KN_R,
    GEOMETRY_RADIAL_LN_R,
    GEOMETRY_RADIAL_S_R,
    GEOMETRY_RADIAL_V_R,
    GEOMETRY_SURFACE_JDIVR,
    GEOMETRY_SURFACE_R,
    GRID_RADIAL_RHO,
    RESIDUAL_ROOT_PSIN,
    RESIDUAL_ROOT_PSIN_R,
    RESIDUAL_ROOT_PSIN_RR,
)

# PJ2-psin-uniform is the only route that materializes psin by a
# fixed-point loop. Keep these as route constants instead of user-facing
# source-plan parameters.
PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_ITER = 16
PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_RESIDUAL = 1.0e-10
PJ2_PSIN_UNIFORM_FIXED_POINT_FINALIZE_ITER = 8
PJ2_PSIN_UNIFORM_BARYCENTRIC_ORDER_CAP = 8

RHO_COORDINATE = 0
PSIN_COORDINATE = 1

COORDINATE_NAMES = {
    RHO_COORDINATE: "rho",
    PSIN_COORDINATE: "psin",
}

COORDINATE_CODES = {
    "rho": RHO_COORDINATE,
    "psin": PSIN_COORDINATE,
}

UNIFORM_NODES = "uniform"
GRID_NODES = "grid"
NODE_NAMES = (UNIFORM_NODES, GRID_NODES)

SOURCE_PARAMETERIZATION_IDENTITY = "identity"
SOURCE_PARAMETERIZATION_SQRT_PSIN = "sqrt_psin"
SOURCE_PARAMETERIZATION_CODE_IDENTITY = 0
SOURCE_PARAMETERIZATION_CODE_SQRT_PSIN = 1

# Scratch slot indices into SourceWorkspace.array_scratch (7 + Nr rows × Nr).  These
# symbolic names are part of the hot-kernel ABI with SourceWorkspace; changing
# the row order requires updating the allocator at the same time.
_SLOT_INTEGRAND = 0
_SLOT_AUX0 = 1
_SLOT_AUX1 = 2
_SLOT_AUX2 = 3
_SLOT_PNr = 4
_SLOT_Pr = 5
_SLOT_Fr = 6
_SLOT_PQ_MATRIX = 7

RouteKey = tuple[str, str, str]

SOURCE_ROUTE_KEYS: tuple[RouteKey, ...] = (
    ("PF", "rho", "uniform"),
    ("PF", "rho", "grid"),
    ("PF", "psin", "uniform"),
    ("PF", "psin", "grid"),
    ("PP", "rho", "uniform"),
    ("PP", "rho", "grid"),
    ("PP", "psin", "uniform"),
    ("PP", "psin", "grid"),
    ("PI", "rho", "uniform"),
    ("PI", "rho", "grid"),
    ("PI", "psin", "uniform"),
    ("PI", "psin", "grid"),
    ("PJ1", "rho", "uniform"),
    ("PJ1", "rho", "grid"),
    ("PJ1", "psin", "uniform"),
    ("PJ1", "psin", "grid"),
    ("PJ2", "rho", "uniform"),
    ("PJ2", "rho", "grid"),
    ("PJ2", "psin", "uniform"),
    ("PJ2", "psin", "grid"),
    ("PQ", "rho", "uniform"),
    ("PQ", "rho", "grid"),
    ("PQ", "psin", "uniform"),
    ("PQ", "psin", "grid"),
)
SOURCE_ROUTE_KEY_SET: frozenset[RouteKey] = frozenset(SOURCE_ROUTE_KEYS)


@dataclass(frozen=True, slots=True)
class _SourceRouteSpec:
    """Registered concrete source-route implementation metadata."""

    route: str
    coordinate: str
    coordinate_code: int
    nodes: str
    implementation: Callable


SOURCE_ROUTE_KERNELS: Registry[RouteKey, Callable] = Registry(tuple, Callable)
ROUTE_REGISTRY: dict[RouteKey, _SourceRouteSpec] = {}


def _normalize_route_key(value: RouteKey | str) -> RouteKey:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("Source route key must be a three-string tuple: (route, coordinate, nodes)")
    route, coordinate, nodes = value
    if not isinstance(route, str) or not isinstance(coordinate, str) or not isinstance(nodes, str):
        raise TypeError(
            "Source route key must contain strings only: "
            f"got {type(route).__name__}, {type(coordinate).__name__}, {type(nodes).__name__}"
        )
    return (
        route.upper(),
        COORDINATE_NAMES[_normalize_coordinate(coordinate)],
        _normalize_nodes(nodes),
    )


def _normalize_coordinate(value: str) -> int:
    coordinate = str(value).lower()
    try:
        return COORDINATE_CODES[coordinate]
    except KeyError as exc:
        raise ValueError(f"Unsupported coordinate {value!r}") from exc


def _normalize_nodes(value: str) -> str:
    nodes = str(value).lower()
    if nodes not in NODE_NAMES:
        raise ValueError(f"Unsupported nodes {value!r}")
    return nodes


def register_source_route(*route_keys: RouteKey) -> Callable[[Callable], Callable]:
    """Register one implementation for one or more canonical source route tuples.

    Source route registration is intentionally tuple-only at the engine boundary:
    each key must be a three-string tuple such as ``("PJ1", "rho", "uniform")``.
    Friendly route-name compatibility belongs at the model/source-plan boundary, not
    in this bind-time registry.
    """

    if not route_keys:
        raise ValueError("At least one source route key is required")

    normalized_keys = tuple(_normalize_route_key(route_key) for route_key in route_keys)

    if len(set(normalized_keys)) != len(normalized_keys):
        raise ValueError(f"Duplicate source route keys in registration: {normalized_keys!r}")

    def decorator(func: Callable) -> Callable:
        for normalized_key in normalized_keys:
            if normalized_key in ROUTE_REGISTRY:
                raise ValueError(f"Source route {normalized_key!r} is already registered")

        SOURCE_ROUTE_KERNELS(*normalized_keys)(func)
        for normalized_key in normalized_keys:
            coordinate_code = _normalize_coordinate(normalized_key[1])
            ROUTE_REGISTRY[normalized_key] = _SourceRouteSpec(
                route=normalized_key[0],
                coordinate=normalized_key[1],
                coordinate_code=coordinate_code,
                nodes=normalized_key[2],
                implementation=func,
            )
        return func

    return decorator


def validate_route(route: str, coordinate: str, nodes: str = UNIFORM_NODES) -> _SourceRouteSpec:
    """Validate a concrete ``(route, coordinate, nodes)`` source route."""

    key = _normalize_route_key((route, coordinate, nodes))
    try:
        return ROUTE_REGISTRY[key]
    except KeyError as exc:
        supported = ", ".join("/".join(route_key) for route_key in sorted(ROUTE_REGISTRY))
        raise KeyError(
            f"Unknown source route {route!r}/{coordinate!r}/{nodes!r}; supported: {supported}"
        ) from exc


def source_parameterization_for_route_key(route_key: RouteKey | str) -> str:
    """Return the source-input parameterization for a registered concrete route key."""

    normalized_key = _normalize_route_key(route_key)
    if normalized_key not in ROUTE_REGISTRY:
        supported = ", ".join("/".join(route_key) for route_key in sorted(ROUTE_REGISTRY))
        raise KeyError(f"Unknown source route {normalized_key!r}; supported: {supported}")
    if normalized_key == ("PP", "psin", "uniform"):
        # This public source axis is sqrt(psin) to give uniform samples more
        # resolution near the magnetic axis.  Internal root fields stay in psin.
        return SOURCE_PARAMETERIZATION_SQRT_PSIN
    return SOURCE_PARAMETERIZATION_IDENTITY


@njit(cache=True, nogil=True)
def _source_output_root_views(
    out_root_fields: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        out_root_fields[RESIDUAL_ROOT_PSIN],
        out_root_fields[RESIDUAL_ROOT_PSIN_R],
        out_root_fields[RESIDUAL_ROOT_PSIN_RR],
    )


@njit(cache=True, nogil=True)
def _source_geometry_workspace_views(
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Keep the tuple order synchronized with GeometryWorkspace row contracts.
    # Named unpacking at call sites is the only documentation numba preserves.
    return (
        radial_fields[GEOMETRY_RADIAL_V_R],
        radial_fields[GEOMETRY_RADIAL_KN],
        radial_fields[GEOMETRY_RADIAL_KN_R],
        radial_fields[GEOMETRY_RADIAL_LN_R],
        radial_fields[GEOMETRY_RADIAL_S_R],
        surface_fields[GEOMETRY_SURFACE_R],
        surface_fields[GEOMETRY_SURFACE_JDIVR],
    )


@njit(cache=True, nogil=True)
def _source_grid_rho(grid_radial_fields: np.ndarray) -> np.ndarray:
    return grid_radial_fields[GRID_RADIAL_RHO]


@njit(cache=True, nogil=True)
def full_differentiation(
    out: np.ndarray, arr: np.ndarray, differentiator: np.ndarray
) -> np.ndarray:
    """Execute full radial differentiation."""
    matvec_into(out, differentiator, arr)
    return out


@njit(cache=True, nogil=True)
def full_integration(out: np.ndarray, arr: np.ndarray, accumulator: np.ndarray) -> np.ndarray:
    """Execute full radial integration."""
    matvec_into(out, accumulator, arr)
    return out


@njit(cache=True, nogil=True)
def _update_psin_coordinate(
    out_psin: np.ndarray,
    psin_r: np.ndarray,
    accumulator: np.ndarray,
) -> np.ndarray:
    # Source kernels solve for a positive psin_r first.  The coordinate itself
    # is then integrated and normalized to the external [0, 1] flux convention.
    full_integration(out_psin, psin_r, accumulator)
    return _normalize_psin_coordinate_inplace(out_psin)


@njit(cache=True, nogil=True)
def _normalize_psin_coordinate_inplace(psin: np.ndarray) -> np.ndarray:
    offset = psin[0]
    scale = psin[-1] - offset
    if abs(scale) < 1e-12:
        raise ValueError("psin does not span a valid normalized flux interval")

    for i in range(psin.shape[0]):
        psin[i] = (psin[i] - offset) / scale
    psin[0] = 0.0
    psin[-1] = 1.0
    return psin


@njit(cache=True, fastmath=True, nogil=True)
def _regularize_axis_linear(profile: np.ndarray, rho: np.ndarray, n_fix: int) -> np.ndarray:
    if n_fix <= 0:
        return profile

    # Axis-near derivatives are ill-conditioned in rho coordinates.  Fit the
    # smooth ratio profile/rho against rho**2 outside the affected region and
    # extrapolate inward.
    anchor0 = n_fix
    anchor1 = n_fix + 1
    rho0 = rho[anchor0]
    rho1 = rho[anchor1]
    x0 = rho0 * rho0
    x1 = rho1 * rho1

    slope0 = profile[anchor0] / rho0
    slope1 = profile[anchor1] / rho1
    slope_gradient = (slope1 - slope0) / (x1 - x0)
    for i in range(n_fix):
        x = rho[i] * rho[i]
        profile[i] = rho[i] * (slope0 + slope_gradient * (x - x0))

    return profile


@njit(cache=True, fastmath=True, nogil=True)
def _regularize_psin_r(psin_r: np.ndarray, rho: np.ndarray, n_fix: int) -> np.ndarray:
    """Repair and floor ``psin_r`` before downstream divisions.

    ``n_fix`` is the number of head samples whose ``rho`` lies inside the
    axis-affected region.  It is pre-computed during operator setup from the
    grid ``rho`` array and the ``fix_rho`` threshold.

    The first two samples outside the affected region (indices ``n_fix`` and
    ``n_fix + 1``) serve as clean anchors.  Extrapolate the smooth even ratio
    ``psin_r / rho`` as a linear function of ``rho^2`` back to all head samples,
    then enforce the single engine-level positive floor used by psin-space
    divisions.
    """
    _regularize_axis_linear(psin_r, rho, n_fix)
    for i in range(psin_r.shape[0]):
        if psin_r[i] < 1.0e-10:
            psin_r[i] = 1.0e-10
    return psin_r


@njit(cache=True, fastmath=True, nogil=True)
def _floor_signed_current_primitive(profile: np.ndarray) -> np.ndarray:
    """Apply a tiny same-sign floor to cumulative current primitives."""
    edge = profile[-1]
    floor_value = max(abs(edge), 1.0) * 1.0e-12
    if edge < 0.0:
        for i in range(profile.shape[0]):
            if profile[i] > -floor_value:
                profile[i] = -floor_value
    else:
        for i in range(profile.shape[0]):
            if profile[i] < floor_value:
                profile[i] = floor_value
    return profile


@njit(cache=True, fastmath=True, nogil=True)
def _regularize_axis_even(profile: np.ndarray, rho: np.ndarray, n_fix: int) -> np.ndarray:
    if n_fix <= 0:
        return profile

    # Even profiles have zero first derivative at the magnetic axis.  Linear
    # extrapolation in rho**2 preserves that parity better than in rho.
    anchor0 = n_fix
    anchor1 = n_fix + 1
    x0 = rho[anchor0] * rho[anchor0]
    x1 = rho[anchor1] * rho[anchor1]
    value0 = profile[anchor0]
    value1 = profile[anchor1]
    value_gradient = (value1 - value0) / (x1 - x0)
    for i in range(n_fix):
        x = rho[i] * rho[i]
        profile[i] = value0 + value_gradient * (x - x0)

    return profile


@njit(cache=True, fastmath=True, nogil=True)
def _regularize_ffn_psin(FFn_psin: np.ndarray, rho: np.ndarray, n_fix: int) -> np.ndarray:
    return _regularize_axis_even(FFn_psin, rho, n_fix)


@njit(cache=True, fastmath=True, nogil=True)
def _enforce_axis_even_profile(profile: np.ndarray, rho: np.ndarray) -> np.ndarray:
    if profile.shape[0] < 3:
        return profile
    x1 = rho[1] * rho[1]
    x2 = rho[2] * rho[2]
    if abs(x2 - x1) < 1e-14:
        return profile
    slope = (profile[2] - profile[1]) / (x2 - x1)
    intercept = profile[1] - slope * x1
    profile[0] = intercept + slope * rho[0] * rho[0]
    profile[1] = intercept + slope * x1
    return profile


@njit(cache=True, nogil=True)
def _compute_Pn_out(
    out_Pn: np.ndarray,
    Pn_r: np.ndarray,
    accumulator: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    full_integration(out_Pn, Pn_r, accumulator)
    out_Pn -= dot(Pn_r, weights)
    return out_Pn


@njit(cache=True, fastmath=True, nogil=True)
def _fill_pf_rho_integrand(
    out: np.ndarray,
    Kn: np.ndarray,
    current_input: np.ndarray,
    Ln_r: np.ndarray,
    V_r: np.ndarray,
    heat_input: np.ndarray,
) -> np.ndarray:
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    for i in range(out.shape[0]):
        out[i] = Kn[i] * (current_input[i] * Ln_r[i] + V_r[i] * heat_input[i] * pressure_factor)
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_pf_psin_integrand(
    out: np.ndarray,
    current_input: np.ndarray,
    Ln_r: np.ndarray,
    V_r: np.ndarray,
    heat_input: np.ndarray,
) -> np.ndarray:
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    for i in range(out.shape[0]):
        out[i] = current_input[i] * Ln_r[i] + V_r[i] * heat_input[i] * pressure_factor
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _weighted_profile_sign(values: np.ndarray, weights: np.ndarray) -> float:
    weighted = dot(values, weights)
    if weighted < 0.0:
        return -1.0
    return 1.0


@njit(cache=True, fastmath=True, nogil=True)
def _signed_sqrt_ratio(numerator: float, denominator: float) -> float:
    ratio = numerator / denominator
    if ratio < 0.0:
        return -np.sqrt(-ratio)
    return np.sqrt(ratio)


@njit(cache=True, fastmath=True, nogil=True)
def _fill_g1n_psin_integrand(
    out: np.ndarray,
    JdivR: np.ndarray,
    FFn_psin: np.ndarray,
    R: np.ndarray,
    Pn_psin: np.ndarray,
) -> np.ndarray:
    nr, nt = out.shape
    for i in range(nr):
        ffn_i = FFn_psin[i]
        pn_i = Pn_psin[i]
        for j in range(nt):
            out[i, j] = JdivR[i, j] * (ffn_i + R[i, j] * R[i, j] * pn_i)
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_g1n_rho_integrand(
    out: np.ndarray,
    JdivR: np.ndarray,
    FFn_r: np.ndarray,
    R: np.ndarray,
    Pn_r: np.ndarray,
    psin_r: np.ndarray,
    source_scale: float,
) -> np.ndarray:
    nr, nt = out.shape
    for i in range(nr):
        ffn_i = source_scale * FFn_r[i]
        pn_i = source_scale * Pn_r[i]
        psin_r_i = psin_r[i]
        for j in range(nt):
            out[i, j] = JdivR[i, j] * (ffn_i + R[i, j] * R[i, j] * pn_i) / psin_r_i
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_pp_ffn_psin(
    out: np.ndarray,
    psin_r: np.ndarray,
    Kn_r: np.ndarray,
    Kn: np.ndarray,
    psin_rr: np.ndarray,
    V_r: np.ndarray,
    Pn_psin: np.ndarray,
    Ln_r: np.ndarray,
    alpha_ratio: float,
) -> np.ndarray:
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    for i in range(out.shape[0]):
        term0 = alpha_ratio * (Kn_r[i] * psin_r[i] + Kn[i] * psin_rr[i])
        term1 = V_r[i] * Pn_psin[i] * pressure_factor
        ffn_r = -(term0 + term1) * (psin_r[i] / Ln_r[i])
        out[i] = ffn_r / psin_r[i]
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_pi_ffn_psin(
    out: np.ndarray,
    Itor_r: np.ndarray,
    V_r: np.ndarray,
    Pn_psin: np.ndarray,
    Ln_r: np.ndarray,
    current_scale: float,
) -> np.ndarray:
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    for i in range(out.shape[0]):
        term0 = current_scale * Itor_r[i]
        term1 = V_r[i] * Pn_psin[i] * pressure_factor
        out[i] = -(term0 + term1) / Ln_r[i]
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _fill_pj_ffn_psin(
    out: np.ndarray,
    jtor: np.ndarray,
    S_r: np.ndarray,
    V_r: np.ndarray,
    Pn_psin: np.ndarray,
    psin_r: np.ndarray,
    Ln_r: np.ndarray,
    current_scale: float,
) -> np.ndarray:
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    for i in range(out.shape[0]):
        term0 = current_scale * jtor[i] * S_r[i]
        term1 = V_r[i] * Pn_psin[i] * pressure_factor
        ffn_r = -(term0 + term1) * (psin_r[i] / Ln_r[i])
        out[i] = ffn_r / psin_r[i]
    return out


@njit(cache=True, nogil=True)
def _dense_solve_one_rhs_inplace(A: np.ndarray, b: np.ndarray, n: int, pivot_tol: float) -> None:
    """Solve ``A x = b`` in-place using dense Gaussian elimination with partial pivoting.

    ``A`` is overwritten by its LU factors and ``b`` is overwritten by the solution.  Only
    the leading ``n x n`` block of ``A`` and the first ``n`` entries of ``b`` are used.
    """
    scale = 0.0
    for i in range(n):
        for j in range(n):
            value = abs(A[i, j])
            if value > scale:
                scale = value
    threshold = pivot_tol
    if scale > 1.0:
        threshold = pivot_tol * scale

    for k in range(n - 1):
        pivot = k
        pivot_abs = abs(A[k, k])
        for i in range(k + 1, n):
            value = abs(A[i, k])
            if value > pivot_abs:
                pivot = i
                pivot_abs = value
        if pivot_abs <= threshold or not np.isfinite(pivot_abs):
            raise ValueError("PQ dense solve failed: singular pivot")

        if pivot != k:
            for j in range(n):
                tmp = A[k, j]
                A[k, j] = A[pivot, j]
                A[pivot, j] = tmp
            tmp_b = b[k]
            b[k] = b[pivot]
            b[pivot] = tmp_b

        akk = A[k, k]
        for i in range(k + 1, n):
            factor = A[i, k] / akk
            A[i, k] = factor
            for j in range(k + 1, n):
                A[i, j] -= factor * A[k, j]
            b[i] -= factor * b[k]

    last_pivot = abs(A[n - 1, n - 1])
    if last_pivot <= threshold or not np.isfinite(last_pivot):
        raise ValueError("PQ dense solve failed: singular last pivot")

    for ii in range(n):
        i = n - 1 - ii
        accum = b[i]
        for j in range(i + 1, n):
            accum -= A[i, j] * b[j]
        b[i] = accum / A[i, i]
        if not np.isfinite(b[i]):
            raise ValueError("PQ dense solve produced non-finite solution")


@njit(cache=True, nogil=True)
def _dense_solve_two_rhs_inplace(
    A: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    n: int,
    pivot_tol: float,
) -> None:
    """Solve two RHS vectors with one dense partial-pivot elimination."""
    scale = 0.0
    for i in range(n):
        for j in range(n):
            value = abs(A[i, j])
            if value > scale:
                scale = value
    threshold = pivot_tol
    if scale > 1.0:
        threshold = pivot_tol * scale

    for k in range(n - 1):
        pivot = k
        pivot_abs = abs(A[k, k])
        for i in range(k + 1, n):
            value = abs(A[i, k])
            if value > pivot_abs:
                pivot = i
                pivot_abs = value
        if pivot_abs <= threshold or not np.isfinite(pivot_abs):
            raise ValueError("PQ dense solve failed: singular pivot")

        if pivot != k:
            for j in range(n):
                tmp = A[k, j]
                A[k, j] = A[pivot, j]
                A[pivot, j] = tmp
            tmp_b0 = b0[k]
            b0[k] = b0[pivot]
            b0[pivot] = tmp_b0
            tmp_b1 = b1[k]
            b1[k] = b1[pivot]
            b1[pivot] = tmp_b1

        akk = A[k, k]
        for i in range(k + 1, n):
            factor = A[i, k] / akk
            A[i, k] = factor
            for j in range(k + 1, n):
                A[i, j] -= factor * A[k, j]
            b0[i] -= factor * b0[k]
            b1[i] -= factor * b1[k]

    last_pivot = abs(A[n - 1, n - 1])
    if last_pivot <= threshold or not np.isfinite(last_pivot):
        raise ValueError("PQ dense solve failed: singular last pivot")

    for ii in range(n):
        i = n - 1 - ii
        accum0 = b0[i]
        accum1 = b1[i]
        for j in range(i + 1, n):
            accum0 -= A[i, j] * b0[j]
            accum1 -= A[i, j] * b1[j]
        b0[i] = accum0 / A[i, i]
        b1[i] = accum1 / A[i, i]
        if not np.isfinite(b0[i]) or not np.isfinite(b1[i]):
            raise ValueError("PQ dense solve produced non-finite solution")


@njit(cache=True, nogil=True)
def _fill_pq_linear_matrix(
    A: np.ndarray,
    rhs: np.ndarray,
    D: np.ndarray,
    coeff_d: np.ndarray,
    coeff_y: np.ndarray,
    forcing: np.ndarray,
    edge_value: float,
    n: int,
) -> None:
    """Assemble the dense first-order PQ collocation system and impose edge value."""
    for i in range(n):
        for j in range(n):
            A[i, j] = coeff_d[i] * D[i, j]
        A[i, i] += coeff_y[i]
        rhs[i] = forcing[i]

    edge = n - 1
    for j in range(n):
        A[edge, j] = 0.0
    A[edge, edge] = 1.0
    rhs[edge] = edge_value


@njit(cache=True, nogil=True)
def _fill_pq_linear_matrix_two_rhs(
    A: np.ndarray,
    rhs0: np.ndarray,
    rhs1: np.ndarray,
    D: np.ndarray,
    coeff_d: np.ndarray,
    coeff_y: np.ndarray,
    forcing0: np.ndarray,
    forcing1: np.ndarray,
    edge_value0: float,
    edge_value1: float,
    n: int,
) -> None:
    """Assemble one dense PQ system with two edge-conditioned RHS vectors."""
    for i in range(n):
        for j in range(n):
            A[i, j] = coeff_d[i] * D[i, j]
        A[i, i] += coeff_y[i]
        rhs0[i] = forcing0[i]
        rhs1[i] = forcing1[i]

    edge = n - 1
    for j in range(n):
        A[edge, j] = 0.0
    A[edge, edge] = 1.0
    rhs0[edge] = edge_value0
    rhs1[edge] = edge_value1


@njit(cache=True, nogil=True)
def _validate_pq_source_scalar(value: float, label_code: int) -> None:
    if not np.isfinite(value):
        raise ValueError("PQ strict solve produced non-finite scalar")
    if label_code == 0 and abs(value) <= 1.0e-14:
        raise ValueError("PQ strict solve produced near-zero alpha2")
    if label_code == 1 and abs(value) <= 1.0e-14:
        raise ValueError("PQ strict solve produced near-zero alpha1")


@njit(cache=True, nogil=True)
def _normalize_pq_signed_psi_r(
    psi_r: np.ndarray,
    weights: np.ndarray,
    rho: np.ndarray,
    n_axis_fix: int,
) -> float:
    alpha2 = dot(psi_r, weights)
    _validate_pq_source_scalar(alpha2, 0)
    scale_into(psi_r, psi_r, 1.0 / alpha2)
    for i in range(psi_r.shape[0]):
        if not np.isfinite(psi_r[i]) or psi_r[i] <= 0.0:
            raise ValueError("PQ strict solve produced invalid normalized psin_r")
    _regularize_psin_r(psi_r, rho, n_axis_fix)
    for i in range(psi_r.shape[0]):
        if not np.isfinite(psi_r[i]) or psi_r[i] <= 0.0:
            raise ValueError("PQ strict solve produced invalid normalized psin_r")
    return alpha2


@njit(cache=True, nogil=True)
def _fill_pq_q_profile(
    out_q: np.ndarray,
    current_input: np.ndarray,
    Kn: np.ndarray,
    Ln_r: np.ndarray,
    edge_F: float,
    Ip: float,
) -> None:
    has_Ip = not np.isnan(Ip)
    if has_Ip:
        if abs(Ip) <= 1.0e-14:
            raise ValueError("PQ strict solve received near-zero Ip")
        if abs(current_input[-1]) <= 1.0e-14:
            raise ValueError("PQ strict solve received near-zero edge q input")
        q_scale = (2.0 * np.pi * edge_F) / Ip
        q_scale *= Kn[-1] * Ln_r[-1] / current_input[-1]
        for i in range(out_q.shape[0]):
            out_q[i] = current_input[i] * q_scale
    else:
        for i in range(out_q.shape[0]):
            out_q[i] = current_input[i]

    for i in range(out_q.shape[0]):
        if not np.isfinite(out_q[i]) or abs(out_q[i]) <= 1.0e-14:
            raise ValueError("PQ strict solve received invalid q profile")


@njit(cache=True, nogil=True)
def _fill_pq_W_and_derivative(
    W: np.ndarray,
    W_r: np.ndarray,
    Kn: np.ndarray,
    Ln_r: np.ndarray,
    q_prof: np.ndarray,
    differentiator: np.ndarray,
) -> None:
    for i in range(W.shape[0]):
        if not np.isfinite(Ln_r[i]) or abs(Ln_r[i]) <= 1.0e-14:
            raise ValueError("PQ strict solve received invalid Ln_r")
        W[i] = Kn[i] * Ln_r[i] / q_prof[i]
        if not np.isfinite(W[i]):
            raise ValueError("PQ strict solve produced invalid W")
    full_differentiation(W_r, W, differentiator)


@njit(cache=True, nogil=True)
def _pq_psin_beta_residual(
    alpha1: float,
    F0: np.ndarray,
    F1: np.ndarray,
    q_prof: np.ndarray,
    Ln_r: np.ndarray,
    heat_input: np.ndarray,
    V_r: np.ndarray,
    weights: np.ndarray,
    accumulator: np.ndarray,
    trial_psin_r: np.ndarray,
    trial_Pn_r: np.ndarray,
    trial_Pn: np.ndarray,
    beta_target: float,
) -> float:
    n = F0.shape[0]
    alpha2 = 0.0
    for i in range(n):
        F_value = F0[i] + alpha1 * F1[i]
        psi_r = F_value * Ln_r[i] / q_prof[i]
        if not np.isfinite(psi_r):
            return np.nan
        trial_psin_r[i] = psi_r
        alpha2 += psi_r * weights[i]
    if not np.isfinite(alpha2) or abs(alpha2) <= 1.0e-14:
        return np.nan
    for i in range(n):
        trial_psin_r[i] /= alpha2
        if not np.isfinite(trial_psin_r[i]) or trial_psin_r[i] <= 0.0:
            return np.nan
        trial_Pn_r[i] = heat_input[i] * trial_psin_r[i]
    _compute_Pn_out(trial_Pn, trial_Pn_r, accumulator, weights)
    beta_den = weighted_dot(trial_Pn, V_r, weights)
    if not np.isfinite(beta_den):
        return np.nan
    return alpha1 * alpha2 * beta_den - beta_target


@njit(cache=True, nogil=True)
def _solve_pq_psin_beta_alpha1(
    F0: np.ndarray,
    F1: np.ndarray,
    q_prof: np.ndarray,
    Ln_r: np.ndarray,
    heat_input: np.ndarray,
    V_r: np.ndarray,
    weights: np.ndarray,
    accumulator: np.ndarray,
    trial_psin_r: np.ndarray,
    trial_Pn_r: np.ndarray,
    trial_Pn: np.ndarray,
    beta_target: float,
) -> float:
    base = 0.0
    r_base = _pq_psin_beta_residual(
        base,
        F0,
        F1,
        q_prof,
        Ln_r,
        heat_input,
        V_r,
        weights,
        accumulator,
        trial_psin_r,
        trial_Pn_r,
        trial_Pn,
        beta_target,
    )
    if not np.isfinite(r_base):
        raise ValueError("PQ/psin strict beta solve failed at lower bracket")

    for direction in (1.0, -1.0):
        upper = direction
        r_upper = _pq_psin_beta_residual(
            upper,
            F0,
            F1,
            q_prof,
            Ln_r,
            heat_input,
            V_r,
            weights,
            accumulator,
            trial_psin_r,
            trial_Pn_r,
            trial_Pn,
            beta_target,
        )
        for _ in range(80):
            # The beta constraint is scalar and monotone on each valid q/F
            # branch.  Try both alpha1 signs because alpha2 carries current
            # direction, so physical equilibria can need a negative alpha1.
            if np.isfinite(r_upper) and r_base * r_upper <= 0.0:
                lower = base
                r_lower = r_base
                for _ in range(80):
                    mid = 0.5 * (lower + upper)
                    r_mid = _pq_psin_beta_residual(
                        mid,
                        F0,
                        F1,
                        q_prof,
                        Ln_r,
                        heat_input,
                        V_r,
                        weights,
                        accumulator,
                        trial_psin_r,
                        trial_Pn_r,
                        trial_Pn,
                        beta_target,
                    )
                    if not np.isfinite(r_mid):
                        upper = mid
                        continue
                    if abs(r_mid) <= 1.0e-12 * (1.0 + abs(beta_target)):
                        return mid
                    if r_lower * r_mid <= 0.0:
                        upper = mid
                        r_upper = r_mid
                    else:
                        lower = mid
                        r_lower = r_mid
                return 0.5 * (lower + upper)
            upper *= 2.0
            r_upper = _pq_psin_beta_residual(
                upper,
                F0,
                F1,
                q_prof,
                Ln_r,
                heat_input,
                V_r,
                weights,
                accumulator,
                trial_psin_r,
                trial_Pn_r,
                trial_Pn,
                beta_target,
            )
    raise ValueError("PQ/psin strict beta solve failed to bracket alpha1")


def build_source_remap_cache(
    coordinate: str,
    source_sample_count: int,
    *,
    rho: np.ndarray | None = None,
    stencil_size: int = DEFAULT_LOCAL_BARYCENTRIC_STENCIL,
    interpolation_kind: str | None = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Build reusable interpolation cache data for sampled source inputs.

    Rho-coordinate sources are tied to the fixed operator grid, so their remap
    matrix can be built once.  Psin-coordinate sources depend on the current
    solution and only keep interpolation weights here; their query is refreshed
    at each source evaluation.
    """
    coord = str(coordinate).lower()
    if coord not in ("rho", "psin"):
        raise ValueError(f"Unsupported coordinate {coordinate!r}")

    count = int(source_sample_count)
    if count < 1:
        raise ValueError(f"source_sample_count must be positive, got {source_sample_count!r}")

    coord_code = PSIN_COORDINATE if coord == "psin" else RHO_COORDINATE
    local_size = min(count, int(stencil_size))
    if local_size < 1:
        raise ValueError(f"stencil_size must be positive, got {stencil_size!r}")
    weights = uniform_barycentric_weights(local_size)
    fixed_remap_matrix = np.empty((0, 0), dtype=np.float64)
    if coord_code == RHO_COORDINATE:
        if rho is None:
            raise ValueError("rho is required when coordinate='rho'")
        query = np.clip(np.asarray(rho, dtype=np.float64), 0.0, 1.0)
        fixed_remap_matrix = build_uniform_source_interpolation_matrix(
            query, count, kind=interpolation_kind
        )

    return local_size, weights, fixed_remap_matrix


def resolve_source_inputs(
    out_heat_input: np.ndarray,
    out_current_input: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    source_sample_count: int,
    barycentric_weights: np.ndarray,
    fixed_remap_matrix: np.ndarray,
    heat_spline_coeff: np.ndarray,
    current_spline_coeff: np.ndarray,
    psin_query: np.ndarray,
    use_barycentric: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve sampled heat/current source inputs onto operator rho nodes."""

    heat = np.asarray(heat_input, dtype=np.float64)
    current = np.asarray(current_input, dtype=np.float64)
    return _resolve_source_inputs_prepared(
        out_heat_input,
        out_current_input,
        heat,
        current,
        coordinate_code,
        source_sample_count,
        barycentric_weights,
        fixed_remap_matrix,
        heat_spline_coeff,
        current_spline_coeff,
        psin_query,
        use_barycentric,
    )


def _resolve_source_inputs_prepared(
    out_heat_input: np.ndarray,
    out_current_input: np.ndarray,
    heat: np.ndarray,
    current: np.ndarray,
    coordinate_code: int,
    source_sample_count: int,
    barycentric_weights: np.ndarray,
    fixed_remap_matrix: np.ndarray,
    heat_spline_coeff: np.ndarray,
    current_spline_coeff: np.ndarray,
    psin_query: np.ndarray,
    use_barycentric: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve source inputs when all arrays are already normalized ndarrays."""

    if heat.ndim != 1 or current.ndim != 1:
        raise ValueError(f"Expected 1D heat/current inputs, got {heat.shape} and {current.shape}")
    if heat.shape != current.shape:
        raise ValueError(f"heat/current shape mismatch: {heat.shape} vs {current.shape}")
    if heat.shape[0] != source_sample_count:
        raise ValueError(f"Expected {source_sample_count} source samples, got {heat.shape[0]}")
    if (
        out_heat_input.ndim != 1
        or out_current_input.ndim != 1
        or out_heat_input.shape != out_current_input.shape
    ):
        raise ValueError(
            "Expected matching 1D output inputs, "
            f"got {out_heat_input.shape} and {out_current_input.shape}"
        )
    if psin_query.ndim != 1:
        raise ValueError(f"Expected psin_query to be 1D, got {psin_query.shape}")

    if coordinate_code == RHO_COORDINATE:
        # Rho inputs use the precomputed linear map; no solver state participates
        # after the cache is built.
        np.matmul(fixed_remap_matrix, heat, out=out_heat_input)
        np.matmul(fixed_remap_matrix, current, out=out_current_input)
        return out_heat_input, out_current_input

    if psin_query.shape != out_heat_input.shape:
        raise ValueError(f"psin_query shape mismatch: {psin_query.shape} vs {out_heat_input.shape}")

    # Psin inputs are materialized against the current psin field.  Spline is
    # smoother for general sampled inputs; local barycentric keeps high-order
    # route variants allocation-free inside fixed-point loops.
    if use_barycentric:
        _local_barycentric_interpolate_pair(
            out_heat_input,
            out_current_input,
            heat,
            current,
            psin_query,
            barycentric_weights,
        )
    else:
        _uniform_spline_interpolate_pair(
            out_heat_input,
            out_current_input,
            heat_spline_coeff,
            current_spline_coeff,
            psin_query,
        )
    return out_heat_input, out_current_input


# ---------------------------------------------------------------------------
# Zero-allocation scratch variants (Phase 3)
# ---------------------------------------------------------------------------

# Route families share one output contract but choose different primitives:
# PF derives psin_r from pressure/current source balance, PP takes psin_r-like
# current data directly, PI works through toroidal-current primitives, PJ routes
# start from current-density-like data, and PQ treats q as strict input.  The
# repeated rho/psin/uniform/grid functions below differ mainly in how the input
# profiles are interpreted or remapped; keep family-level comments here instead
# of duplicating them in every variant.
#
# Docs-facing route meanings in compact form:
# - PF: heat drives pressure-gradient data; current drives FF' data.
# - PP: current drives normalized flux-gradient/psin_r data.
# - PI/PJ1/PJ2: current drives cumulative/current-density/parallel-current data.
# - PQ: current is safety factor q, so F or F**2 is solved from q and edge F.


@register_source_route(
    ("PF", "rho", "uniform"),
    ("PF", "rho", "grid"),
)
@njit(cache=True, nogil=True)
def _update_pf_from_rho_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, _, Ln_r, _, R, JdivR = _source_geometry_workspace_views(radial_fields, surface_fields)
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand = array_scratch[_SLOT_INTEGRAND]
    _fill_pf_rho_integrand(integrand, Kn, current_input, Ln_r, V_r, heat_input)
    full_integration(out_psin_r, integrand, accumulator)
    out_psin_r *= -2.0
    psi_square_sign = _weighted_profile_sign(out_psin_r, weights)
    if psi_square_sign < 0.0:
        out_psin_r *= -1.0
    for i in range(out_psin_r.shape[0]):
        if out_psin_r[i] < 1.0e-6:
            out_psin_r[i] = 1.0e-6
    out_psin_r[:] = np.sqrt(out_psin_r)
    out_psin_r /= Kn
    _regularize_psin_r(out_psin_r, rho, n_axis_fix)
    prof = out_psin_r
    integral_prof = dot(prof, weights)
    # alpha2 stores the pre-normalization integral; psin_r itself is normalized
    # to integrate to one so downstream geometry uses the canonical psin scale.
    out_psin_r /= integral_prof
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)
    if (not has_Ip) and (not has_beta):
        # With no global Ip/beta target, PF determines both alpha scales from
        # the integrated source profiles.  Rho-coordinate PF inputs are
        # derivatives with respect to rho, so their global sign belongs to the
        # flux-direction gauge carried by alpha2, not to the solved shape.
        alpha2 = psi_square_sign * integral_prof
        alpha1 = -dot(heat_input, weights) / integral_prof
        source_scale = psi_square_sign / (alpha1 * alpha2)
        scaled_ratio_into(out_Pn_psin, heat_input, out_psin_r, source_scale)
        scaled_ratio_into(out_FFn_psin, current_input, out_psin_r, source_scale)
        _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
        return alpha1, alpha2
    c2 = integral_prof * integral_prof
    if has_Ip and (not has_beta):
        g1n_integrand = matrix_scratch[0]
        _fill_g1n_rho_integrand(
            g1n_integrand,
            JdivR,
            current_input,
            R,
            heat_input,
            out_psin_r,
            psi_square_sign,
        )
        radial_scratch = array_scratch[_SLOT_AUX0]
        nt = g1n_integrand.shape[1]
        for j in range(nt):
            s = 0.0
            for i in range(g1n_integrand.shape[0]):
                s += weights[i] * g1n_integrand[i, j]
            radial_scratch[j] = s
        G1n_integral = 0.0
        for j in range(nt):
            G1n_integral += radial_scratch[j]
        G1n_integral = (2.0 * np.pi / nt) * G1n_integral
        alpha1 = -Ip / G1n_integral
    elif has_beta and (not has_Ip):
        scratch_aux = array_scratch[_SLOT_AUX0]
        _compute_Pn_out(scratch_aux, heat_input, accumulator, weights)
        c1 = 0.5 * beta * B0**2 * dot(V_r, weights) / weighted_dot(scratch_aux, V_r, weights)
        alpha1 = _signed_sqrt_ratio(c1, c2)
    else:
        raise ValueError("PF does not support applying Ip and beta constraints simultaneously")
    alpha2 = c2 * alpha1
    scaled_ratio_into(out_Pn_psin, heat_input, out_psin_r, psi_square_sign)
    scaled_ratio_into(out_FFn_psin, current_input, out_psin_r, psi_square_sign)
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PF", "psin", "uniform"))
@njit(cache=True, nogil=True)
def _update_pf_from_psin_uniform_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, _, Ln_r, _, R, JdivR = _source_geometry_workspace_views(radial_fields, surface_fields)
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand = array_scratch[_SLOT_INTEGRAND]
    _fill_pf_psin_integrand(integrand, current_input, Ln_r, V_r, heat_input)
    full_integration(out_psin_r, integrand, accumulator)
    out_psin_r *= -1.0
    out_psin_r /= Kn
    psi_scale_sign = _weighted_profile_sign(out_psin_r, weights)
    if psi_scale_sign < 0.0:
        out_psin_r *= -1.0
    _regularize_psin_r(out_psin_r, rho, n_axis_fix)
    integral_prof = dot(out_psin_r, weights)
    out_psin_r /= integral_prof
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)
    if (not has_Ip) and (not has_beta):
        alpha2 = psi_scale_sign * integral_prof
        pressure_profile = array_scratch[_SLOT_AUX0]
        product_into(pressure_profile, heat_input, out_psin_r)
        alpha1 = -dot(pressure_profile, weights)
        scale_into(out_Pn_psin, heat_input, 1.0 / alpha1)
        scale_into(out_FFn_psin, current_input, 1.0 / alpha1)
        _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
        return alpha1, alpha2
    c2 = integral_prof
    copy_into(out_Pn_psin, heat_input)
    copy_into(out_FFn_psin, current_input)
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    if has_Ip and (not has_beta):
        g1n_integrand = matrix_scratch[0]
        _fill_g1n_psin_integrand(g1n_integrand, JdivR, out_FFn_psin, R, out_Pn_psin)
        radial_scratch = array_scratch[_SLOT_AUX0]
        nt = g1n_integrand.shape[1]
        for j in range(nt):
            s = 0.0
            for i in range(g1n_integrand.shape[0]):
                s += weights[i] * g1n_integrand[i, j]
            radial_scratch[j] = s
        G1n_integral = 0.0
        for j in range(nt):
            G1n_integral += radial_scratch[j]
        G1n_integral = (2.0 * np.pi / nt) * G1n_integral
        alpha1 = -Ip / G1n_integral
    elif has_beta and (not has_Ip):
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX1]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        c1 = 0.5 * beta * B0**2 * dot(V_r, weights) / weighted_dot(scratch_aux, V_r, weights)
        alpha1 = _signed_sqrt_ratio(c1, c2)
    else:
        raise ValueError("PF does not support applying Ip and beta constraints simultaneously")
    alpha2 = c2 * alpha1
    return alpha1, alpha2


@register_source_route(("PF", "psin", "grid"))
@njit(cache=True, nogil=True)
def _update_pf_from_psin_grid_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, _, Ln_r, _, R, JdivR = _source_geometry_workspace_views(radial_fields, surface_fields)
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand = array_scratch[_SLOT_INTEGRAND]
    _fill_pf_psin_integrand(integrand, current_input, Ln_r, V_r, heat_input)
    full_integration(out_psin_r, integrand, accumulator)
    out_psin_r *= -1.0
    out_psin_r /= Kn
    psi_scale_sign = _weighted_profile_sign(out_psin_r, weights)
    if psi_scale_sign < 0.0:
        out_psin_r *= -1.0
    _regularize_psin_r(out_psin_r, rho, n_axis_fix)
    integral_prof = dot(out_psin_r, weights)
    out_psin_r /= integral_prof
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)
    if (not has_Ip) and (not has_beta):
        alpha2 = psi_scale_sign * integral_prof
        pressure_profile = array_scratch[_SLOT_AUX0]
        product_into(pressure_profile, heat_input, out_psin_r)
        alpha1 = -dot(pressure_profile, weights)
        scale_into(out_Pn_psin, heat_input, 1.0 / alpha1)
        scale_into(out_FFn_psin, current_input, 1.0 / alpha1)
        _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
        return alpha1, alpha2
    c2 = integral_prof
    copy_into(out_Pn_psin, heat_input)
    copy_into(out_FFn_psin, current_input)
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    if has_Ip and (not has_beta):
        g1n_integrand = matrix_scratch[0]
        _fill_g1n_psin_integrand(g1n_integrand, JdivR, out_FFn_psin, R, out_Pn_psin)
        radial_scratch = array_scratch[_SLOT_AUX0]
        nt = g1n_integrand.shape[1]
        for j in range(nt):
            s = 0.0
            for i in range(g1n_integrand.shape[0]):
                s += weights[i] * g1n_integrand[i, j]
            radial_scratch[j] = s
        G1n_integral = 0.0
        for j in range(nt):
            G1n_integral += radial_scratch[j]
        G1n_integral = (2.0 * np.pi / nt) * G1n_integral
        alpha1 = -Ip / G1n_integral
    elif has_beta and (not has_Ip):
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX1]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        c1 = 0.5 * beta * B0**2 * dot(V_r, weights) / weighted_dot(scratch_aux, V_r, weights)
        alpha1 = _signed_sqrt_ratio(c1, c2)
    else:
        raise ValueError("PF does not support applying Ip and beta constraints simultaneously")
    alpha2 = c2 * alpha1
    return alpha1, alpha2


@register_source_route(
    ("PP", "rho", "uniform"),
    ("PP", "rho", "grid"),
)
@njit(cache=True, nogil=True)
def _update_pp_from_rho_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    if has_Ip:
        # PP treats current_input as the unnormalized psin_r shape.  Ip pins the
        # absolute scale through the edge value; otherwise alpha2 is the weighted
        # normalization integral.
        copy_into(out_psin_r, current_input)
        alpha2 = Ip / (2.0 * np.pi * Kn[-1] * out_psin_r[-1])
    else:
        alpha2 = dot(current_input, weights)
        scale_into(out_psin_r, current_input, 1.0 / alpha2)
    _regularize_psin_r(out_psin_r, rho, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)
    if has_beta:
        scaled_ratio_into(out_Pn_psin, heat_input, out_psin_r, 1.0)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX0]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            / alpha2
            * dot(V_r, weights)
            / weighted_dot(scratch_aux, V_r, weights)
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        copy_into(scratch_Pr, heat_input)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        scaled_ratio_into(out_Pn_psin, scratch_Pr, out_psin_r, 1.0 / (alpha1 * alpha2))
    _fill_pp_ffn_psin(
        out_FFn_psin,
        out_psin_r,
        Kn_r,
        Kn,
        out_psin_rr,
        V_r,
        out_Pn_psin,
        Ln_r,
        alpha2 / alpha1,
    )
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PP", "psin", "uniform"))
@njit(cache=True, nogil=True)
def _update_pp_from_psin_uniform_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    if has_Ip:
        copy_into(out_psin_r, current_input)
        alpha2 = Ip / (2.0 * np.pi * Kn[-1] * out_psin_r[-1])
    else:
        alpha2 = dot(current_input, weights)
        scale_into(out_psin_r, current_input, 1.0 / alpha2)
    _regularize_psin_r(out_psin_r, rho, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)
    if has_beta:
        copy_into(out_Pn_psin, heat_input)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX0]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            / alpha2
            * dot(V_r, weights)
            / weighted_dot(scratch_aux, V_r, weights)
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        scaled_product_into(scratch_Pr, heat_input, out_psin_r, alpha2)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        scaled_ratio_into(out_Pn_psin, scratch_Pr, out_psin_r, 1.0 / (alpha1 * alpha2))
    _fill_pp_ffn_psin(
        out_FFn_psin,
        out_psin_r,
        Kn_r,
        Kn,
        out_psin_rr,
        V_r,
        out_Pn_psin,
        Ln_r,
        alpha2 / alpha1,
    )
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PP", "psin", "grid"))
@njit(cache=True, nogil=True)
def _update_pp_from_psin_grid_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    if has_Ip:
        copy_into(out_psin_r, current_input)
        alpha2 = Ip / (2.0 * np.pi * Kn[-1] * out_psin_r[-1])
    else:
        alpha2 = dot(current_input, weights)
        scale_into(out_psin_r, current_input, 1.0 / alpha2)
    _regularize_psin_r(out_psin_r, rho, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)
    if has_beta:
        copy_into(out_Pn_psin, heat_input)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX0]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            / alpha2
            * dot(V_r, weights)
            / weighted_dot(scratch_aux, V_r, weights)
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        scaled_product_into(scratch_Pr, heat_input, out_psin_r, alpha2)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        scaled_ratio_into(out_Pn_psin, scratch_Pr, out_psin_r, 1.0 / (alpha1 * alpha2))
    _fill_pp_ffn_psin(
        out_FFn_psin,
        out_psin_r,
        Kn_r,
        Kn,
        out_psin_rr,
        V_r,
        out_Pn_psin,
        Ln_r,
        alpha2 / alpha1,
    )
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    return alpha1, alpha2


@register_source_route(
    ("PI", "rho", "uniform"),
    ("PI", "rho", "grid"),
)
@njit(cache=True, nogil=True)
def _update_pi_from_rho_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    Itor = array_scratch[_SLOT_AUX0]
    if has_Ip:
        # PI source profiles represent cumulative toroidal current.  Rescale the
        # whole primitive when Ip is prescribed, then differentiate only after
        # psin_r has been normalized.
        scale_into(Itor, current_input, Ip / current_input[-1])
    else:
        copy_into(Itor, current_input)
    _floor_signed_current_primitive(Itor)
    itor_over_kn = array_scratch[_SLOT_INTEGRAND]
    scaled_ratio_into(itor_over_kn, Itor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(itor_over_kn, weights)
    scaled_ratio_into(out_psin_r, Itor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    _regularize_psin_r(out_psin_r, rho, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)
    Itor_r = array_scratch[_SLOT_AUX1]
    full_differentiation(Itor_r, Itor, differentiator)
    _regularize_axis_linear(Itor_r, rho, n_axis_fix)
    if has_beta:
        scaled_ratio_into(out_Pn_psin, heat_input, out_psin_r, 1.0)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX2]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            / alpha2
            * dot(V_r, weights)
            / weighted_dot(scratch_aux, V_r, weights)
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        copy_into(scratch_Pr, heat_input)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        scaled_ratio_into(out_Pn_psin, scratch_Pr, out_psin_r, 1.0 / (alpha1 * alpha2))
    _fill_pi_ffn_psin(out_FFn_psin, Itor_r, V_r, out_Pn_psin, Ln_r, 1.0 / (2.0 * np.pi * alpha1))
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PI", "psin", "uniform"))
@njit(cache=True, nogil=True)
def _update_pi_from_psin_uniform_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    Itor = array_scratch[_SLOT_AUX0]
    if has_Ip:
        scale_into(Itor, current_input, Ip / current_input[-1])
    else:
        copy_into(Itor, current_input)
    _floor_signed_current_primitive(Itor)
    itor_over_kn = array_scratch[_SLOT_INTEGRAND]
    scaled_ratio_into(itor_over_kn, Itor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(itor_over_kn, weights)
    scaled_ratio_into(out_psin_r, Itor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    _regularize_psin_r(out_psin_r, rho, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)
    Itor_r = array_scratch[_SLOT_AUX1]
    full_differentiation(Itor_r, Itor, differentiator)
    _regularize_axis_linear(Itor_r, rho, n_axis_fix)
    if has_beta:
        copy_into(out_Pn_psin, heat_input)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX2]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            / alpha2
            * dot(V_r, weights)
            / weighted_dot(scratch_aux, V_r, weights)
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        scaled_product_into(scratch_Pr, heat_input, out_psin_r, alpha2)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        scaled_ratio_into(out_Pn_psin, scratch_Pr, out_psin_r, 1.0 / (alpha1 * alpha2))
    _fill_pi_ffn_psin(out_FFn_psin, Itor_r, V_r, out_Pn_psin, Ln_r, 1.0 / (2.0 * np.pi * alpha1))
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PI", "psin", "grid"))
@njit(cache=True, nogil=True)
def _update_pi_from_psin_grid_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    Itor = array_scratch[_SLOT_AUX0]
    if has_Ip:
        scale_into(Itor, current_input, Ip / current_input[-1])
    else:
        copy_into(Itor, current_input)
    _floor_signed_current_primitive(Itor)
    itor_over_kn = array_scratch[_SLOT_INTEGRAND]
    scaled_ratio_into(itor_over_kn, Itor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(itor_over_kn, weights)
    scaled_ratio_into(out_psin_r, Itor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    _regularize_psin_r(out_psin_r, rho, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)
    Itor_r = array_scratch[_SLOT_AUX1]
    full_differentiation(Itor_r, Itor, differentiator)
    _regularize_axis_linear(Itor_r, rho, n_axis_fix)
    if has_beta:
        copy_into(out_Pn_psin, heat_input)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX2]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            / alpha2
            * dot(V_r, weights)
            / weighted_dot(scratch_aux, V_r, weights)
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        scaled_product_into(scratch_Pr, heat_input, out_psin_r, alpha2)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        scaled_ratio_into(out_Pn_psin, scratch_Pr, out_psin_r, 1.0 / (alpha1 * alpha2))
    _fill_pi_ffn_psin(out_FFn_psin, Itor_r, V_r, out_Pn_psin, Ln_r, 1.0 / (2.0 * np.pi * alpha1))
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    return alpha1, alpha2


@register_source_route(
    ("PJ1", "rho", "uniform"),
    ("PJ1", "rho", "grid"),
)
@njit(cache=True, nogil=True)
def _update_pj1_from_rho_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand_j = array_scratch[_SLOT_INTEGRAND]
    product_into(integrand_j, current_input, S_r)
    full_integration(out_psin_r, integrand_j, accumulator)
    I_tor_prof = array_scratch[_SLOT_AUX0]
    copy_into(I_tor_prof, out_psin_r)
    I_tor = array_scratch[_SLOT_AUX1]
    jtor = array_scratch[_SLOT_AUX2]
    if has_Ip:
        # PJ1 integrates a current-density-like input into I_tor first; the same
        # Ip scale must be applied to the primitive and to the local jtor profile.
        scale_into(I_tor, I_tor_prof, Ip / I_tor_prof[-1])
        scale_into(jtor, current_input, Ip / I_tor_prof[-1])
    else:
        copy_into(I_tor, I_tor_prof)
        copy_into(jtor, current_input)
    _enforce_axis_even_profile(jtor, rho)
    _floor_signed_current_primitive(I_tor)
    itor_over_kn = array_scratch[_SLOT_INTEGRAND]
    scaled_ratio_into(itor_over_kn, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(itor_over_kn, weights)
    scaled_ratio_into(out_psin_r, I_tor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    _regularize_psin_r(out_psin_r, rho, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)
    if has_beta:
        scaled_ratio_into(out_Pn_psin, heat_input, out_psin_r, 1.0)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_INTEGRAND]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            / alpha2
            * dot(V_r, weights)
            / weighted_dot(scratch_aux, V_r, weights)
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        copy_into(scratch_Pr, heat_input)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        scaled_ratio_into(out_Pn_psin, scratch_Pr, out_psin_r, 1.0 / (alpha1 * alpha2))
    _fill_pj_ffn_psin(
        out_FFn_psin,
        jtor,
        S_r,
        V_r,
        out_Pn_psin,
        out_psin_r,
        Ln_r,
        1.0 / (2.0 * np.pi * alpha1),
    )
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PJ1", "psin", "uniform"))
@njit(cache=True, nogil=True)
def _update_pj1_from_psin_uniform_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand_j = array_scratch[_SLOT_INTEGRAND]
    product_into(integrand_j, current_input, S_r)
    full_integration(out_psin_r, integrand_j, accumulator)
    I_tor_prof = array_scratch[_SLOT_AUX0]
    copy_into(I_tor_prof, out_psin_r)
    I_tor = array_scratch[_SLOT_AUX1]
    jtor = array_scratch[_SLOT_AUX2]
    if has_Ip:
        scale_into(I_tor, I_tor_prof, Ip / I_tor_prof[-1])
        scale_into(jtor, current_input, Ip / I_tor_prof[-1])
    else:
        copy_into(I_tor, I_tor_prof)
        copy_into(jtor, current_input)
    _enforce_axis_even_profile(jtor, rho)
    _floor_signed_current_primitive(I_tor)
    itor_over_kn = array_scratch[_SLOT_INTEGRAND]
    scaled_ratio_into(itor_over_kn, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(itor_over_kn, weights)
    scaled_ratio_into(out_psin_r, I_tor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    _regularize_psin_r(out_psin_r, rho, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)
    if has_beta:
        copy_into(out_Pn_psin, heat_input)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_INTEGRAND]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            / alpha2
            * dot(V_r, weights)
            / weighted_dot(scratch_aux, V_r, weights)
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        scaled_product_into(scratch_Pr, heat_input, out_psin_r, alpha2)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        scaled_ratio_into(out_Pn_psin, scratch_Pr, out_psin_r, 1.0 / (alpha1 * alpha2))
    _fill_pj_ffn_psin(
        out_FFn_psin,
        jtor,
        S_r,
        V_r,
        out_Pn_psin,
        out_psin_r,
        Ln_r,
        1.0 / (2.0 * np.pi * alpha1),
    )
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PJ1", "psin", "grid"))
@njit(cache=True, nogil=True)
def _update_pj1_from_psin_grid_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand_j = array_scratch[_SLOT_INTEGRAND]
    product_into(integrand_j, current_input, S_r)
    full_integration(out_psin_r, integrand_j, accumulator)
    I_tor_prof = array_scratch[_SLOT_AUX0]
    copy_into(I_tor_prof, out_psin_r)
    I_tor = array_scratch[_SLOT_AUX1]
    jtor = array_scratch[_SLOT_AUX2]
    if has_Ip:
        scale_into(I_tor, I_tor_prof, Ip / I_tor_prof[-1])
        scale_into(jtor, current_input, Ip / I_tor_prof[-1])
    else:
        copy_into(I_tor, I_tor_prof)
        copy_into(jtor, current_input)
    _enforce_axis_even_profile(jtor, rho)
    _floor_signed_current_primitive(I_tor)
    itor_over_kn = array_scratch[_SLOT_INTEGRAND]
    scaled_ratio_into(itor_over_kn, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(itor_over_kn, weights)
    scaled_ratio_into(out_psin_r, I_tor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    _regularize_psin_r(out_psin_r, rho, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)
    if has_beta:
        copy_into(out_Pn_psin, heat_input)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_INTEGRAND]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            / alpha2
            * dot(V_r, weights)
            / weighted_dot(scratch_aux, V_r, weights)
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        scaled_product_into(scratch_Pr, heat_input, out_psin_r, alpha2)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        scaled_ratio_into(out_Pn_psin, scratch_Pr, out_psin_r, 1.0 / (alpha1 * alpha2))
    _fill_pj_ffn_psin(
        out_FFn_psin,
        jtor,
        S_r,
        V_r,
        out_Pn_psin,
        out_psin_r,
        Ln_r,
        1.0 / (2.0 * np.pi * alpha1),
    )
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PJ2", "psin", "uniform"))
@njit(cache=True, nogil=True)
def _update_pj2_from_psin_uniform_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, _, Ln_r, _, _, _ = _source_geometry_workspace_views(radial_fields, surface_fields)
    F = F_fields[0]
    F_r = F_fields[1]
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand = array_scratch[_SLOT_INTEGRAND]
    integral_val = array_scratch[_SLOT_AUX0]
    I_tor = array_scratch[_SLOT_AUX1]
    scratch_Pn_r = array_scratch[_SLOT_PNr]
    scratch_aux = array_scratch[_SLOT_AUX2]

    scaled_product_ratio_into(integrand, Ln_r, current_input, F, 1.0)
    full_integration(out_psin_r, integrand, accumulator)
    copy_into(integral_val, out_psin_r)

    if has_Ip:
        # PJ2 couples the source current to the current F profile.  In psin
        # routes the edge normalization uses the physical edge F=R0*B0.
        scaled_product_into(I_tor, F, integral_val, Ip / (R0 * B0 * integral_val[-1]))
    else:
        scaled_product_into(I_tor, F, integral_val, 2.0 * np.pi)
    scaled_ratio_into(integrand, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(integrand, weights)
    scale_into(out_psin_r, integrand, 1.0 / alpha2)
    _regularize_psin_r(out_psin_r, rho, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)

    if has_beta:
        product_into(scratch_Pn_r, heat_input, out_psin_r)
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            / alpha2
            * dot(V_r, weights)
            / weighted_dot(
                scratch_aux,
                V_r,
                weights,
            )
        )
        copy_into(out_Pn_psin, heat_input)
    else:
        alpha1 = -weighted_dot(heat_input, out_psin_r, weights)
        scaled_product_ratio_into(out_Pn_psin, heat_input, out_psin_r, out_psin_r, 1.0 / alpha1)

    product_into(out_FFn_psin, F, F_r)
    scaled_ratio_into(out_FFn_psin, out_FFn_psin, out_psin_r, 1.0 / (alpha1 * alpha2))
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PJ2", "psin", "grid"))
@njit(cache=True, nogil=True)
def _update_pj2_from_psin_grid_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, _, Ln_r, _, _, _ = _source_geometry_workspace_views(radial_fields, surface_fields)
    F = F_fields[0]
    F_r = F_fields[1]
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand = array_scratch[_SLOT_INTEGRAND]
    integral_val = array_scratch[_SLOT_AUX0]
    I_tor = array_scratch[_SLOT_AUX1]
    scratch_Pn_r = array_scratch[_SLOT_PNr]
    scratch_aux = array_scratch[_SLOT_AUX2]

    scaled_product_ratio_into(integrand, Ln_r, current_input, F, 1.0)
    full_integration(out_psin_r, integrand, accumulator)
    copy_into(integral_val, out_psin_r)

    if has_Ip:
        scaled_product_into(I_tor, F, integral_val, Ip / (R0 * B0 * integral_val[-1]))
    else:
        scaled_product_into(I_tor, F, integral_val, 2.0 * np.pi)
    scaled_ratio_into(integrand, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(integrand, weights)
    scale_into(out_psin_r, integrand, 1.0 / alpha2)
    _regularize_psin_r(out_psin_r, rho, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)

    if has_beta:
        product_into(scratch_Pn_r, heat_input, out_psin_r)
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            / alpha2
            * dot(V_r, weights)
            / weighted_dot(
                scratch_aux,
                V_r,
                weights,
            )
        )
        copy_into(out_Pn_psin, heat_input)
    else:
        alpha1 = -weighted_dot(heat_input, out_psin_r, weights)
        scaled_product_ratio_into(out_Pn_psin, heat_input, out_psin_r, out_psin_r, 1.0 / alpha1)

    product_into(out_FFn_psin, F, F_r)
    scaled_ratio_into(out_FFn_psin, out_FFn_psin, out_psin_r, 1.0 / (alpha1 * alpha2))
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    return alpha1, alpha2


@register_source_route(
    ("PJ2", "rho", "uniform"),
    ("PJ2", "rho", "grid"),
)
@njit(cache=True, nogil=True)
def _update_pj2_from_rho_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    F = F_fields[0]
    F_r = F_fields[1]
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand = array_scratch[_SLOT_INTEGRAND]
    scaled_product_ratio_into(integrand, Ln_r, current_input, F, 1.0)
    full_integration(out_psin_r, integrand, accumulator)
    integral_val = array_scratch[_SLOT_AUX0]
    copy_into(integral_val, out_psin_r)
    I_tor = array_scratch[_SLOT_AUX1]
    if has_Ip:
        scaled_product_into(I_tor, F, integral_val, Ip / (F[-1] * integral_val[-1]))
    else:
        scaled_product_into(I_tor, F, integral_val, 2.0 * np.pi)
    itor_over_kn = array_scratch[_SLOT_INTEGRAND]
    scaled_ratio_into(itor_over_kn, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(itor_over_kn, weights)
    scaled_ratio_into(out_psin_r, I_tor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    _regularize_psin_r(out_psin_r, rho, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)
    if has_beta:
        scaled_ratio_into(out_Pn_psin, heat_input, out_psin_r, 1.0)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX2]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            / alpha2
            * dot(V_r, weights)
            / weighted_dot(scratch_aux, V_r, weights)
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        copy_into(scratch_Pr, heat_input)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        scaled_ratio_into(out_Pn_psin, scratch_Pr, out_psin_r, 1.0 / (alpha1 * alpha2))
    scaled_product_into(out_FFn_psin, F, F_r, 1.0 / (alpha1 * alpha2))
    scaled_ratio_into(out_FFn_psin, out_FFn_psin, out_psin_r, 1.0)
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PQ", "psin", "uniform"))
@njit(cache=True, nogil=True)
def _update_pq_from_psin_uniform_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, _, Ln_r, _, _, _ = _source_geometry_workspace_views(radial_fields, surface_fields)
    n = rho.shape[0]
    edge_F = R0 * B0
    if not np.isfinite(edge_F) or abs(edge_F) <= 1.0e-14:
        raise ValueError("PQ/psin strict solve received invalid edge F")

    W = array_scratch[_SLOT_INTEGRAND]
    q_prof = array_scratch[_SLOT_AUX0]
    coeff_d = array_scratch[_SLOT_AUX1]
    coeff_y = array_scratch[_SLOT_AUX2]
    rhs = array_scratch[_SLOT_PNr]
    F_solved = array_scratch[_SLOT_Pr]
    F_r = array_scratch[_SLOT_Fr]
    A = array_scratch[_SLOT_PQ_MATRIX : _SLOT_PQ_MATRIX + n, :]

    _fill_pq_q_profile(q_prof, current_input, Kn, Ln_r, edge_F, Ip)
    _fill_pq_W_and_derivative(W, F_r, Kn, Ln_r, q_prof, differentiator)

    # PQ/psin treats q as strict input.  The unknown F profile solves a dense
    # first-order collocation system, then psin_r follows from q = F*Ln_r/psin_r.
    # This is intentionally more constrained than the PF/PP/PI/PJ routes: an
    # invalid q profile should fail early instead of being silently regularized.
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    for i in range(n):
        coeff_d[i] = W[i] + q_prof[i]
        coeff_y[i] = F_r[i]
        if not np.isfinite(coeff_d[i]) or not np.isfinite(coeff_y[i]):
            raise ValueError("PQ/psin strict solve assembled non-finite matrix")

    has_beta = not np.isnan(beta)
    if has_beta:
        # Solve A F0 = b_edge and A F1 = b_pressure, then determine alpha1 from
        # the scalar beta constraint with F = F0 + alpha1 * F1.
        for i in range(n):
            F_solved[i] = 0.0
            W[i] = -pressure_factor * V_r[i] * heat_input[i]
            if not np.isfinite(W[i]):
                raise ValueError("PQ/psin strict beta solve assembled non-finite pressure RHS")
        _fill_pq_linear_matrix_two_rhs(
            A,
            F_solved,
            W,
            differentiator,
            coeff_d,
            coeff_y,
            F_solved,
            W,
            edge_F,
            0.0,
            n,
        )
        _dense_solve_two_rhs_inplace(A, F_solved, W, n, 1.0e-12)

        beta_target = 0.5 * beta * B0**2 * dot(V_r, weights)
        alpha1 = _solve_pq_psin_beta_alpha1(
            F_solved,
            W,
            q_prof,
            Ln_r,
            heat_input,
            V_r,
            weights,
            accumulator,
            out_psin_r,
            coeff_d,
            coeff_y,
            beta_target,
        )
        for i in range(n):
            F_solved[i] = F_solved[i] + alpha1 * W[i]
        copy_into(out_Pn_psin, heat_input)
    else:
        for i in range(n):
            rhs[i] = -pressure_factor * V_r[i] * heat_input[i]
            if not np.isfinite(rhs[i]):
                raise ValueError("PQ/psin strict solve assembled non-finite pressure RHS")
        _fill_pq_linear_matrix(A, rhs, differentiator, coeff_d, coeff_y, rhs, edge_F, n)
        copy_into(F_solved, rhs)
        _dense_solve_one_rhs_inplace(A, F_solved, n, 1.0e-12)
        alpha1 = 0.0

    for i in range(n):
        out_psin_r[i] = F_solved[i] * Ln_r[i] / q_prof[i]
        if not np.isfinite(out_psin_r[i]):
            raise ValueError("PQ/psin strict solve produced invalid psi_r")

    # F*Ln/q is signed physical psi_r.  Keep that sign in alpha2, then normalize
    # the solver psin_r branch back to a positive [0, 1] coordinate.
    alpha2 = _normalize_pq_signed_psi_r(out_psin_r, weights, rho, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)

    if not has_beta:
        alpha1 = -weighted_dot(heat_input, out_psin_r, weights)
        for i in range(n):
            out_Pn_psin[i] = heat_input[i] / alpha1
    _validate_pq_source_scalar(alpha1, 1)

    full_differentiation(F_r, F_solved, differentiator)

    for i in range(n):
        if abs(Ln_r[i]) <= 1.0e-14:
            raise ValueError("PQ/psin strict solve received invalid Ln_r")
        out_FFn_psin[i] = (q_prof[i] * F_r[i] / Ln_r[i]) / alpha1
        if not np.isfinite(out_FFn_psin[i]) or not np.isfinite(out_Pn_psin[i]):
            raise ValueError("PQ/psin strict solve produced non-finite normalized source")
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PQ", "psin", "grid"))
@njit(cache=True, nogil=True)
def _update_pq_from_psin_grid_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, _, Ln_r, _, _, _ = _source_geometry_workspace_views(radial_fields, surface_fields)
    n = rho.shape[0]
    edge_F = R0 * B0
    if not np.isfinite(edge_F) or abs(edge_F) <= 1.0e-14:
        raise ValueError("PQ/psin strict solve received invalid edge F")

    W = array_scratch[_SLOT_INTEGRAND]
    q_prof = array_scratch[_SLOT_AUX0]
    coeff_d = array_scratch[_SLOT_AUX1]
    coeff_y = array_scratch[_SLOT_AUX2]
    rhs = array_scratch[_SLOT_PNr]
    F_solved = array_scratch[_SLOT_Pr]
    F_r = array_scratch[_SLOT_Fr]
    A = array_scratch[_SLOT_PQ_MATRIX : _SLOT_PQ_MATRIX + n, :]

    _fill_pq_q_profile(q_prof, current_input, Kn, Ln_r, edge_F, Ip)
    _fill_pq_W_and_derivative(W, F_r, Kn, Ln_r, q_prof, differentiator)

    # Grid and uniform psin variants share the same strict-q algebra after the
    # source inputs have been materialized onto the operator grid.
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    for i in range(n):
        coeff_d[i] = W[i] + q_prof[i]
        coeff_y[i] = F_r[i]
        if not np.isfinite(coeff_d[i]) or not np.isfinite(coeff_y[i]):
            raise ValueError("PQ/psin strict solve assembled non-finite matrix")

    has_beta = not np.isnan(beta)
    if has_beta:
        # Solve A F0 = b_edge and A F1 = b_pressure, then determine alpha1 from
        # the scalar beta constraint with F = F0 + alpha1 * F1.
        for i in range(n):
            F_solved[i] = 0.0
            W[i] = -pressure_factor * V_r[i] * heat_input[i]
            if not np.isfinite(W[i]):
                raise ValueError("PQ/psin strict beta solve assembled non-finite pressure RHS")
        _fill_pq_linear_matrix_two_rhs(
            A,
            F_solved,
            W,
            differentiator,
            coeff_d,
            coeff_y,
            F_solved,
            W,
            edge_F,
            0.0,
            n,
        )
        _dense_solve_two_rhs_inplace(A, F_solved, W, n, 1.0e-12)

        beta_target = 0.5 * beta * B0**2 * dot(V_r, weights)
        alpha1 = _solve_pq_psin_beta_alpha1(
            F_solved,
            W,
            q_prof,
            Ln_r,
            heat_input,
            V_r,
            weights,
            accumulator,
            out_psin_r,
            coeff_d,
            coeff_y,
            beta_target,
        )
        for i in range(n):
            F_solved[i] = F_solved[i] + alpha1 * W[i]
        copy_into(out_Pn_psin, heat_input)
    else:
        for i in range(n):
            rhs[i] = -pressure_factor * V_r[i] * heat_input[i]
            if not np.isfinite(rhs[i]):
                raise ValueError("PQ/psin strict solve assembled non-finite pressure RHS")
        _fill_pq_linear_matrix(A, rhs, differentiator, coeff_d, coeff_y, rhs, edge_F, n)
        copy_into(F_solved, rhs)
        _dense_solve_one_rhs_inplace(A, F_solved, n, 1.0e-12)
        alpha1 = 0.0

    for i in range(n):
        out_psin_r[i] = F_solved[i] * Ln_r[i] / q_prof[i]
        if not np.isfinite(out_psin_r[i]):
            raise ValueError("PQ/psin strict solve produced invalid psi_r")

    alpha2 = _normalize_pq_signed_psi_r(out_psin_r, weights, rho, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)

    if not has_beta:
        alpha1 = -weighted_dot(heat_input, out_psin_r, weights)
        for i in range(n):
            out_Pn_psin[i] = heat_input[i] / alpha1
    _validate_pq_source_scalar(alpha1, 1)

    full_differentiation(F_r, F_solved, differentiator)

    for i in range(n):
        if abs(Ln_r[i]) <= 1.0e-14:
            raise ValueError("PQ/psin strict solve received invalid Ln_r")
        out_FFn_psin[i] = (q_prof[i] * F_r[i] / Ln_r[i]) / alpha1
        if not np.isfinite(out_FFn_psin[i]) or not np.isfinite(out_Pn_psin[i]):
            raise ValueError("PQ/psin strict solve produced non-finite normalized source")
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    return alpha1, alpha2


@register_source_route(
    ("PQ", "rho", "uniform"),
    ("PQ", "rho", "grid"),
)
@njit(cache=True, nogil=True)
def _update_pq_from_rho_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    coordinate_code: int,
    R0: float,
    B0: float,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    grid_radial_fields: np.ndarray,
    n_axis_fix: int,
    radial_fields: np.ndarray,
    surface_fields: np.ndarray,
    F_fields: np.ndarray,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    rho = _source_grid_rho(grid_radial_fields)
    V_r, Kn, _, Ln_r, _, _, _ = _source_geometry_workspace_views(radial_fields, surface_fields)
    n = rho.shape[0]
    edge_F = R0 * B0
    if not np.isfinite(edge_F) or abs(edge_F) <= 1.0e-14:
        raise ValueError("PQ/rho strict solve received invalid edge F")

    W = array_scratch[_SLOT_INTEGRAND]
    q_prof = array_scratch[_SLOT_AUX0]
    coeff_d = array_scratch[_SLOT_AUX1]
    coeff_y = array_scratch[_SLOT_AUX2]
    rhs = array_scratch[_SLOT_PNr]
    Y = array_scratch[_SLOT_Pr]
    Y_r = array_scratch[_SLOT_Fr]
    A = array_scratch[_SLOT_PQ_MATRIX : _SLOT_PQ_MATRIX + n, :]

    _fill_pq_q_profile(q_prof, current_input, Kn, Ln_r, edge_F, Ip)
    _fill_pq_W_and_derivative(W, Y_r, Kn, Ln_r, q_prof, differentiator)

    # In rho-coordinate PQ, solving for Y=F**2 keeps the strict edge condition
    # sign-safe; F is recovered only after the dense system succeeds.
    has_beta = not np.isnan(beta)
    pressure_scale = 1.0
    beta_C = 0.0
    if has_beta:
        copy_into(rhs, heat_input)
        _compute_Pn_out(coeff_y, rhs, accumulator, weights)
        beta_den_pre = weighted_dot(coeff_y, V_r, weights)
        if not np.isfinite(beta_den_pre) or abs(beta_den_pre) <= 1.0e-14:
            raise ValueError("PQ/rho strict beta solve produced invalid pressure integral")
        beta_C = 0.5 * beta * B0**2 * dot(V_r, weights) / beta_den_pre
        pressure_scale = beta_C

    pressure_factor = 1.0 / (2.0 * np.pi**2)
    for i in range(n):
        coeff_d[i] = W[i] + q_prof[i]
        coeff_y[i] = 2.0 * Y_r[i]
        rhs[i] = -pressure_factor * pressure_scale * V_r[i] * heat_input[i] * q_prof[i] / Ln_r[i]
        if not np.isfinite(coeff_d[i]) or not np.isfinite(coeff_y[i]) or not np.isfinite(rhs[i]):
            raise ValueError("PQ/rho strict solve assembled non-finite system")

    _fill_pq_linear_matrix(A, rhs, differentiator, coeff_d, coeff_y, rhs, edge_F * edge_F, n)
    copy_into(Y, rhs)
    _dense_solve_one_rhs_inplace(A, Y, n, 1.0e-12)

    sign_F = 1.0
    if edge_F < 0.0:
        sign_F = -1.0
    for i in range(n):
        if not np.isfinite(Y[i]) or Y[i] <= 0.0:
            raise ValueError("PQ/rho strict solve produced non-positive F squared")
        F_i = sign_F * np.sqrt(Y[i])
        out_psin_r[i] = F_i * Ln_r[i] / q_prof[i]
        if not np.isfinite(out_psin_r[i]):
            raise ValueError("PQ/rho strict solve produced invalid psi_r")

    alpha2 = _normalize_pq_signed_psi_r(out_psin_r, weights, rho, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)

    if has_beta:
        scaled_ratio_into(out_Pn_psin, heat_input, out_psin_r, 1.0)
        alpha1 = beta_C / alpha2
    else:
        alpha1 = -dot(heat_input, weights) / alpha2
        for i in range(n):
            denom = alpha1 * alpha2 * out_psin_r[i]
            if abs(denom) <= 1.0e-14:
                raise ValueError("PQ/rho strict solve produced invalid pressure denominator")
            out_Pn_psin[i] = heat_input[i] / denom
    _validate_pq_source_scalar(alpha1, 1)

    full_differentiation(Y_r, Y, differentiator)
    for i in range(n):
        denom = alpha1 * alpha2 * out_psin_r[i]
        if abs(denom) <= 1.0e-14:
            raise ValueError("PQ/rho strict solve produced invalid FFn denominator")
        out_FFn_psin[i] = 0.5 * Y_r[i] / denom
        if not np.isfinite(out_FFn_psin[i]) or not np.isfinite(out_Pn_psin[i]):
            raise ValueError("PQ/rho strict solve produced non-finite normalized source")
    _regularize_ffn_psin(out_FFn_psin, rho, n_axis_fix)
    return alpha1, alpha2


def update_fourier_family_fields(
    out_c_fields: np.ndarray,
    out_s_fields: np.ndarray,
    base_c_fields: np.ndarray,
    base_s_fields: np.ndarray,
    profile_fields: np.ndarray,
    c_source_profile_ids: np.ndarray,
    s_source_profile_ids: np.ndarray,
    c_active_order: int,
    s_active_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine c/s Fourier family base fields into active family fields.

    Orders whose source profile id is negative keep their boundary/base fields;
    orders above the active truncation are zeroed so stale higher-order profiles
    cannot leak from an earlier case into the current geometry update.
    """
    if out_c_fields.ndim != 3 or out_s_fields.ndim != 3:
        raise ValueError(
            f"Expected 3D c/s outputs, got {out_c_fields.shape} and {out_s_fields.shape}"
        )
    if base_c_fields.shape != out_c_fields.shape or base_s_fields.shape != out_s_fields.shape:
        raise ValueError(
            f"Base/output c/s shape mismatch: {base_c_fields.shape} and {base_s_fields.shape}"
        )
    if profile_fields.ndim != 3:
        raise ValueError(f"Expected profile_fields to be 3D, got {profile_fields.shape}")
    if c_source_profile_ids.ndim != 1 or s_source_profile_ids.ndim != 1:
        raise ValueError(
            f"Expected 1D c/s profile ids, got "
            f"{c_source_profile_ids.shape} and {s_source_profile_ids.shape}"
        )

    _update_fourier_family_fields_impl(
        out_c_fields,
        out_s_fields,
        base_c_fields,
        base_s_fields,
        profile_fields,
        c_source_profile_ids,
        s_source_profile_ids,
        int(c_active_order),
        int(s_active_order),
    )
    return out_c_fields, out_s_fields


@njit(cache=True, fastmath=True, nogil=True)
def _update_fixed_point_psin_query_impl(
    query: np.ndarray,
    psin: np.ndarray,
    max_residual: float,
) -> bool:
    max_abs_diff = 0.0
    for i in range(query.shape[0]):
        diff = abs(psin[i] - query[i])
        if diff > max_abs_diff:
            max_abs_diff = diff
        query[i] = psin[i]
    return max_abs_diff <= max_residual


@njit(cache=True, fastmath=True, nogil=True)
def _update_fixed_point_psin_query_and_spline_uniform_inputs_impl(
    query: np.ndarray,
    psin: np.ndarray,
    max_residual: float,
    out_heat_input: np.ndarray,
    out_current_input: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    heat_spline_coeff: np.ndarray,
    current_spline_coeff: np.ndarray,
) -> bool:
    max_abs_diff = 0.0
    for i in range(query.shape[0]):
        q = psin[i]
        diff = abs(q - query[i])
        if diff > max_abs_diff:
            max_abs_diff = diff
        query[i] = q

    _uniform_spline_interpolate_pair(
        out_heat_input,
        out_current_input,
        heat_spline_coeff,
        current_spline_coeff,
        query,
    )
    return max_abs_diff <= max_residual


@njit(cache=True, fastmath=True, nogil=True)
def _update_fixed_point_psin_query_and_local_barycentric_inputs_impl(
    query: np.ndarray,
    psin: np.ndarray,
    max_residual: float,
    out_heat_input: np.ndarray,
    out_current_input: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    weights: np.ndarray,
) -> bool:
    max_abs_diff = 0.0
    source_sample_count = heat_input.shape[0]
    if source_sample_count == 1:
        heat0 = heat_input[0]
        current0 = current_input[0]
        for i in range(query.shape[0]):
            q = psin[i]
            diff = abs(q - query[i])
            if diff > max_abs_diff:
                max_abs_diff = diff
            query[i] = q
            out_heat_input[i] = heat0
            out_current_input[i] = current0
        return max_abs_diff <= max_residual

    local_size = weights.shape[0]
    denom_scale = source_sample_count - 1.0
    for i in range(query.shape[0]):
        q = psin[i]
        diff = abs(q - query[i])
        if diff > max_abs_diff:
            max_abs_diff = diff
        query[i] = q

        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0

        start = _local_uniform_stencil_start(q, source_sample_count, local_size)
        hit = -1
        for local_j in range(local_size):
            j = start + local_j
            xj = j / denom_scale
            if abs(q - xj) <= 1e-14:
                hit = j
                break
        if hit >= 0:
            out_heat_input[i] = heat_input[hit]
            out_current_input[i] = current_input[hit]
            continue

        denominator = 0.0
        numerator_heat = 0.0
        numerator_current = 0.0
        for local_j in range(local_size):
            j = start + local_j
            term = weights[local_j] / (q - j / denom_scale)
            denominator += term
            numerator_heat += term * heat_input[j]
            numerator_current += term * current_input[j]
        out_heat_input[i] = numerator_heat / denominator
        out_current_input[i] = numerator_current / denominator
    return max_abs_diff <= max_residual


@njit(cache=True, fastmath=True, nogil=True)
def _materialize_profile_owned_psin_source_impl(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_source_psin_query: np.ndarray,
    out_parameter_query: np.ndarray,
    out_heat_input: np.ndarray,
    out_current_input: np.ndarray,
    psin_fields: np.ndarray,
    heat_input: np.ndarray,
    current_input: np.ndarray,
    heat_spline_coeff: np.ndarray,
    current_spline_coeff: np.ndarray,
    parameterization_code: int,
    grid_radial_fields: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    n_axis_fix: int,
    barycentric_weights: np.ndarray,
    use_barycentric: bool,
) -> None:
    rho = _source_grid_rho(grid_radial_fields)
    # Copy only psin_r from optimized profile fields; psin and psin_rr are
    # reconstructed so all source paths share the same axis regularization and
    # integration conventions.
    for i in range(out_psin.shape[0]):
        out_psin_r[i] = psin_fields[1, i]

    _regularize_psin_r(out_psin_r, rho, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator)

    for i in range(out_psin.shape[0]):
        psin_value = out_psin[i]
        out_source_psin_query[i] = psin_value
        out_parameter_query[i] = psin_value

    if parameterization_code == SOURCE_PARAMETERIZATION_CODE_SQRT_PSIN:
        # PP/psin/uniform accepts samples in sqrt(psin) to give more resolution
        # near the magnetic axis while keeping the internal query in psin.
        for i in range(out_parameter_query.shape[0]):
            value = out_parameter_query[i]
            if value < 0.0:
                value = 0.0
            out_parameter_query[i] = np.sqrt(value)
    elif parameterization_code != SOURCE_PARAMETERIZATION_CODE_IDENTITY:
        raise ValueError("Unsupported source parameterization code")

    if use_barycentric:
        _local_barycentric_interpolate_pair(
            out_heat_input,
            out_current_input,
            heat_input,
            current_input,
            out_parameter_query,
            barycentric_weights,
        )
    else:
        _uniform_spline_interpolate_pair(
            out_heat_input,
            out_current_input,
            heat_spline_coeff,
            current_spline_coeff,
            out_parameter_query,
        )


@njit(cache=True, fastmath=True, nogil=True)
def _update_fourier_family_fields_impl(
    out_c_fields: np.ndarray,
    out_s_fields: np.ndarray,
    base_c_fields: np.ndarray,
    base_s_fields: np.ndarray,
    profile_fields: np.ndarray,
    c_source_profile_ids: np.ndarray,
    s_source_profile_ids: np.ndarray,
    c_active_order: int,
    s_active_order: int,
) -> None:
    # The c0 mode is a regular radial profile and may be source-owned; s0 is not
    # a physical sine mode, so it always remains the base zero/placeholder field.
    for order in range(out_c_fields.shape[0]):
        if order <= c_active_order:
            profile_id = c_source_profile_ids[order]
            if profile_id >= 0:
                for d in range(out_c_fields.shape[1]):
                    for i in range(out_c_fields.shape[2]):
                        out_c_fields[order, d, i] = profile_fields[profile_id, d, i]
            else:
                for d in range(out_c_fields.shape[1]):
                    for i in range(out_c_fields.shape[2]):
                        out_c_fields[order, d, i] = base_c_fields[order, d, i]
        else:
            for d in range(out_c_fields.shape[1]):
                for i in range(out_c_fields.shape[2]):
                    out_c_fields[order, d, i] = 0.0

    for d in range(out_s_fields.shape[1]):
        for i in range(out_s_fields.shape[2]):
            out_s_fields[0, d, i] = base_s_fields[0, d, i]
    for order in range(1, out_s_fields.shape[0]):
        if order <= s_active_order:
            profile_id = s_source_profile_ids[order]
            if profile_id >= 0:
                for d in range(out_s_fields.shape[1]):
                    for i in range(out_s_fields.shape[2]):
                        out_s_fields[order, d, i] = profile_fields[profile_id, d, i]
            else:
                for d in range(out_s_fields.shape[1]):
                    for i in range(out_s_fields.shape[2]):
                        out_s_fields[order, d, i] = base_s_fields[order, d, i]
        else:
            for d in range(out_s_fields.shape[1]):
                for i in range(out_s_fields.shape[2]):
                    out_s_fields[order, d, i] = 0.0


@njit(cache=True, fastmath=True, nogil=True)
def _uniform_spline_interpolate_pair(
    out0: np.ndarray,
    out1: np.ndarray,
    coeff0: np.ndarray,
    coeff1: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    interval_count = coeff0.shape[0]
    if interval_count == 1:
        for i in range(out0.shape[0]):
            q = query[i]
            if q < 0.0:
                q = 0.0
            elif q > 1.0:
                q = 1.0
            out0[i] = ((coeff0[0, 3] * q + coeff0[0, 2]) * q + coeff0[0, 1]) * q + coeff0[0, 0]
            out1[i] = ((coeff1[0, 3] * q + coeff1[0, 2]) * q + coeff1[0, 1]) * q + coeff1[0, 0]
        return out0, out1

    denom_scale = float(interval_count)
    last_interval = interval_count - 1
    for i in range(out0.shape[0]):
        q = query[i]
        # Source queries are clipped rather than extrapolated.  Most source
        # inputs represent tabulated physical profiles on [0, 1], and allowing
        # cubic extrapolation at the edge can dominate the nonlinear solve.
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0

        position = q * denom_scale
        interval = int(position)
        if interval > last_interval:
            interval = last_interval
            t = 1.0
        else:
            t = position - interval

        out0[i] = (
            (coeff0[interval, 3] * t + coeff0[interval, 2]) * t + coeff0[interval, 1]
        ) * t + coeff0[interval, 0]
        out1[i] = (
            (coeff1[interval, 3] * t + coeff1[interval, 2]) * t + coeff1[interval, 1]
        ) * t + coeff1[interval, 0]
    return out0, out1


@njit(cache=True, fastmath=True, nogil=True)
def _local_barycentric_interpolate_pair(
    out0: np.ndarray,
    out1: np.ndarray,
    values0: np.ndarray,
    values1: np.ndarray,
    query: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_sample_count = values0.shape[0]
    if source_sample_count == 1:
        value0 = values0[0]
        value1 = values1[0]
        for i in range(out0.shape[0]):
            out0[i] = value0
            out1[i] = value1
        return out0, out1

    local_size = weights.shape[0]
    denom_scale = source_sample_count - 1.0
    for i in range(out0.shape[0]):
        q = query[i]
        # Local barycentric interpolation keeps the polynomial stencil bounded;
        # exact grid hits are special-cased below to avoid the 1/(q-x_j) pole.
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0

        start = _local_uniform_stencil_start(q, source_sample_count, local_size)
        hit = -1
        for local_j in range(local_size):
            j = start + local_j
            xj = j / denom_scale
            if abs(q - xj) <= 1e-14:
                hit = j
                break
        if hit >= 0:
            out0[i] = values0[hit]
            out1[i] = values1[hit]
            continue

        denominator = 0.0
        numerator0 = 0.0
        numerator1 = 0.0
        for local_j in range(local_size):
            j = start + local_j
            term = weights[local_j] / (q - j / denom_scale)
            denominator += term
            numerator0 += term * values0[j]
            numerator1 += term * values1[j]
        out0[i] = numerator0 / denominator
        out1[i] = numerator1 / denominator
    return out0, out1


@njit(cache=True, fastmath=True, nogil=True)
def uniform_barycentric_weights(source_sample_count: int) -> np.ndarray:
    weights = np.empty(source_sample_count, dtype=np.float64)
    weights[0] = 1.0
    for j in range(1, source_sample_count):
        weights[j] = -weights[j - 1] * (source_sample_count - j) / j
    return weights


@njit(cache=True, fastmath=True, nogil=True)
def _local_uniform_stencil_start(q: float, source_sample_count: int, stencil_size: int) -> int:
    if stencil_size >= source_sample_count:
        return 0
    pos = q * (source_sample_count - 1.0)
    center = int(pos)
    if pos > center:
        center += 1
    start = center - stencil_size // 2
    if start < 0:
        return 0
    max_start = source_sample_count - stencil_size
    if start > max_start:
        return max_start
    return start
