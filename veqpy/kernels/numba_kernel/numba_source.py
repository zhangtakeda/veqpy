"""
Module: veqpy.kernels.numba_kernel.numba_source

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
- The runtime binding layer selects one source runner as the Stage-C entrypoint.
- Each route must fill the same normalized root/source contract:
  psin, psin_r, psin_rr, Pn_psin, FFn_psin, alpha1, and alpha2.  The route
  name only changes which user source profile is treated as primitive.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numba import njit

from veqpy.kernels.numba_kernel.jit_math import (
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
from veqpy.kernels.numba_kernel.workspace.field_rows import (
    GEOMETRY_RADIAL_KN,
    GEOMETRY_RADIAL_KN_R,
    GEOMETRY_RADIAL_LN_R,
    GEOMETRY_RADIAL_S_R,
    GEOMETRY_RADIAL_V_R,
    GEOMETRY_SURFACE_JDIVR,
    GEOMETRY_SURFACE_R,
    GRID_RADIAL_R,
    RESIDUAL_ROOT_PSIN,
    RESIDUAL_ROOT_PSIN_R,
    RESIDUAL_ROOT_PSIN_RR,
)
from veqpy.numerics import (
    DEFAULT_LOCAL_BARYCENTRIC_STENCIL,
    build_uniform_source_interpolation_matrix,
)
from veqpy.numerics.registry import Registry

# F-coupled PJ2/PJ3 psin-uniform routes materialize psin by a fixed-point
# loop. Keep these as route constants instead of user-facing
# source-plan parameters.
PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_ITER = 16
PJ2_PSIN_UNIFORM_FIXED_POINT_MAX_RESIDUAL = 1.0e-10
PJ2_PSIN_UNIFORM_FIXED_POINT_FINALIZE_ITER = 8
PJ2_PSIN_UNIFORM_BARYCENTRIC_ORDER_CAP = 8

# Geometric-r PJ2/PJ3 strict closures stop as soon as their dimensionless
# Picard defect reaches the default outer VEQ residual scale.  A shared cap keeps
# source materialization bounded without paying route-specific fixed sweep counts.
PJ23_STRICT_FIXED_POINT_MAX_ITER = 10
PJ23_STRICT_FIXED_POINT_MAX_RESIDUAL = 1.0e-6

# Native sqrt(Phi_N) routes rebuild their evolving source coordinate inside
# every residual evaluation.  The loop is deliberately cold-started from
# s=r so one residual is independent of all previous evaluations.
RHO_FIXED_POINT_MAX_ITER = 16
RHO_FIXED_POINT_MAX_RESIDUAL = 1.0e-6
PSIN_DERIVATIVE_FIXED_POINT_MAX_ITER = 64
PSIN_DERIVATIVE_FIXED_POINT_MAX_RESIDUAL = 1.0e-10

RETAINED_SOURCE_UNIFORM_SPLINE = 0
RETAINED_SOURCE_LOCAL_BARYCENTRIC = 1
RETAINED_SOURCE_GRID_BARYCENTRIC = 2
RETAINED_SOURCE_EXPLICIT_PCHIP = 3

R_COORDINATE = 0
PSIN_COORDINATE = 1
RHO_COORDINATE = 2

COORDINATE_NAMES = {
    R_COORDINATE: "r",
    PSIN_COORDINATE: "psin",
    RHO_COORDINATE: "rho",
}

COORDINATE_CODES = {
    "r": R_COORDINATE,
    "psin": PSIN_COORDINATE,
    "rho": RHO_COORDINATE,
}

UNIFORM_NODES = "uniform"
GRID_NODES = "grid"
EXPLICIT_NODES = "explicit"
NODE_NAMES = (UNIFORM_NODES, GRID_NODES, EXPLICIT_NODES)

SOURCE_PARAMETERIZATION_IDENTITY = "identity"
SOURCE_PARAMETERIZATION_SQRT_PSIN = "sqrt_psin"
SOURCE_PARAMETERIZATION_CODE_IDENTITY = 0
SOURCE_PARAMETERIZATION_CODE_SQRT_PSIN = 1

# Scratch slot indices into SourceWorkspace.array_scratch (8 + Nr rows × Nr).  These
# symbolic names are part of the hot-kernel ABI with SourceWorkspace; changing
# the row order requires updating the allocator at the same time.
_SLOT_INTEGRAND = 0
_SLOT_AUX0 = 1
_SLOT_AUX1 = 2
_SLOT_AUX2 = 3
_SLOT_PNr = 4
_SLOT_Pr = 5
_SLOT_Fr = 6
_SLOT_EFFECTIVE_DRIVER = 7
_SLOT_PQ_MATRIX = 8

RouteKey = tuple[str, str, str]

SOURCE_ROUTE_KEYS: tuple[RouteKey, ...] = tuple(
    (route, coordinate, nodes)
    for route in ("PF", "PP", "PI", "PJ1", "PJ2", "PJ3", "PQ")
    for coordinate in ("r", "psin", "rho")
    for nodes in NODE_NAMES
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

    Source route registration is intentionally tuple-only at the runtime boundary:
    each key must be a three-string tuple such as ``("PJ1", "r", "uniform")``.
    Friendly route-name aliases belong at the model/source-plan boundary, not
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
        if key[2] == EXPLICIT_NODES:
            base = ROUTE_REGISTRY[(key[0], key[1], UNIFORM_NODES)]
            return _SourceRouteSpec(
                route=base.route,
                coordinate=base.coordinate,
                coordinate_code=base.coordinate_code,
                nodes=EXPLICIT_NODES,
                implementation=base.implementation,
            )
        supported = ", ".join("/".join(route_key) for route_key in sorted(ROUTE_REGISTRY))
        raise KeyError(
            f"Unknown source route {route!r}/{coordinate!r}/{nodes!r}; supported: {supported}"
        ) from exc


def source_parameterization_for_route_key(route_key: RouteKey | str) -> str:
    """Return the source-input parameterization for a registered concrete route key."""

    normalized_key = _normalize_route_key(route_key)
    validate_route(*normalized_key)
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
def _source_grid_r(grid_radial_fields: np.ndarray) -> np.ndarray:
    return grid_radial_fields[GRID_RADIAL_R]


@njit(cache=True, nogil=True, inline="always")
def _source_grid_axis_weights(grid_radial_fields: np.ndarray) -> np.ndarray:
    return grid_radial_fields[-2]


@njit(cache=True, nogil=True, inline="always")
def _source_grid_edge_weights(grid_radial_fields: np.ndarray) -> np.ndarray:
    return grid_radial_fields[-1]


@njit(cache=True, nogil=True, inline="always")
def _axis_eval(profile: np.ndarray, grid_radial_fields: np.ndarray) -> float:
    return dot(profile, _source_grid_axis_weights(grid_radial_fields))


@njit(cache=True, nogil=True, inline="always")
def _edge_eval(profile: np.ndarray, grid_radial_fields: np.ndarray) -> float:
    return dot(profile, _source_grid_edge_weights(grid_radial_fields))


@njit(cache=True, nogil=True, inline="always")
def _full_integral(profile: np.ndarray, weights: np.ndarray) -> float:
    return dot(profile, weights)


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
    weights: np.ndarray,
) -> np.ndarray:
    # The prefix accumulator is anchored at the physical axis, even though the
    # Gauss nodes do not include r=0.  Normalize with the full [0, 1]
    # quadrature, not the first/last interior samples.
    full_integration(out_psin, psin_r, accumulator)
    return _normalize_psin_coordinate_inplace(out_psin, _full_integral(psin_r, weights))


@njit(cache=True, nogil=True)
def _normalize_psin_coordinate_inplace(psin: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale) < 1e-12:
        raise ValueError("psin does not span a valid normalized flux interval")

    for i in range(psin.shape[0]):
        psin[i] /= scale
    return psin


@njit(cache=True, nogil=True)
def _regularize_axis_linear(profile: np.ndarray, r: np.ndarray, n_fix: int) -> np.ndarray:
    if n_fix <= 0:
        return profile

    # Axis-near derivatives are ill-conditioned in r coordinates.  Fit the
    # smooth ratio profile/r against r**2 outside the affected region and
    # extrapolate inward.
    anchor0 = n_fix
    anchor1 = n_fix + 1
    r0 = r[anchor0]
    r1 = r[anchor1]
    x0 = r0 * r0
    x1 = r1 * r1

    slope0 = profile[anchor0] / r0
    slope1 = profile[anchor1] / r1
    slope_gradient = (slope1 - slope0) / (x1 - x0)
    for i in range(n_fix):
        x = r[i] * r[i]
        profile[i] = r[i] * (slope0 + slope_gradient * (x - x0))

    return profile


@njit(cache=True, nogil=True)
def _regularize_psin_r(psin_r: np.ndarray, r: np.ndarray, n_fix: int) -> np.ndarray:
    """Repair and floor ``psin_r`` before downstream divisions.

    ``n_fix`` is the number of head samples whose ``r`` lies inside the
    axis-affected region.  It is pre-computed during operator setup from the
    grid ``r`` array and the ``fix_r`` threshold.

    The first two samples outside the affected region (indices ``n_fix`` and
    ``n_fix + 1``) serve as clean anchors.  Extrapolate the smooth even ratio
    ``psin_r / r`` as a linear function of ``r^2`` back to all head samples,
    then enforce the single runtime-level positive floor used by psin-space
    divisions.
    """
    _regularize_axis_linear(psin_r, r, n_fix)
    for i in range(psin_r.shape[0]):
        if psin_r[i] < 1.0e-10:
            psin_r[i] = 1.0e-10
    return psin_r


@njit(cache=True, nogil=True)
def _regularize_psin_r_with_derivative(
    psin_r: np.ndarray,
    psin_rr: np.ndarray,
    r: np.ndarray,
    n_fix: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the PP axis extension to a field and its analytic derivative."""
    if n_fix > 0:
        anchor0 = n_fix
        anchor1 = n_fix + 1
        r0 = r[anchor0]
        r1 = r[anchor1]
        x0 = r0 * r0
        x1 = r1 * r1
        slope0 = psin_r[anchor0] / r0
        slope1 = psin_r[anchor1] / r1
        slope_gradient = (slope1 - slope0) / (x1 - x0)
        for i in range(n_fix):
            x = r[i] * r[i]
            psin_r[i] = r[i] * (slope0 + slope_gradient * (x - x0))
            psin_rr[i] = slope0 + slope_gradient * (3.0 * x - x0)

    # Keep the existing runtime floor semantics.  Where clipping is active the
    # realized field is locally constant, so its consistent derivative is zero.
    for i in range(psin_r.shape[0]):
        if psin_r[i] < 1.0e-10:
            psin_r[i] = 1.0e-10
            psin_rr[i] = 0.0
    return psin_r, psin_rr


@njit(cache=True, nogil=True)
def _floor_signed_current_primitive(profile: np.ndarray, edge: float) -> np.ndarray:
    """Apply a tiny same-sign floor to cumulative current primitives."""
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


@njit(cache=True, nogil=True)
def _regularize_axis_even(profile: np.ndarray, r: np.ndarray, n_fix: int) -> np.ndarray:
    if n_fix <= 0:
        return profile

    # Even profiles have zero first derivative at the magnetic axis.  Linear
    # extrapolation in r**2 preserves that parity better than in r.
    anchor0 = n_fix
    anchor1 = n_fix + 1
    x0 = r[anchor0] * r[anchor0]
    x1 = r[anchor1] * r[anchor1]
    value0 = profile[anchor0]
    value1 = profile[anchor1]
    value_gradient = (value1 - value0) / (x1 - x0)
    for i in range(n_fix):
        x = r[i] * r[i]
        profile[i] = value0 + value_gradient * (x - x0)

    return profile


@njit(cache=True, nogil=True)
def _regularize_ffn_psin(FFn_psin: np.ndarray, r: np.ndarray, n_fix: int) -> np.ndarray:
    return _regularize_axis_even(FFn_psin, r, n_fix)


@njit(cache=True, nogil=True)
def _enforce_axis_even_profile(profile: np.ndarray, r: np.ndarray) -> np.ndarray:
    if profile.shape[0] < 3:
        return profile
    x1 = r[1] * r[1]
    x2 = r[2] * r[2]
    if abs(x2 - x1) < 1e-14:
        return profile
    slope = (profile[2] - profile[1]) / (x2 - x1)
    intercept = profile[1] - slope * x1
    profile[0] = intercept + slope * r[0] * r[0]
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


@njit(cache=True, nogil=True)
def _build_unscaled_pressure_profile(
    out_pressure: np.ndarray,
    radial_scratch: np.ndarray,
    pprime_input: np.ndarray,
    coordinate_code: int,
    scaled_p0: float,
    alpha2: float,
    psin_r: np.ndarray,
    accumulator: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Build mu0*p before the route-level common pressure multiplier."""
    if coordinate_code == R_COORDINATE:
        _compute_Pn_out(out_pressure, pprime_input, accumulator, weights)
    else:
        product_into(radial_scratch, pprime_input, psin_r)
        _compute_Pn_out(out_pressure, radial_scratch, accumulator, weights)
        out_pressure *= alpha2
    out_pressure += scaled_p0
    return out_pressure


@njit(cache=True, nogil=True)
def _pressure_alpha_fallback(
    alpha1: float,
    pprime_input: np.ndarray,
    coordinate_code: int,
    scaled_p0: float,
    alpha2: float,
    psin_r: np.ndarray,
    accumulator: np.ndarray,
    weights: np.ndarray,
    pressure_scratch: np.ndarray,
    radial_scratch: np.ndarray,
) -> float:
    """Avoid the legacy integral-of-pprime zero when full p is non-zero."""
    if np.isfinite(alpha1) and abs(alpha1) > 1.0e-14:
        return alpha1
    if not np.isfinite(alpha2) or abs(alpha2) <= 1.0e-14:
        raise ValueError("Pressure normalization received invalid alpha2")
    _build_unscaled_pressure_profile(
        pressure_scratch,
        radial_scratch,
        pprime_input,
        coordinate_code,
        scaled_p0,
        alpha2,
        psin_r,
        accumulator,
        weights,
    )
    pressure_scale = 0.0
    for i in range(pressure_scratch.shape[0]):
        value = abs(pressure_scratch[i])
        if value > pressure_scale:
            pressure_scale = value
    if not np.isfinite(pressure_scale) or pressure_scale <= 1.0e-14:
        raise ValueError("The complete pressure profile is identically zero")
    return pressure_scale / alpha2


@njit(cache=True, nogil=True)
def _ensure_pressure_alpha1(
    alpha1: float,
    pprime_input: np.ndarray,
    coordinate_code: int,
    scaled_p0: float,
    alpha2: float,
    psin_r: np.ndarray,
    accumulator: np.ndarray,
    weights: np.ndarray,
    array_scratch: np.ndarray,
) -> float:
    return _pressure_alpha_fallback(
        alpha1,
        pprime_input,
        coordinate_code,
        scaled_p0,
        alpha2,
        psin_r,
        accumulator,
        weights,
        array_scratch[_SLOT_PQ_MATRIX],
        array_scratch[_SLOT_PQ_MATRIX + 1],
    )


@njit(cache=True, nogil=True)
def finalize_pressure_normalization(
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    coordinate_code: int,
    scaled_p0: float,
    beta: float,
    alpha1: float,
    alpha2: float,
    psin_r: np.ndarray,
    accumulator: np.ndarray,
    weights: np.ndarray,
    pressure_scratch: np.ndarray,
    radial_scratch: np.ndarray,
    pressure_state: np.ndarray,
) -> float:
    """Normalize alpha1 with max(abs(mu0*p)) while preserving physical sources."""
    if not np.isfinite(alpha1) or abs(alpha1) <= 1.0e-14:
        raise ValueError("Pressure normalization received invalid provisional alpha1")
    if not np.isfinite(alpha2) or abs(alpha2) <= 1.0e-14:
        raise ValueError("Pressure normalization received invalid alpha2")

    # Infer the common multiplier that the selected route applied to the raw
    # pressure derivative. This covers unconstrained, Ip, beta, and Ip+beta
    # branches without tying the normalization layer to route names.
    numerator = 0.0
    denominator = 0.0
    for i in range(pprime_input.shape[0]):
        raw = pprime_input[i]
        denominator += raw * raw * weights[i]
        if coordinate_code == R_COORDINATE:
            realized = alpha1 * alpha2 * out_Pn_psin[i] * psin_r[i]
        else:
            realized = alpha1 * out_Pn_psin[i]
        numerator += realized * raw * weights[i]
    if denominator > 1.0e-28:
        pressure_multiplier = numerator / denominator
    elif np.isfinite(beta):
        pressure_multiplier = alpha1 * alpha2 if coordinate_code == R_COORDINATE else alpha1
    else:
        # With pprime == 0, p0 is already an absolute pressure unless a beta
        # branch explicitly selected a different multiplier before this helper.
        pressure_multiplier = 1.0

    if not np.isfinite(pressure_multiplier):
        raise ValueError("Pressure normalization produced a non-finite pressure multiplier")
    _build_unscaled_pressure_profile(
        pressure_scratch,
        radial_scratch,
        pprime_input,
        coordinate_code,
        scaled_p0,
        alpha2,
        psin_r,
        accumulator,
        weights,
    )
    pressure_scale = 0.0
    for i in range(pressure_scratch.shape[0]):
        pressure_scratch[i] *= pressure_multiplier
        value = abs(pressure_scratch[i])
        if value > pressure_scale:
            pressure_scale = value
    if not np.isfinite(pressure_scale) or pressure_scale <= 1.0e-14:
        raise ValueError("The complete pressure profile is identically zero")

    normalized_alpha1 = pressure_scale / alpha2
    if not np.isfinite(normalized_alpha1) or abs(normalized_alpha1) <= 1.0e-14:
        raise ValueError("Pressure normalization produced invalid alpha1")
    source_rescale = alpha1 / normalized_alpha1
    for i in range(out_Pn_psin.shape[0]):
        out_Pn_psin[i] *= source_rescale
        out_FFn_psin[i] *= source_rescale
    pressure_state[0] = pressure_multiplier * scaled_p0
    pressure_state[1] = pressure_multiplier
    return normalized_alpha1


@njit(cache=True, nogil=True)
def _r_beta_pressure_denominator(
    relative_pressure: np.ndarray,
    V_r: np.ndarray,
    weights: np.ndarray,
    scaled_p0: float,
) -> float:
    return weighted_dot(relative_pressure, V_r, weights) + scaled_p0 * dot(V_r, weights)


@njit(cache=True, nogil=True)
def _psin_beta_pressure_denominator(
    relative_pressure: np.ndarray,
    V_r: np.ndarray,
    weights: np.ndarray,
    scaled_p0: float,
    alpha2: float,
) -> float:
    return alpha2 * weighted_dot(relative_pressure, V_r, weights) + scaled_p0 * dot(V_r, weights)


@njit(cache=True, nogil=True)
def _solve_pf_psin_beta_alpha1(
    relative_pressure: np.ndarray,
    V_r: np.ndarray,
    weights: np.ndarray,
    scaled_p0: float,
    alpha2_per_alpha1: float,
    beta_target: float,
) -> float:
    # beta_target = alpha1*p0*V + alpha1**2*c2*<p_rel>.
    quadratic = alpha2_per_alpha1 * weighted_dot(relative_pressure, V_r, weights)
    linear = scaled_p0 * dot(V_r, weights)
    if abs(quadratic) <= 1.0e-14:
        if abs(linear) <= 1.0e-14:
            raise ValueError("PF/psin beta constraint received zero complete pressure")
        return beta_target / linear
    discriminant = linear * linear + 4.0 * quadratic * beta_target
    if not np.isfinite(discriminant) or discriminant < 0.0:
        raise ValueError("PF/psin beta constraint has no real pressure scale")
    root = np.sqrt(discriminant)
    root_plus = (-linear + root) / (2.0 * quadratic)
    root_minus = (-linear - root) / (2.0 * quadratic)
    preferred = _signed_sqrt_ratio(
        beta_target,
        quadratic,
    )
    if preferred >= 0.0:
        if root_plus >= 0.0:
            return root_plus
        return root_minus
    if root_minus <= 0.0:
        return root_minus
    return root_plus


@njit(cache=True, nogil=True)
def _fill_pf_r_integrand(
    out: np.ndarray,
    Kn: np.ndarray,
    driver_input: np.ndarray,
    Ln_r: np.ndarray,
    V_r: np.ndarray,
    pprime_input: np.ndarray,
) -> np.ndarray:
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    for i in range(out.shape[0]):
        out[i] = Kn[i] * (driver_input[i] * Ln_r[i] + V_r[i] * pprime_input[i] * pressure_factor)
    return out


@njit(cache=True, nogil=True)
def _fill_pf_psin_integrand(
    out: np.ndarray,
    driver_input: np.ndarray,
    Ln_r: np.ndarray,
    V_r: np.ndarray,
    pprime_input: np.ndarray,
) -> np.ndarray:
    pressure_factor = 1.0 / (4.0 * np.pi**2)
    for i in range(out.shape[0]):
        out[i] = driver_input[i] * Ln_r[i] + V_r[i] * pprime_input[i] * pressure_factor
    return out


@njit(cache=True, nogil=True)
def _weighted_profile_sign(values: np.ndarray, weights: np.ndarray) -> float:
    weighted = dot(values, weights)
    if weighted < 0.0:
        return -1.0
    return 1.0


@njit(cache=True, nogil=True)
def _signed_sqrt_ratio(numerator: float, denominator: float) -> float:
    ratio = numerator / denominator
    if ratio < 0.0:
        return -np.sqrt(-ratio)
    return np.sqrt(ratio)


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
def _fill_g1n_r_integrand(
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


@njit(cache=True, nogil=True)
def _g1n_psin_integral_from_radial_moments(
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
    Ln_r: np.ndarray,
    V_r: np.ndarray,
    weights: np.ndarray,
) -> float:
    total = 0.0
    two_pi = 2.0 * np.pi
    inv_two_pi = 1.0 / two_pi
    for i in range(FFn_psin.shape[0]):
        total += weights[i] * (two_pi * Ln_r[i] * FFn_psin[i] + inv_two_pi * V_r[i] * Pn_psin[i])
    return total


@njit(cache=True, nogil=True)
def _g1n_r_integral_from_radial_moments(
    FFn_r: np.ndarray,
    Pn_r: np.ndarray,
    psin_r: np.ndarray,
    Ln_r: np.ndarray,
    V_r: np.ndarray,
    weights: np.ndarray,
    source_scale: float,
) -> float:
    total = 0.0
    two_pi = 2.0 * np.pi
    inv_two_pi = 1.0 / two_pi
    for i in range(FFn_r.shape[0]):
        total += (
            weights[i]
            * source_scale
            / psin_r[i]
            * (two_pi * Ln_r[i] * FFn_r[i] + inv_two_pi * V_r[i] * Pn_r[i])
        )
    return total


@njit(cache=True, nogil=True)
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
        out[i] = -(term0 + term1) / Ln_r[i]
    return out


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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
        out[i] = -(term0 + term1) / Ln_r[i]
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
    accumulator: np.ndarray,
    weights: np.ndarray,
    coeff_d: np.ndarray,
    coeff_y: np.ndarray,
    forcing: np.ndarray,
    edge_value: float,
    n: int,
) -> None:
    """Assemble the edge-conditioned integral form of a first-order PQ ODE."""
    for i in range(n):
        rhs_i = edge_value
        for j in range(n):
            if abs(coeff_d[j]) <= 1.0e-14:
                raise ValueError("PQ integral solve received near-zero derivative coefficient")
            edge_relative_integral = accumulator[i, j] - weights[j]
            A[i, j] = edge_relative_integral * coeff_y[j] / coeff_d[j]
            rhs_i += edge_relative_integral * forcing[j] / coeff_d[j]
        A[i, i] += 1.0
        rhs[i] = rhs_i


@njit(cache=True, nogil=True)
def _fill_pq_linear_matrix_two_rhs(
    A: np.ndarray,
    rhs0: np.ndarray,
    rhs1: np.ndarray,
    accumulator: np.ndarray,
    weights: np.ndarray,
    coeff_d: np.ndarray,
    coeff_y: np.ndarray,
    forcing0: np.ndarray,
    forcing1: np.ndarray,
    edge_value0: float,
    edge_value1: float,
    n: int,
) -> None:
    """Assemble one integral PQ system with two edge-conditioned RHS vectors."""
    for i in range(n):
        rhs_i0 = edge_value0
        rhs_i1 = edge_value1
        for j in range(n):
            if abs(coeff_d[j]) <= 1.0e-14:
                raise ValueError("PQ integral solve received near-zero derivative coefficient")
            edge_relative_integral = accumulator[i, j] - weights[j]
            inv_coeff_d = 1.0 / coeff_d[j]
            A[i, j] = edge_relative_integral * coeff_y[j] * inv_coeff_d
            rhs_i0 += edge_relative_integral * forcing0[j] * inv_coeff_d
            rhs_i1 += edge_relative_integral * forcing1[j] * inv_coeff_d
        A[i, i] += 1.0
        rhs0[i] = rhs_i0
        rhs1[i] = rhs_i1


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
    r: np.ndarray,
    n_axis_fix: int,
) -> float:
    alpha2 = dot(psi_r, weights)
    _validate_pq_source_scalar(alpha2, 0)
    scale_into(psi_r, psi_r, 1.0 / alpha2)
    for i in range(psi_r.shape[0]):
        if not np.isfinite(psi_r[i]) or psi_r[i] <= 0.0:
            raise ValueError("PQ strict solve produced invalid normalized psin_r")
    _regularize_psin_r(psi_r, r, n_axis_fix)
    for i in range(psi_r.shape[0]):
        if not np.isfinite(psi_r[i]) or psi_r[i] <= 0.0:
            raise ValueError("PQ strict solve produced invalid normalized psin_r")
    return alpha2


@njit(cache=True, nogil=True)
def _fill_pq_q_profile(
    out_q: np.ndarray,
    driver_input: np.ndarray,
    Kn: np.ndarray,
    Ln_r: np.ndarray,
    edge_weights: np.ndarray,
    edge_F: float,
    Ip: float,
) -> None:
    has_Ip = not np.isnan(Ip)
    if has_Ip:
        if abs(Ip) <= 1.0e-14:
            raise ValueError("PQ strict solve received near-zero Ip")
        edge_q = dot(driver_input, edge_weights)
        edge_Kn = dot(Kn, edge_weights)
        edge_Ln_r = dot(Ln_r, edge_weights)
        if abs(edge_q) <= 1.0e-14:
            raise ValueError("PQ strict solve received near-zero edge q input")
        q_scale = (2.0 * np.pi * edge_F) / Ip
        q_scale *= edge_Kn * edge_Ln_r / edge_q
        for i in range(out_q.shape[0]):
            out_q[i] = driver_input[i] * q_scale
    else:
        for i in range(out_q.shape[0]):
            out_q[i] = driver_input[i]

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
    pprime_input: np.ndarray,
    V_r: np.ndarray,
    weights: np.ndarray,
    accumulator: np.ndarray,
    trial_psin_r: np.ndarray,
    trial_Pn_r: np.ndarray,
    trial_Pn: np.ndarray,
    scaled_p0: float,
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
        trial_Pn_r[i] = pprime_input[i] * trial_psin_r[i]
    _compute_Pn_out(trial_Pn, trial_Pn_r, accumulator, weights)
    beta_den = weighted_dot(trial_Pn, V_r, weights)
    if not np.isfinite(beta_den):
        return np.nan
    return alpha1 * (scaled_p0 * dot(V_r, weights) + alpha2 * beta_den) - beta_target


@njit(cache=True, nogil=True)
def _solve_pq_psin_beta_alpha1(
    F0: np.ndarray,
    F1: np.ndarray,
    q_prof: np.ndarray,
    Ln_r: np.ndarray,
    pprime_input: np.ndarray,
    V_r: np.ndarray,
    weights: np.ndarray,
    accumulator: np.ndarray,
    trial_psin_r: np.ndarray,
    trial_Pn_r: np.ndarray,
    trial_Pn: np.ndarray,
    scaled_p0: float,
    beta_target: float,
) -> float:
    base = 0.0
    r_base = _pq_psin_beta_residual(
        base,
        F0,
        F1,
        q_prof,
        Ln_r,
        pprime_input,
        V_r,
        weights,
        accumulator,
        trial_psin_r,
        trial_Pn_r,
        trial_Pn,
        scaled_p0,
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
            pprime_input,
            V_r,
            weights,
            accumulator,
            trial_psin_r,
            trial_Pn_r,
            trial_Pn,
            scaled_p0,
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
                        pprime_input,
                        V_r,
                        weights,
                        accumulator,
                        trial_psin_r,
                        trial_Pn_r,
                        trial_Pn,
                        scaled_p0,
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
                pprime_input,
                V_r,
                weights,
                accumulator,
                trial_psin_r,
                trial_Pn_r,
                trial_Pn,
                scaled_p0,
                beta_target,
            )
    raise ValueError("PQ/psin strict beta solve failed to bracket alpha1")


def build_source_remap_cache(
    coordinate: str,
    source_sample_count: int,
    *,
    r: np.ndarray | None = None,
    stencil_size: int = DEFAULT_LOCAL_BARYCENTRIC_STENCIL,
    interpolation_kind: str | None = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Build reusable interpolation cache data for sampled source inputs.

    r-coordinate sources are tied to the fixed operator grid, so their remap
    matrix can be built once.  Psin-coordinate sources depend on the current
    solution and only keep interpolation weights here; their query is refreshed
    at each source evaluation.
    """
    coord = str(coordinate).lower()
    if coord not in ("r", "psin", "rho"):
        raise ValueError(f"Unsupported coordinate {coordinate!r}")

    count = int(source_sample_count)
    if count < 1:
        raise ValueError(f"source_sample_count must be positive, got {source_sample_count!r}")

    if coord == "psin":
        coord_code = PSIN_COORDINATE
    elif coord == "rho":
        coord_code = RHO_COORDINATE
    else:
        coord_code = R_COORDINATE
    local_size = min(count, int(stencil_size))
    if local_size < 1:
        raise ValueError(f"stencil_size must be positive, got {stencil_size!r}")
    weights = uniform_barycentric_weights(local_size)
    fixed_remap_matrix = np.empty((0, 0), dtype=np.float64)
    if coord_code == R_COORDINATE:
        if r is None:
            raise ValueError("r is required when coordinate='r'")
        query = np.clip(np.asarray(r, dtype=np.float64), 0.0, 1.0)
        fixed_remap_matrix = build_uniform_source_interpolation_matrix(
            query, count, kind=interpolation_kind
        )

    return local_size, weights, fixed_remap_matrix


def resolve_source_inputs(
    out_pprime_input: np.ndarray,
    out_driver_input: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
    coordinate_code: int,
    source_sample_count: int,
    barycentric_weights: np.ndarray,
    fixed_remap_matrix: np.ndarray,
    pprime_spline_coeff: np.ndarray,
    driver_spline_coeff: np.ndarray,
    psin_query: np.ndarray,
    use_barycentric: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve sampled pprime/driver inputs onto operator r nodes."""

    pprime = np.asarray(pprime_input, dtype=np.float64)
    driver = np.asarray(driver_input, dtype=np.float64)
    return _resolve_source_inputs_prepared(
        out_pprime_input,
        out_driver_input,
        pprime,
        driver,
        coordinate_code,
        source_sample_count,
        barycentric_weights,
        fixed_remap_matrix,
        pprime_spline_coeff,
        driver_spline_coeff,
        psin_query,
        use_barycentric,
    )


def _resolve_source_inputs_prepared(
    out_pprime_input: np.ndarray,
    out_driver_input: np.ndarray,
    pprime: np.ndarray,
    driver: np.ndarray,
    coordinate_code: int,
    source_sample_count: int,
    barycentric_weights: np.ndarray,
    fixed_remap_matrix: np.ndarray,
    pprime_spline_coeff: np.ndarray,
    driver_spline_coeff: np.ndarray,
    psin_query: np.ndarray,
    use_barycentric: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve source inputs when all arrays are already normalized ndarrays."""

    if pprime.ndim != 1 or driver.ndim != 1:
        raise ValueError(f"Expected 1D pprime/driver inputs, got {pprime.shape} and {driver.shape}")
    if pprime.shape != driver.shape:
        raise ValueError(f"pprime/driver shape mismatch: {pprime.shape} vs {driver.shape}")
    if pprime.shape[0] != source_sample_count:
        raise ValueError(f"Expected {source_sample_count} source samples, got {pprime.shape[0]}")
    if (
        out_pprime_input.ndim != 1
        or out_driver_input.ndim != 1
        or out_pprime_input.shape != out_driver_input.shape
    ):
        raise ValueError(
            "Expected matching 1D output inputs, "
            f"got {out_pprime_input.shape} and {out_driver_input.shape}"
        )
    if psin_query.ndim != 1:
        raise ValueError(f"Expected psin_query to be 1D, got {psin_query.shape}")

    if coordinate_code == R_COORDINATE:
        # Rho inputs use the precomputed linear map; no solver state participates
        # after the cache is built.
        np.matmul(fixed_remap_matrix, pprime, out=out_pprime_input)
        np.matmul(fixed_remap_matrix, driver, out=out_driver_input)
        return out_pprime_input, out_driver_input

    if psin_query.shape != out_pprime_input.shape:
        raise ValueError(
            f"psin_query shape mismatch: {psin_query.shape} vs {out_pprime_input.shape}"
        )

    # Psin inputs are materialized against the current psin field.  Spline is
    # smoother for general sampled inputs; local barycentric keeps high-order
    # route variants allocation-free inside fixed-point loops.
    if use_barycentric:
        _local_barycentric_interpolate_pair(
            out_pprime_input,
            out_driver_input,
            pprime,
            driver,
            psin_query,
            barycentric_weights,
        )
    else:
        _uniform_spline_interpolate_pair(
            out_pprime_input,
            out_driver_input,
            pprime_spline_coeff,
            driver_spline_coeff,
            psin_query,
        )
    return out_pprime_input, out_driver_input


# ---------------------------------------------------------------------------
# Zero-allocation scratch variants (Phase 3)
# ---------------------------------------------------------------------------

# Route families share one output contract but choose different primitives:
# PF derives psin_r from the coordinate-matched P/FF derivative balance, PP
# takes psi_r directly, PI
# works through toroidal-current primitives, PJ routes start from current-density
# data, and PQ treats q as strict input. The
# repeated r/psin/uniform/grid functions below differ mainly in how the input
# profiles are interpreted or remapped; keep family-level comments here instead
# of duplicating them in every variant.
#
# Docs-facing route meanings in compact form:
# - PF: P_r/P_rho/P_psin is paired with FF_r/FF_rho/FF_psin.
# - PP: pprime is paired with psi_r.
# - PI/PJ1/PJ2/PJ3: pprime is paired with itor/jtor/jpara/jtotal respectively.
# - PQ: pprime is paired with q, so F or F**2 is solved from q and edge F.


@register_source_route(
    ("PF", "r", "uniform"),
    ("PF", "r", "grid"),
    ("PF", "rho", "uniform"),
    ("PF", "rho", "grid"),
)
@njit(cache=True, nogil=True)
def _update_pf_from_r_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
    V_r, Kn, _, Ln_r, _, _, _ = _source_geometry_workspace_views(radial_fields, surface_fields)
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand = array_scratch[_SLOT_INTEGRAND]
    _fill_pf_r_integrand(integrand, Kn, driver_input, Ln_r, V_r, pprime_input)
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
    _regularize_psin_r(out_psin_r, r, n_axis_fix)
    prof = out_psin_r
    integral_prof = dot(prof, weights)
    # alpha2 stores the pre-normalization integral; psin_r itself is normalized
    # to integrate to one so downstream geometry uses the canonical psin scale.
    out_psin_r /= integral_prof
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)
    if (not has_Ip) and (not has_beta):
        # With no global Ip/beta target, PF determines both alpha scales from
        # the integrated source profiles. r-coordinate PF inputs are
        # derivatives with respect to r, so their global sign belongs to the
        # flux-direction gauge carried by alpha2, not to the solved shape.
        alpha2 = psi_square_sign * integral_prof
        alpha1 = -dot(pprime_input, weights) / integral_prof
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
        source_scale = psi_square_sign / (alpha1 * alpha2)
        scaled_ratio_into(out_Pn_psin, pprime_input, out_psin_r, source_scale)
        scaled_ratio_into(out_FFn_psin, driver_input, out_psin_r, source_scale)
        return alpha1, alpha2
    c2 = integral_prof * integral_prof
    if has_Ip and (not has_beta):
        G1n_integral = _g1n_r_integral_from_radial_moments(
            driver_input,
            pprime_input,
            out_psin_r,
            Ln_r,
            V_r,
            weights,
            psi_square_sign,
        )
        alpha1 = -Ip / G1n_integral
    elif has_beta and (not has_Ip):
        scratch_aux = array_scratch[_SLOT_AUX0]
        _compute_Pn_out(scratch_aux, pprime_input, accumulator, weights)
        c1 = (
            0.5
            * beta
            * B0**2
            * dot(V_r, weights)
            / _r_beta_pressure_denominator(scratch_aux, V_r, weights, scaled_p0)
        )
        alpha1 = _signed_sqrt_ratio(c1, c2)
    else:
        raise ValueError("PF does not support applying Ip and beta constraints simultaneously")
    alpha2 = c2 * alpha1
    scaled_ratio_into(out_Pn_psin, pprime_input, out_psin_r, psi_square_sign)
    scaled_ratio_into(out_FFn_psin, driver_input, out_psin_r, psi_square_sign)
    return alpha1, alpha2


@register_source_route(("PF", "psin", "uniform"))
@njit(cache=True, nogil=True)
def _update_pf_from_psin_uniform_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
    V_r, Kn, _, Ln_r, _, _, _ = _source_geometry_workspace_views(radial_fields, surface_fields)
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand = array_scratch[_SLOT_INTEGRAND]
    _fill_pf_psin_integrand(integrand, driver_input, Ln_r, V_r, pprime_input)
    full_integration(out_psin_r, integrand, accumulator)
    out_psin_r *= -1.0
    out_psin_r /= Kn
    psi_scale_sign = _weighted_profile_sign(out_psin_r, weights)
    if psi_scale_sign < 0.0:
        out_psin_r *= -1.0
    _regularize_psin_r(out_psin_r, r, n_axis_fix)
    integral_prof = dot(out_psin_r, weights)
    out_psin_r /= integral_prof
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)
    if (not has_Ip) and (not has_beta):
        alpha2 = psi_scale_sign * integral_prof
        pressure_profile = array_scratch[_SLOT_AUX0]
        product_into(pressure_profile, pprime_input, out_psin_r)
        alpha1 = -dot(pressure_profile, weights)
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
        scale_into(out_Pn_psin, pprime_input, 1.0 / alpha1)
        scale_into(out_FFn_psin, driver_input, 1.0 / alpha1)
        _regularize_ffn_psin(out_FFn_psin, r, n_axis_fix)
        return alpha1, alpha2
    c2 = integral_prof
    copy_into(out_Pn_psin, pprime_input)
    copy_into(out_FFn_psin, driver_input)
    _regularize_ffn_psin(out_FFn_psin, r, n_axis_fix)
    if has_Ip and (not has_beta):
        G1n_integral = _g1n_psin_integral_from_radial_moments(
            out_FFn_psin,
            out_Pn_psin,
            Ln_r,
            V_r,
            weights,
        )
        alpha1 = -Ip / G1n_integral
    elif has_beta and (not has_Ip):
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX1]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = _solve_pf_psin_beta_alpha1(
            scratch_aux,
            V_r,
            weights,
            scaled_p0,
            c2,
            0.5 * beta * B0**2 * dot(V_r, weights),
        )
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
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
    V_r, Kn, _, Ln_r, _, _, _ = _source_geometry_workspace_views(radial_fields, surface_fields)
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand = array_scratch[_SLOT_INTEGRAND]
    _fill_pf_psin_integrand(integrand, driver_input, Ln_r, V_r, pprime_input)
    full_integration(out_psin_r, integrand, accumulator)
    out_psin_r *= -1.0
    out_psin_r /= Kn
    psi_scale_sign = _weighted_profile_sign(out_psin_r, weights)
    if psi_scale_sign < 0.0:
        out_psin_r *= -1.0
    _regularize_psin_r(out_psin_r, r, n_axis_fix)
    integral_prof = dot(out_psin_r, weights)
    out_psin_r /= integral_prof
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)
    if (not has_Ip) and (not has_beta):
        alpha2 = psi_scale_sign * integral_prof
        pressure_profile = array_scratch[_SLOT_AUX0]
        product_into(pressure_profile, pprime_input, out_psin_r)
        alpha1 = -dot(pressure_profile, weights)
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
        scale_into(out_Pn_psin, pprime_input, 1.0 / alpha1)
        scale_into(out_FFn_psin, driver_input, 1.0 / alpha1)
        _regularize_ffn_psin(out_FFn_psin, r, n_axis_fix)
        return alpha1, alpha2
    c2 = integral_prof
    copy_into(out_Pn_psin, pprime_input)
    copy_into(out_FFn_psin, driver_input)
    _regularize_ffn_psin(out_FFn_psin, r, n_axis_fix)
    if has_Ip and (not has_beta):
        G1n_integral = _g1n_psin_integral_from_radial_moments(
            out_FFn_psin,
            out_Pn_psin,
            Ln_r,
            V_r,
            weights,
        )
        alpha1 = -Ip / G1n_integral
    elif has_beta and (not has_Ip):
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX1]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = _solve_pf_psin_beta_alpha1(
            scratch_aux,
            V_r,
            weights,
            scaled_p0,
            c2,
            0.5 * beta * B0**2 * dot(V_r, weights),
        )
    else:
        raise ValueError("PF does not support applying Ip and beta constraints simultaneously")
    alpha2 = c2 * alpha1
    return alpha1, alpha2


@njit(cache=True, nogil=True)
def _update_pp_from_r_inputs_impl(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
    driver_derivative: np.ndarray,
    use_materialized_driver_derivative: bool,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    if has_Ip:
        # PP treats driver_input as the unnormalized psin_r shape.  Ip pins the
        # absolute scale through the edge value; otherwise alpha2 is the weighted
        # normalization integral.
        copy_into(out_psin_r, driver_input)
        if use_materialized_driver_derivative:
            copy_into(out_psin_rr, driver_derivative)
            _regularize_psin_r_with_derivative(out_psin_r, out_psin_rr, r, n_axis_fix)
        else:
            _regularize_psin_r(out_psin_r, r, n_axis_fix)
        alpha2 = Ip / (
            2.0
            * np.pi
            * _edge_eval(Kn, grid_radial_fields)
            * _edge_eval(out_psin_r, grid_radial_fields)
        )
    else:
        alpha2 = dot(driver_input, weights)
        scale_into(out_psin_r, driver_input, 1.0 / alpha2)
        if use_materialized_driver_derivative:
            scale_into(out_psin_rr, driver_derivative, 1.0 / alpha2)
            _regularize_psin_r_with_derivative(out_psin_r, out_psin_rr, r, n_axis_fix)
        else:
            _regularize_psin_r(out_psin_r, r, n_axis_fix)
    if not use_materialized_driver_derivative:
        full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)
    if has_beta:
        scaled_ratio_into(out_Pn_psin, pprime_input, out_psin_r, 1.0)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX0]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            * dot(V_r, weights)
            / (alpha2 * _r_beta_pressure_denominator(scratch_aux, V_r, weights, scaled_p0))
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        copy_into(scratch_Pr, pprime_input)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
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
    return alpha1, alpha2


@register_source_route(
    ("PP", "r", "uniform"),
    ("PP", "r", "grid"),
    ("PP", "rho", "uniform"),
    ("PP", "rho", "grid"),
)
@njit(cache=True, nogil=True)
def _update_pp_from_r_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    return _update_pp_from_r_inputs_impl(
        out_root_fields,
        out_FFn_psin,
        out_Pn_psin,
        pprime_input,
        driver_input,
        driver_input,
        False,
        coordinate_code,
        R0,
        B0,
        weights,
        differentiator,
        accumulator,
        grid_radial_fields,
        n_axis_fix,
        radial_fields,
        surface_fields,
        F_fields,
        scaled_p0,
        Ip,
        beta,
        array_scratch,
        matrix_scratch,
    )


@njit(cache=True, nogil=True)
def _update_pp_from_r_explicit_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
    driver_derivative: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    """PP-r closure with PCHIP and axis-consistent psi_r derivatives."""
    return _update_pp_from_r_inputs_impl(
        out_root_fields,
        out_FFn_psin,
        out_Pn_psin,
        pprime_input,
        driver_input,
        driver_derivative,
        True,
        coordinate_code,
        R0,
        B0,
        weights,
        differentiator,
        accumulator,
        grid_radial_fields,
        n_axis_fix,
        radial_fields,
        surface_fields,
        F_fields,
        scaled_p0,
        Ip,
        beta,
        array_scratch,
        matrix_scratch,
    )


@register_source_route(("PP", "psin", "uniform"))
@njit(cache=True, nogil=True)
def _update_pp_from_psin_uniform_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    if has_Ip:
        copy_into(out_psin_r, driver_input)
        _regularize_psin_r(out_psin_r, r, n_axis_fix)
        alpha2 = Ip / (
            2.0
            * np.pi
            * _edge_eval(Kn, grid_radial_fields)
            * _edge_eval(out_psin_r, grid_radial_fields)
        )
    else:
        alpha2 = dot(driver_input, weights)
        scale_into(out_psin_r, driver_input, 1.0 / alpha2)
        _regularize_psin_r(out_psin_r, r, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)
    if has_beta:
        copy_into(out_Pn_psin, pprime_input)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX0]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            * dot(V_r, weights)
            / _psin_beta_pressure_denominator(scratch_aux, V_r, weights, scaled_p0, alpha2)
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        scaled_product_into(scratch_Pr, pprime_input, out_psin_r, alpha2)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
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
    _regularize_ffn_psin(out_FFn_psin, r, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PP", "psin", "grid"))
@njit(cache=True, nogil=True)
def _update_pp_from_psin_grid_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    if has_Ip:
        copy_into(out_psin_r, driver_input)
        _regularize_psin_r(out_psin_r, r, n_axis_fix)
        alpha2 = Ip / (
            2.0
            * np.pi
            * _edge_eval(Kn, grid_radial_fields)
            * _edge_eval(out_psin_r, grid_radial_fields)
        )
    else:
        alpha2 = dot(driver_input, weights)
        scale_into(out_psin_r, driver_input, 1.0 / alpha2)
        _regularize_psin_r(out_psin_r, r, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)
    if has_beta:
        copy_into(out_Pn_psin, pprime_input)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX0]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            * dot(V_r, weights)
            / _psin_beta_pressure_denominator(scratch_aux, V_r, weights, scaled_p0, alpha2)
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        scaled_product_into(scratch_Pr, pprime_input, out_psin_r, alpha2)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
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
    _regularize_ffn_psin(out_FFn_psin, r, n_axis_fix)
    return alpha1, alpha2


@njit(cache=True, nogil=True)
def _update_pi_from_r_inputs_impl(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
    driver_derivative: np.ndarray,
    use_materialized_driver_derivative: bool,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    Itor = array_scratch[_SLOT_AUX0]
    driver_scale = 1.0
    if has_Ip:
        # PI source profiles represent cumulative toroidal current.  Rescale the
        # whole primitive when Ip is prescribed, then differentiate only after
        # psin_r has been normalized.
        driver_scale = Ip / _edge_eval(driver_input, grid_radial_fields)
        scale_into(Itor, driver_input, driver_scale)
    else:
        copy_into(Itor, driver_input)
    _floor_signed_current_primitive(Itor, _edge_eval(Itor, grid_radial_fields))
    itor_over_kn = array_scratch[_SLOT_INTEGRAND]
    scaled_ratio_into(itor_over_kn, Itor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(itor_over_kn, weights)
    scaled_ratio_into(out_psin_r, Itor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    _regularize_psin_r(out_psin_r, r, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)
    Itor_r = array_scratch[_SLOT_AUX1]
    if use_materialized_driver_derivative:
        scale_into(Itor_r, driver_derivative, driver_scale)
    else:
        full_differentiation(Itor_r, Itor, differentiator)
    _regularize_axis_linear(Itor_r, r, n_axis_fix)
    if has_beta:
        scaled_ratio_into(out_Pn_psin, pprime_input, out_psin_r, 1.0)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX2]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            * dot(V_r, weights)
            / (alpha2 * _r_beta_pressure_denominator(scratch_aux, V_r, weights, scaled_p0))
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        copy_into(scratch_Pr, pprime_input)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
        scaled_ratio_into(out_Pn_psin, scratch_Pr, out_psin_r, 1.0 / (alpha1 * alpha2))
    _fill_pi_ffn_psin(out_FFn_psin, Itor_r, V_r, out_Pn_psin, Ln_r, 1.0 / (2.0 * np.pi * alpha1))
    return alpha1, alpha2


@register_source_route(
    ("PI", "r", "uniform"),
    ("PI", "r", "grid"),
    ("PI", "rho", "uniform"),
    ("PI", "rho", "grid"),
)
@njit(cache=True, nogil=True)
def _update_pi_from_r_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    return _update_pi_from_r_inputs_impl(
        out_root_fields,
        out_FFn_psin,
        out_Pn_psin,
        pprime_input,
        driver_input,
        driver_input,
        False,
        coordinate_code,
        R0,
        B0,
        weights,
        differentiator,
        accumulator,
        grid_radial_fields,
        n_axis_fix,
        radial_fields,
        surface_fields,
        F_fields,
        scaled_p0,
        Ip,
        beta,
        array_scratch,
        matrix_scratch,
    )


@njit(cache=True, nogil=True)
def _update_pi_from_r_explicit_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
    driver_derivative: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    """PI-r closure using the derivative of its retained PCHIP primitive."""
    return _update_pi_from_r_inputs_impl(
        out_root_fields,
        out_FFn_psin,
        out_Pn_psin,
        pprime_input,
        driver_input,
        driver_derivative,
        True,
        coordinate_code,
        R0,
        B0,
        weights,
        differentiator,
        accumulator,
        grid_radial_fields,
        n_axis_fix,
        radial_fields,
        surface_fields,
        F_fields,
        scaled_p0,
        Ip,
        beta,
        array_scratch,
        matrix_scratch,
    )


@register_source_route(("PI", "psin", "uniform"))
@njit(cache=True, nogil=True)
def _update_pi_from_psin_uniform_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    Itor = array_scratch[_SLOT_AUX0]
    if has_Ip:
        scale_into(Itor, driver_input, Ip / _edge_eval(driver_input, grid_radial_fields))
    else:
        copy_into(Itor, driver_input)
    _floor_signed_current_primitive(Itor, _edge_eval(Itor, grid_radial_fields))
    itor_over_kn = array_scratch[_SLOT_INTEGRAND]
    scaled_ratio_into(itor_over_kn, Itor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(itor_over_kn, weights)
    scaled_ratio_into(out_psin_r, Itor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    _regularize_psin_r(out_psin_r, r, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)
    Itor_r = array_scratch[_SLOT_AUX1]
    full_differentiation(Itor_r, Itor, differentiator)
    _regularize_axis_linear(Itor_r, r, n_axis_fix)
    if has_beta:
        copy_into(out_Pn_psin, pprime_input)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX2]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            * dot(V_r, weights)
            / _psin_beta_pressure_denominator(scratch_aux, V_r, weights, scaled_p0, alpha2)
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        scaled_product_into(scratch_Pr, pprime_input, out_psin_r, alpha2)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
        scaled_ratio_into(out_Pn_psin, scratch_Pr, out_psin_r, 1.0 / (alpha1 * alpha2))
    _fill_pi_ffn_psin(out_FFn_psin, Itor_r, V_r, out_Pn_psin, Ln_r, 1.0 / (2.0 * np.pi * alpha1))
    _regularize_ffn_psin(out_FFn_psin, r, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PI", "psin", "grid"))
@njit(cache=True, nogil=True)
def _update_pi_from_psin_grid_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    Itor = array_scratch[_SLOT_AUX0]
    if has_Ip:
        scale_into(Itor, driver_input, Ip / _edge_eval(driver_input, grid_radial_fields))
    else:
        copy_into(Itor, driver_input)
    _floor_signed_current_primitive(Itor, _edge_eval(Itor, grid_radial_fields))
    itor_over_kn = array_scratch[_SLOT_INTEGRAND]
    scaled_ratio_into(itor_over_kn, Itor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(itor_over_kn, weights)
    scaled_ratio_into(out_psin_r, Itor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    _regularize_psin_r(out_psin_r, r, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)
    Itor_r = array_scratch[_SLOT_AUX1]
    full_differentiation(Itor_r, Itor, differentiator)
    _regularize_axis_linear(Itor_r, r, n_axis_fix)
    if has_beta:
        copy_into(out_Pn_psin, pprime_input)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX2]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            * dot(V_r, weights)
            / _psin_beta_pressure_denominator(scratch_aux, V_r, weights, scaled_p0, alpha2)
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        scaled_product_into(scratch_Pr, pprime_input, out_psin_r, alpha2)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
        scaled_ratio_into(out_Pn_psin, scratch_Pr, out_psin_r, 1.0 / (alpha1 * alpha2))
    _fill_pi_ffn_psin(out_FFn_psin, Itor_r, V_r, out_Pn_psin, Ln_r, 1.0 / (2.0 * np.pi * alpha1))
    _regularize_ffn_psin(out_FFn_psin, r, n_axis_fix)
    return alpha1, alpha2


@register_source_route(
    ("PJ1", "r", "uniform"),
    ("PJ1", "r", "grid"),
    ("PJ1", "rho", "uniform"),
    ("PJ1", "rho", "grid"),
)
@njit(cache=True, nogil=True)
def _update_pj1_from_r_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand_j = array_scratch[_SLOT_INTEGRAND]
    product_into(integrand_j, driver_input, S_r)
    full_integration(out_psin_r, integrand_j, accumulator)
    I_tor_prof = array_scratch[_SLOT_AUX0]
    copy_into(I_tor_prof, out_psin_r)
    I_tor = array_scratch[_SLOT_AUX1]
    jtor = array_scratch[_SLOT_AUX2]
    current_edge = _full_integral(integrand_j, weights)
    source_scale = 1.0
    if has_Ip:
        # PJ1 integrates a current-density-like input into I_tor first; the same
        # Ip scale must be applied to the primitive and to the local jtor profile.
        source_scale = Ip / current_edge
        scale_into(I_tor, I_tor_prof, source_scale)
        scale_into(jtor, driver_input, source_scale)
    else:
        copy_into(I_tor, I_tor_prof)
        copy_into(jtor, driver_input)
    _floor_signed_current_primitive(I_tor, current_edge * source_scale)
    itor_over_kn = array_scratch[_SLOT_INTEGRAND]
    scaled_ratio_into(itor_over_kn, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(itor_over_kn, weights)
    scaled_ratio_into(out_psin_r, I_tor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    _regularize_psin_r(out_psin_r, r, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)
    if has_beta:
        scaled_ratio_into(out_Pn_psin, pprime_input, out_psin_r, 1.0)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_INTEGRAND]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            * dot(V_r, weights)
            / (alpha2 * _r_beta_pressure_denominator(scratch_aux, V_r, weights, scaled_p0))
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        copy_into(scratch_Pr, pprime_input)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
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
    return alpha1, alpha2


@register_source_route(("PJ1", "psin", "uniform"))
@njit(cache=True, nogil=True)
def _update_pj1_from_psin_uniform_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand_j = array_scratch[_SLOT_INTEGRAND]
    product_into(integrand_j, driver_input, S_r)
    full_integration(out_psin_r, integrand_j, accumulator)
    I_tor_prof = array_scratch[_SLOT_AUX0]
    copy_into(I_tor_prof, out_psin_r)
    I_tor = array_scratch[_SLOT_AUX1]
    jtor = array_scratch[_SLOT_AUX2]
    current_edge = _full_integral(integrand_j, weights)
    source_scale = 1.0
    if has_Ip:
        source_scale = Ip / current_edge
        scale_into(I_tor, I_tor_prof, source_scale)
        scale_into(jtor, driver_input, source_scale)
    else:
        copy_into(I_tor, I_tor_prof)
        copy_into(jtor, driver_input)
    _enforce_axis_even_profile(jtor, r)
    _floor_signed_current_primitive(I_tor, current_edge * source_scale)
    itor_over_kn = array_scratch[_SLOT_INTEGRAND]
    scaled_ratio_into(itor_over_kn, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(itor_over_kn, weights)
    scaled_ratio_into(out_psin_r, I_tor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    _regularize_psin_r(out_psin_r, r, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)
    if has_beta:
        copy_into(out_Pn_psin, pprime_input)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_INTEGRAND]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            * dot(V_r, weights)
            / _psin_beta_pressure_denominator(scratch_aux, V_r, weights, scaled_p0, alpha2)
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        scaled_product_into(scratch_Pr, pprime_input, out_psin_r, alpha2)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
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
    _regularize_ffn_psin(out_FFn_psin, r, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PJ1", "psin", "grid"))
@njit(cache=True, nogil=True)
def _update_pj1_from_psin_grid_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand_j = array_scratch[_SLOT_INTEGRAND]
    product_into(integrand_j, driver_input, S_r)
    full_integration(out_psin_r, integrand_j, accumulator)
    I_tor_prof = array_scratch[_SLOT_AUX0]
    copy_into(I_tor_prof, out_psin_r)
    I_tor = array_scratch[_SLOT_AUX1]
    jtor = array_scratch[_SLOT_AUX2]
    current_edge = _full_integral(integrand_j, weights)
    source_scale = 1.0
    if has_Ip:
        source_scale = Ip / current_edge
        scale_into(I_tor, I_tor_prof, source_scale)
        scale_into(jtor, driver_input, source_scale)
    else:
        copy_into(I_tor, I_tor_prof)
        copy_into(jtor, driver_input)
    _enforce_axis_even_profile(jtor, r)
    _floor_signed_current_primitive(I_tor, current_edge * source_scale)
    itor_over_kn = array_scratch[_SLOT_INTEGRAND]
    scaled_ratio_into(itor_over_kn, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(itor_over_kn, weights)
    scaled_ratio_into(out_psin_r, I_tor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    _regularize_psin_r(out_psin_r, r, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)
    if has_beta:
        copy_into(out_Pn_psin, pprime_input)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_INTEGRAND]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            * dot(V_r, weights)
            / _psin_beta_pressure_denominator(scratch_aux, V_r, weights, scaled_p0, alpha2)
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        scaled_product_into(scratch_Pr, pprime_input, out_psin_r, alpha2)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
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
    _regularize_ffn_psin(out_FFn_psin, r, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PJ2", "psin", "uniform"))
@njit(cache=True, nogil=True)
def _update_pj2_from_psin_uniform_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
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

    scaled_product_ratio_into(integrand, Ln_r, driver_input, F, 1.0)
    full_integration(out_psin_r, integrand, accumulator)
    copy_into(integral_val, out_psin_r)

    if has_Ip:
        # PJ2 couples the source current to the current F profile.  In psin
        # routes the edge normalization uses the physical edge F=R0*B0.
        scaled_product_into(
            I_tor,
            F,
            integral_val,
            Ip / (R0 * B0 * _full_integral(integrand, weights)),
        )
    else:
        scaled_product_into(I_tor, F, integral_val, 2.0 * np.pi)
    scaled_ratio_into(integrand, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(integrand, weights)
    scale_into(out_psin_r, integrand, 1.0 / alpha2)
    _regularize_psin_r(out_psin_r, r, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)

    if has_beta:
        product_into(scratch_Pn_r, pprime_input, out_psin_r)
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            * dot(V_r, weights)
            / _psin_beta_pressure_denominator(scratch_aux, V_r, weights, scaled_p0, alpha2)
        )
        copy_into(out_Pn_psin, pprime_input)
    else:
        alpha1 = -weighted_dot(pprime_input, out_psin_r, weights)
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
        scaled_product_ratio_into(out_Pn_psin, pprime_input, out_psin_r, out_psin_r, 1.0 / alpha1)

    product_into(out_FFn_psin, F, F_r)
    scaled_ratio_into(out_FFn_psin, out_FFn_psin, out_psin_r, 1.0 / (alpha1 * alpha2))
    _regularize_ffn_psin(out_FFn_psin, r, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PJ2", "psin", "grid"))
@njit(cache=True, nogil=True)
def _update_pj2_from_psin_grid_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
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

    scaled_product_ratio_into(integrand, Ln_r, driver_input, F, 1.0)
    full_integration(out_psin_r, integrand, accumulator)
    copy_into(integral_val, out_psin_r)

    if has_Ip:
        scaled_product_into(
            I_tor,
            F,
            integral_val,
            Ip / (R0 * B0 * _full_integral(integrand, weights)),
        )
    else:
        scaled_product_into(I_tor, F, integral_val, 2.0 * np.pi)
    scaled_ratio_into(integrand, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(integrand, weights)
    scale_into(out_psin_r, integrand, 1.0 / alpha2)
    _regularize_psin_r(out_psin_r, r, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)

    if has_beta:
        product_into(scratch_Pn_r, pprime_input, out_psin_r)
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            * dot(V_r, weights)
            / _psin_beta_pressure_denominator(scratch_aux, V_r, weights, scaled_p0, alpha2)
        )
        copy_into(out_Pn_psin, pprime_input)
    else:
        alpha1 = -weighted_dot(pprime_input, out_psin_r, weights)
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
        scaled_product_ratio_into(out_Pn_psin, pprime_input, out_psin_r, out_psin_r, 1.0 / alpha1)

    product_into(out_FFn_psin, F, F_r)
    scaled_ratio_into(out_FFn_psin, out_FFn_psin, out_psin_r, 1.0 / (alpha1 * alpha2))
    _regularize_ffn_psin(out_FFn_psin, r, n_axis_fix)
    return alpha1, alpha2


@register_source_route(
    ("PJ2", "r", "uniform"),
    ("PJ2", "r", "grid"),
    ("PJ2", "rho", "uniform"),
    ("PJ2", "rho", "grid"),
)
@njit(cache=True, nogil=True)
def _update_pj2_from_r_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
    V_r, Kn, Kn_r, Ln_r, S_r, R, JdivR = _source_geometry_workspace_views(
        radial_fields, surface_fields
    )
    F = F_fields[0]
    F_r = F_fields[1]
    has_Ip = not np.isnan(Ip)
    has_beta = not np.isnan(beta)
    integrand = array_scratch[_SLOT_INTEGRAND]
    scaled_product_ratio_into(integrand, Ln_r, driver_input, F, 1.0)
    full_integration(out_psin_r, integrand, accumulator)
    integral_val = array_scratch[_SLOT_AUX0]
    copy_into(integral_val, out_psin_r)
    I_tor = array_scratch[_SLOT_AUX1]
    if has_Ip:
        scaled_product_into(
            I_tor,
            F,
            integral_val,
            Ip / (_edge_eval(F, grid_radial_fields) * _full_integral(integrand, weights)),
        )
    else:
        scaled_product_into(I_tor, F, integral_val, 2.0 * np.pi)
    itor_over_kn = array_scratch[_SLOT_INTEGRAND]
    scaled_ratio_into(itor_over_kn, I_tor, Kn, 1.0 / (2.0 * np.pi))
    alpha2 = dot(itor_over_kn, weights)
    scaled_ratio_into(out_psin_r, I_tor, Kn, 1.0 / (2.0 * np.pi * alpha2))
    _regularize_psin_r(out_psin_r, r, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)
    if has_beta:
        scaled_ratio_into(out_Pn_psin, pprime_input, out_psin_r, 1.0)
        scratch_Pn_r = array_scratch[_SLOT_PNr]
        product_into(scratch_Pn_r, out_Pn_psin, out_psin_r)
        scratch_aux = array_scratch[_SLOT_AUX2]
        _compute_Pn_out(scratch_aux, scratch_Pn_r, accumulator, weights)
        alpha1 = (
            0.5
            * beta
            * B0**2
            * dot(V_r, weights)
            / (alpha2 * _r_beta_pressure_denominator(scratch_aux, V_r, weights, scaled_p0))
        )
    else:
        scratch_Pr = array_scratch[_SLOT_Pr]
        copy_into(scratch_Pr, pprime_input)
        alpha1 = -dot(scratch_Pr, weights) / alpha2
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
        scaled_ratio_into(out_Pn_psin, scratch_Pr, out_psin_r, 1.0 / (alpha1 * alpha2))
    scaled_product_into(out_FFn_psin, F, F_r, 1.0 / (alpha1 * alpha2))
    scaled_ratio_into(out_FFn_psin, out_FFn_psin, out_psin_r, 1.0)
    return alpha1, alpha2


@njit(cache=True, nogil=True)
def _fill_pj23_strict_rhs(
    ru0: np.ndarray,
    ru1: np.ndarray,
    rc0: np.ndarray,
    rc1: np.ndarray,
    F: np.ndarray,
    u: np.ndarray,
    C: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
    pressure_multiplier: float,
    B0: float,
    F_edge: float,
    Kn: np.ndarray,
    Ln_r: np.ndarray,
    V_r: np.ndarray,
    use_jtotal_semantics: bool,
) -> None:
    """Build the affine-in-current-scale strict PJ2/PJ3 radial RHS."""

    four_pi2 = 4.0 * np.pi * np.pi
    for i in range(u.shape[0]):
        F_i = F_edge * np.exp(0.5 * u[i])
        F[i] = F_i
        x_i = C[i] / Kn[i]
        g1_i = four_pi2 * Ln_r[i] / V_r[i]
        H_i = four_pi2 * Kn[i] / V_r[i]
        g5_i = F_i * F_i * g1_i + x_i * x_i * H_i
        if not np.isfinite(g5_i) or g5_i <= 1.0e-20:
            raise ValueError("strict PJ2/PJ3 closure produced invalid g5")

        ru0_i = -2.0 * pressure_multiplier * pprime_input[i] / g5_i
        if use_jtotal_semantics:
            if not np.isfinite(F_i) or abs(F_i) <= 1.0e-14:
                raise ValueError("strict PJ3 closure produced invalid F")
            ru1_i = -2.0 * x_i * B0 * driver_input[i] / (F_i * g5_i)
            rc1_i = Ln_r[i] * B0 * driver_input[i] / (F_i * g1_i)
        else:
            ru1_i = -2.0 * x_i * g1_i * driver_input[i] / g5_i
            rc1_i = Ln_r[i] * driver_input[i]

        ru0[i] = ru0_i
        ru1[i] = ru1_i
        rc0[i] = 0.5 * C[i] * ru0_i
        rc1[i] = rc1_i + 0.5 * C[i] * ru1_i


@njit(cache=True, nogil=True)
def _strict_current_scale_and_combine_rhs(
    ru0: np.ndarray,
    ru1: np.ndarray,
    rc0: np.ndarray,
    rc1: np.ndarray,
    weights: np.ndarray,
    scaled_Ip: float,
) -> float:
    current_multiplier = 1.0
    if not np.isnan(scaled_Ip):
        edge0 = dot(rc0, weights)
        edge1 = dot(rc1, weights)
        if not np.isfinite(edge1) or abs(edge1) <= 1.0e-20:
            raise ValueError("strict PJ2/PJ3 closure cannot enforce Ip: zero current response")
        current_multiplier = (scaled_Ip / (2.0 * np.pi) - edge0) / edge1
        if not np.isfinite(current_multiplier):
            raise ValueError("strict PJ2/PJ3 closure produced invalid current multiplier")
    for i in range(ru0.shape[0]):
        ru0[i] += current_multiplier * ru1[i]
        rc0[i] += current_multiplier * rc1[i]
    return current_multiplier


@njit(cache=True, nogil=True)
def _strict_fixed_point_map(
    out_u: np.ndarray,
    out_C: np.ndarray,
    ru0: np.ndarray,
    ru1: np.ndarray,
    rc0: np.ndarray,
    rc1: np.ndarray,
    F: np.ndarray,
    u: np.ndarray,
    C: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
    pressure_multiplier: float,
    B0: float,
    F_edge: float,
    Kn: np.ndarray,
    Ln_r: np.ndarray,
    V_r: np.ndarray,
    weights: np.ndarray,
    accumulator: np.ndarray,
    scaled_Ip: float,
    use_jtotal_semantics: bool,
) -> float:
    _fill_pj23_strict_rhs(
        ru0,
        ru1,
        rc0,
        rc1,
        F,
        u,
        C,
        pprime_input,
        driver_input,
        pressure_multiplier,
        B0,
        F_edge,
        Kn,
        Ln_r,
        V_r,
        use_jtotal_semantics,
    )
    current_multiplier = _strict_current_scale_and_combine_rhs(
        ru0,
        ru1,
        rc0,
        rc1,
        weights,
        scaled_Ip,
    )
    full_integration(out_C, rc0, accumulator)
    full_integration(out_u, ru0, accumulator)
    edge_u = dot(ru0, weights)
    for i in range(out_u.shape[0]):
        out_u[i] -= edge_u
    return current_multiplier


@njit(cache=True, nogil=True)
def _strict_fixed_point_defect(
    u: np.ndarray,
    C: np.ndarray,
    mapped_u: np.ndarray,
    mapped_C: np.ndarray,
    scaled_Ip: float,
) -> float:
    u_scale = 1.0
    C_scale = 1.0e-14
    if not np.isnan(scaled_Ip):
        C_scale = max(C_scale, abs(scaled_Ip / (2.0 * np.pi)))
    for i in range(u.shape[0]):
        u_scale = max(u_scale, abs(u[i]))
        C_scale = max(C_scale, abs(C[i]))
    defect = 0.0
    for i in range(u.shape[0]):
        defect = max(defect, abs(mapped_u[i] - u[i]) / u_scale)
        defect = max(defect, abs(mapped_C[i] - C[i]) / C_scale)
    return defect


@njit(cache=True, nogil=True)
def _pj23_joint_pressure_multiplier(
    pprime_input: np.ndarray,
    beta: float,
    B0: float,
    scaled_p0: float,
    V_r: np.ndarray,
    accumulator: np.ndarray,
    weights: np.ndarray,
    relative_pressure: np.ndarray,
) -> float:
    """Return the algebraic pressure multiplier for one joint PJ2/PJ3 map."""
    if np.isnan(beta):
        return 1.0
    _compute_Pn_out(relative_pressure, pprime_input, accumulator, weights)
    denominator = _r_beta_pressure_denominator(
        relative_pressure,
        V_r,
        weights,
        scaled_p0,
    )
    if not np.isfinite(denominator) or abs(denominator) <= 1.0e-20:
        raise ValueError("joint PJ2/PJ3 beta closure received zero pressure response")
    return 0.5 * beta * B0 * B0 * dot(V_r, weights) / denominator


@njit(cache=True, nogil=True)
def _pj23_joint_fixed_point_map_with_scratch(
    out_u: np.ndarray,
    out_C: np.ndarray,
    F: np.ndarray,
    u: np.ndarray,
    C: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
    pressure_multiplier: float,
    B0: float,
    F_edge: float,
    Kn: np.ndarray,
    Ln_r: np.ndarray,
    V_r: np.ndarray,
    weights: np.ndarray,
    accumulator: np.ndarray,
    scaled_Ip: float,
    use_jtotal_semantics: bool,
    array_scratch: np.ndarray,
) -> float:
    """Apply one strict physics map and return its dimensionless defect."""
    ru0 = array_scratch[_SLOT_PNr]
    ru1 = array_scratch[_SLOT_Pr]
    rc0 = array_scratch[_SLOT_Fr]
    rc1 = array_scratch[_SLOT_EFFECTIVE_DRIVER]
    _strict_fixed_point_map(
        out_u,
        out_C,
        ru0,
        ru1,
        rc0,
        rc1,
        F,
        u,
        C,
        pprime_input,
        driver_input,
        pressure_multiplier,
        B0,
        F_edge,
        Kn,
        Ln_r,
        V_r,
        weights,
        accumulator,
        scaled_Ip,
        use_jtotal_semantics,
    )
    return _strict_fixed_point_defect(u, C, out_u, out_C, scaled_Ip)


@njit(cache=True, nogil=True)
def _publish_pj23_joint_state(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    F_fields: np.ndarray,
    u: np.ndarray,
    C: np.ndarray,
    pprime_input: np.ndarray,
    pressure_multiplier: float,
    scaled_p0: float,
    beta: float,
    R0: float,
    B0: float,
    Kn: np.ndarray,
    weights: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    array_scratch: np.ndarray,
    pressure_state: np.ndarray,
) -> tuple[float, float]:
    """Publish one accepted joint ``(u, C)`` state and normalize pressure."""
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    F = F_fields[0]
    F_r = F_fields[1]
    F_rr = F_fields[2]
    ru0 = array_scratch[_SLOT_PNr]
    F_edge = R0 * B0
    for i in range(F.shape[0]):
        F[i] = F_edge * np.exp(0.5 * u[i])
        F_r[i] = 0.5 * F[i] * ru0[i]
        out_psin_r[i] = C[i] / Kn[i]
    alpha2 = dot(out_psin_r, weights)
    if not np.isfinite(alpha2) or abs(alpha2) <= 1.0e-14:
        raise ValueError("joint PJ2/PJ3 closure produced invalid flux scale")
    out_psin_r /= alpha2
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)
    full_differentiation(F_rr, F_r, differentiator)

    if not np.isnan(beta):
        alpha1 = pressure_multiplier / alpha2
    else:
        alpha1 = -dot(pprime_input, weights) / alpha2
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            R_COORDINATE,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
    for i in range(out_Pn_psin.shape[0]):
        x_i = alpha2 * out_psin_r[i]
        denominator = alpha1 * x_i
        if not np.isfinite(denominator) or abs(denominator) <= 1.0e-20:
            raise ValueError("joint PJ2/PJ3 closure produced invalid source normalization")
        out_Pn_psin[i] = pressure_multiplier * pprime_input[i] / denominator
        out_FFn_psin[i] = F[i] * F_r[i] / denominator
    alpha1 = finalize_pressure_normalization(
        out_FFn_psin,
        out_Pn_psin,
        pprime_input,
        R_COORDINATE,
        scaled_p0,
        beta,
        alpha1,
        alpha2,
        out_psin_r,
        accumulator,
        weights,
        array_scratch[0],
        array_scratch[1],
        pressure_state,
    )
    return alpha1, alpha2


@njit(cache=True, nogil=True)
def _update_pj23_strict_from_r_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
    use_jtotal_semantics: bool,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    V_r, Kn, _, Ln_r, _, _, _ = _source_geometry_workspace_views(radial_fields, surface_fields)

    # beta scales the complete r-coordinate pressure profile by one scalar;
    # unlike the flux/current closure, this multiplier is algebraic and does
    # not depend on the Picard state.
    pressure_multiplier = 1.0
    if not np.isnan(beta):
        relative_pressure = array_scratch[_SLOT_AUX2]
        _compute_Pn_out(relative_pressure, pprime_input, accumulator, weights)
        pressure_denominator = _r_beta_pressure_denominator(
            relative_pressure,
            V_r,
            weights,
            scaled_p0,
        )
        if not np.isfinite(pressure_denominator) or abs(pressure_denominator) <= 1.0e-20:
            raise ValueError("strict PJ2/PJ3 beta closure received zero pressure response")
        pressure_multiplier = 0.5 * beta * B0 * B0 * dot(V_r, weights) / pressure_denominator

    u = array_scratch[_SLOT_INTEGRAND]
    C = array_scratch[_SLOT_AUX0]
    mapped_u = array_scratch[_SLOT_AUX1]
    mapped_C = array_scratch[_SLOT_AUX2]
    ru0 = array_scratch[_SLOT_PNr]
    ru1 = array_scratch[_SLOT_Pr]
    rc0 = array_scratch[_SLOT_Fr]
    rc1 = array_scratch[_SLOT_EFFECTIVE_DRIVER]
    u.fill(0.0)
    C.fill(0.0)
    F = F_fields[0]
    F.fill(R0 * B0)

    defect = np.inf
    for _ in range(PJ23_STRICT_FIXED_POINT_MAX_ITER):
        _strict_fixed_point_map(
            mapped_u,
            mapped_C,
            ru0,
            ru1,
            rc0,
            rc1,
            F,
            u,
            C,
            pprime_input,
            driver_input,
            pressure_multiplier,
            B0,
            R0 * B0,
            Kn,
            Ln_r,
            V_r,
            weights,
            accumulator,
            Ip,
            use_jtotal_semantics,
        )
        defect = _strict_fixed_point_defect(u, C, mapped_u, mapped_C, Ip)
        if np.isfinite(defect) and defect <= PJ23_STRICT_FIXED_POINT_MAX_RESIDUAL:
            break
        copy_into(u, mapped_u)
        copy_into(C, mapped_C)

    if not np.isfinite(defect) or defect > PJ23_STRICT_FIXED_POINT_MAX_RESIDUAL:
        raise ValueError(
            "strict PJ2/PJ3 fixed-point closure did not reach 1e-6 within 10 iterations"
        )

    # The accepted non-mutating map has refreshed F and ru0 at the published
    # (u, C) state; mapped_u/mapped_C differ from it by at most the gate above.
    F_r = F_fields[1]
    F_rr = F_fields[2]
    for i in range(F.shape[0]):
        F_r[i] = 0.5 * F[i] * ru0[i]
        out_psin_r[i] = C[i] / Kn[i]
    alpha2 = dot(out_psin_r, weights)
    if not np.isfinite(alpha2) or abs(alpha2) <= 1.0e-14:
        raise ValueError("strict PJ2/PJ3 closure produced invalid flux scale")
    out_psin_r /= alpha2
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)
    full_differentiation(F_rr, F_r, differentiator)

    if not np.isnan(beta):
        alpha1 = pressure_multiplier / alpha2
    else:
        alpha1 = -dot(pprime_input, weights) / alpha2
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            R_COORDINATE,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
    for i in range(out_Pn_psin.shape[0]):
        x_i = alpha2 * out_psin_r[i]
        denominator = alpha1 * x_i
        if not np.isfinite(denominator) or abs(denominator) <= 1.0e-20:
            raise ValueError("strict PJ2/PJ3 closure produced invalid source normalization")
        out_Pn_psin[i] = pressure_multiplier * pprime_input[i] / denominator
        out_FFn_psin[i] = F[i] * F_r[i] / denominator
    return alpha1, alpha2


@njit(cache=True, nogil=True)
def _update_pj2_strict_from_r_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    return _update_pj23_strict_from_r_inputs_with_scratch(
        out_root_fields,
        out_FFn_psin,
        out_Pn_psin,
        pprime_input,
        driver_input,
        coordinate_code,
        R0,
        B0,
        weights,
        differentiator,
        accumulator,
        grid_radial_fields,
        n_axis_fix,
        radial_fields,
        surface_fields,
        F_fields,
        scaled_p0,
        Ip,
        beta,
        array_scratch,
        matrix_scratch,
        False,
    )


@njit(cache=True, nogil=True)
def _update_pj3_strict_from_r_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    return _update_pj23_strict_from_r_inputs_with_scratch(
        out_root_fields,
        out_FFn_psin,
        out_Pn_psin,
        pprime_input,
        driver_input,
        coordinate_code,
        R0,
        B0,
        weights,
        differentiator,
        accumulator,
        grid_radial_fields,
        n_axis_fix,
        radial_fields,
        surface_fields,
        F_fields,
        scaled_p0,
        Ip,
        beta,
        array_scratch,
        matrix_scratch,
        True,
    )


@njit(cache=True, nogil=True)
def _materialize_pj3_effective_jpara(
    out: np.ndarray,
    jtotal_input: np.ndarray,
    B0: float,
    V_r: np.ndarray,
    Ln_r: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    """Convert IMAS ``<J·B>/B0`` to PJ2 current using the current geometry/F.

    The conversion is deliberately performed inside every source evaluation:
    ``gm1=<R^-2>=(2*pi)^2*Ln_r/V_r`` and
    ``jpara=B0*jtotal/(F*gm1)``.  PJ3 therefore remains coupled to the active
    F profile instead of freezing a setup-time PJ2 approximation.
    """

    gm1_scale = (2.0 * np.pi) ** 2
    for i in range(out.shape[0]):
        denominator = gm1_scale * F[i] * Ln_r[i]
        if not np.isfinite(denominator) or abs(denominator) <= 1.0e-14:
            raise ValueError("PJ3 source received invalid F*Ln_r geometry factor")
        out[i] = B0 * jtotal_input[i] * V_r[i] / denominator
        if not np.isfinite(out[i]):
            raise ValueError("PJ3 source produced non-finite effective jpara")
    return out


@register_source_route(("PJ3", "psin", "uniform"))
@njit(cache=True, nogil=True)
def _update_pj3_from_psin_uniform_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    V_r, _, _, Ln_r, _, _, _ = _source_geometry_workspace_views(radial_fields, surface_fields)
    effective_jpara = array_scratch[_SLOT_EFFECTIVE_DRIVER]
    _materialize_pj3_effective_jpara(effective_jpara, driver_input, B0, V_r, Ln_r, F_fields[0])
    return _update_pj2_from_psin_uniform_inputs_with_scratch(
        out_root_fields,
        out_FFn_psin,
        out_Pn_psin,
        pprime_input,
        effective_jpara,
        coordinate_code,
        R0,
        B0,
        weights,
        differentiator,
        accumulator,
        grid_radial_fields,
        n_axis_fix,
        radial_fields,
        surface_fields,
        F_fields,
        scaled_p0,
        Ip,
        beta,
        array_scratch,
        matrix_scratch,
    )


@register_source_route(("PJ3", "psin", "grid"))
@njit(cache=True, nogil=True)
def _update_pj3_from_psin_grid_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    V_r, _, _, Ln_r, _, _, _ = _source_geometry_workspace_views(radial_fields, surface_fields)
    effective_jpara = array_scratch[_SLOT_EFFECTIVE_DRIVER]
    _materialize_pj3_effective_jpara(effective_jpara, driver_input, B0, V_r, Ln_r, F_fields[0])
    return _update_pj2_from_psin_grid_inputs_with_scratch(
        out_root_fields,
        out_FFn_psin,
        out_Pn_psin,
        pprime_input,
        effective_jpara,
        coordinate_code,
        R0,
        B0,
        weights,
        differentiator,
        accumulator,
        grid_radial_fields,
        n_axis_fix,
        radial_fields,
        surface_fields,
        F_fields,
        scaled_p0,
        Ip,
        beta,
        array_scratch,
        matrix_scratch,
    )


@register_source_route(
    ("PJ3", "r", "uniform"),
    ("PJ3", "r", "grid"),
    ("PJ3", "rho", "uniform"),
    ("PJ3", "rho", "grid"),
)
@njit(cache=True, nogil=True)
def _update_pj3_from_r_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    V_r, _, _, Ln_r, _, _, _ = _source_geometry_workspace_views(radial_fields, surface_fields)
    effective_jpara = array_scratch[_SLOT_EFFECTIVE_DRIVER]
    _materialize_pj3_effective_jpara(effective_jpara, driver_input, B0, V_r, Ln_r, F_fields[0])
    return _update_pj2_from_r_inputs_with_scratch(
        out_root_fields,
        out_FFn_psin,
        out_Pn_psin,
        pprime_input,
        effective_jpara,
        coordinate_code,
        R0,
        B0,
        weights,
        differentiator,
        accumulator,
        grid_radial_fields,
        n_axis_fix,
        radial_fields,
        surface_fields,
        F_fields,
        scaled_p0,
        Ip,
        beta,
        array_scratch,
        matrix_scratch,
    )


@register_source_route(("PQ", "psin", "uniform"))
@njit(cache=True, nogil=True)
def _update_pq_from_psin_uniform_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
    edge_weights = _source_grid_edge_weights(grid_radial_fields)
    V_r, Kn, _, Ln_r, _, _, _ = _source_geometry_workspace_views(radial_fields, surface_fields)
    n = r.shape[0]
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

    _fill_pq_q_profile(q_prof, driver_input, Kn, Ln_r, edge_weights, edge_F, Ip)
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
            rhs[i] = 0.0
            W[i] = -pressure_factor * V_r[i] * pprime_input[i]
            if not np.isfinite(W[i]):
                raise ValueError("PQ/psin strict beta solve assembled non-finite pressure RHS")
        _fill_pq_linear_matrix_two_rhs(
            A,
            F_solved,
            F_r,
            accumulator,
            weights,
            coeff_d,
            coeff_y,
            rhs,
            W,
            edge_F,
            0.0,
            n,
        )
        _dense_solve_two_rhs_inplace(A, F_solved, F_r, n, 1.0e-12)

        beta_target = 0.5 * beta * B0**2 * dot(V_r, weights)
        alpha1 = _solve_pq_psin_beta_alpha1(
            F_solved,
            F_r,
            q_prof,
            Ln_r,
            pprime_input,
            V_r,
            weights,
            accumulator,
            out_psin_r,
            rhs,
            W,
            scaled_p0,
            beta_target,
        )
        for i in range(n):
            F_solved[i] = F_solved[i] + alpha1 * F_r[i]
        copy_into(out_Pn_psin, pprime_input)
    else:
        for i in range(n):
            rhs[i] = -pressure_factor * V_r[i] * pprime_input[i]
            if not np.isfinite(rhs[i]):
                raise ValueError("PQ/psin strict solve assembled non-finite pressure RHS")
        _fill_pq_linear_matrix(A, F_solved, accumulator, weights, coeff_d, coeff_y, rhs, edge_F, n)
        _dense_solve_one_rhs_inplace(A, F_solved, n, 1.0e-12)
        alpha1 = 0.0

    for i in range(n):
        out_psin_r[i] = F_solved[i] * Ln_r[i] / q_prof[i]
        if not np.isfinite(out_psin_r[i]):
            raise ValueError("PQ/psin strict solve produced invalid psi_r")

    # F*Ln/q is signed physical psi_r.  Keep that sign in alpha2, then normalize
    # the solver psin_r branch back to a positive [0, 1] coordinate.
    alpha2 = _normalize_pq_signed_psi_r(out_psin_r, weights, r, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)

    if not has_beta:
        alpha1 = -weighted_dot(pprime_input, out_psin_r, weights)
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
        for i in range(n):
            out_Pn_psin[i] = pprime_input[i] / alpha1
    _validate_pq_source_scalar(alpha1, 1)

    forcing_scale = alpha1 if has_beta else 1.0
    for i in range(n):
        forcing_i = -pressure_factor * forcing_scale * V_r[i] * pprime_input[i]
        F_r[i] = (forcing_i - coeff_y[i] * F_solved[i]) / coeff_d[i]

    for i in range(n):
        if abs(Ln_r[i]) <= 1.0e-14:
            raise ValueError("PQ/psin strict solve received invalid Ln_r")
        out_FFn_psin[i] = (q_prof[i] * F_r[i] / Ln_r[i]) / alpha1
        if not np.isfinite(out_FFn_psin[i]) or not np.isfinite(out_Pn_psin[i]):
            raise ValueError("PQ/psin strict solve produced non-finite normalized source")
    _regularize_ffn_psin(out_FFn_psin, r, n_axis_fix)
    return alpha1, alpha2


@register_source_route(("PQ", "psin", "grid"))
@njit(cache=True, nogil=True)
def _update_pq_from_psin_grid_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
    edge_weights = _source_grid_edge_weights(grid_radial_fields)
    V_r, Kn, _, Ln_r, _, _, _ = _source_geometry_workspace_views(radial_fields, surface_fields)
    n = r.shape[0]
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

    _fill_pq_q_profile(q_prof, driver_input, Kn, Ln_r, edge_weights, edge_F, Ip)
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
            rhs[i] = 0.0
            W[i] = -pressure_factor * V_r[i] * pprime_input[i]
            if not np.isfinite(W[i]):
                raise ValueError("PQ/psin strict beta solve assembled non-finite pressure RHS")
        _fill_pq_linear_matrix_two_rhs(
            A,
            F_solved,
            F_r,
            accumulator,
            weights,
            coeff_d,
            coeff_y,
            rhs,
            W,
            edge_F,
            0.0,
            n,
        )
        _dense_solve_two_rhs_inplace(A, F_solved, F_r, n, 1.0e-12)

        beta_target = 0.5 * beta * B0**2 * dot(V_r, weights)
        alpha1 = _solve_pq_psin_beta_alpha1(
            F_solved,
            F_r,
            q_prof,
            Ln_r,
            pprime_input,
            V_r,
            weights,
            accumulator,
            out_psin_r,
            rhs,
            W,
            scaled_p0,
            beta_target,
        )
        for i in range(n):
            F_solved[i] = F_solved[i] + alpha1 * F_r[i]
        copy_into(out_Pn_psin, pprime_input)
    else:
        for i in range(n):
            rhs[i] = -pressure_factor * V_r[i] * pprime_input[i]
            if not np.isfinite(rhs[i]):
                raise ValueError("PQ/psin strict solve assembled non-finite pressure RHS")
        _fill_pq_linear_matrix(A, F_solved, accumulator, weights, coeff_d, coeff_y, rhs, edge_F, n)
        _dense_solve_one_rhs_inplace(A, F_solved, n, 1.0e-12)
        alpha1 = 0.0

    for i in range(n):
        out_psin_r[i] = F_solved[i] * Ln_r[i] / q_prof[i]
        if not np.isfinite(out_psin_r[i]):
            raise ValueError("PQ/psin strict solve produced invalid psi_r")

    alpha2 = _normalize_pq_signed_psi_r(out_psin_r, weights, r, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)

    if not has_beta:
        alpha1 = -weighted_dot(pprime_input, out_psin_r, weights)
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
        for i in range(n):
            out_Pn_psin[i] = pprime_input[i] / alpha1
    _validate_pq_source_scalar(alpha1, 1)

    forcing_scale = alpha1 if has_beta else 1.0
    for i in range(n):
        forcing_i = -pressure_factor * forcing_scale * V_r[i] * pprime_input[i]
        F_r[i] = (forcing_i - coeff_y[i] * F_solved[i]) / coeff_d[i]

    for i in range(n):
        if abs(Ln_r[i]) <= 1.0e-14:
            raise ValueError("PQ/psin strict solve received invalid Ln_r")
        out_FFn_psin[i] = (q_prof[i] * F_r[i] / Ln_r[i]) / alpha1
        if not np.isfinite(out_FFn_psin[i]) or not np.isfinite(out_Pn_psin[i]):
            raise ValueError("PQ/psin strict solve produced non-finite normalized source")
    _regularize_ffn_psin(out_FFn_psin, r, n_axis_fix)
    return alpha1, alpha2


@register_source_route(
    ("PQ", "r", "uniform"),
    ("PQ", "r", "grid"),
    ("PQ", "rho", "uniform"),
    ("PQ", "rho", "grid"),
)
@njit(cache=True, nogil=True)
def _update_pq_from_r_inputs_with_scratch(
    out_root_fields: np.ndarray,
    out_FFn_psin: np.ndarray,
    out_Pn_psin: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
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
    scaled_p0: float,
    Ip: float,
    beta: float,
    array_scratch: np.ndarray,
    matrix_scratch: np.ndarray,
) -> tuple[float, float]:
    out_psin, out_psin_r, out_psin_rr = _source_output_root_views(out_root_fields)
    r = _source_grid_r(grid_radial_fields)
    edge_weights = _source_grid_edge_weights(grid_radial_fields)
    V_r, Kn, _, Ln_r, _, _, _ = _source_geometry_workspace_views(radial_fields, surface_fields)
    n = r.shape[0]
    edge_F = R0 * B0
    if not np.isfinite(edge_F) or abs(edge_F) <= 1.0e-14:
        raise ValueError("PQ/r strict solve received invalid edge F")

    W = array_scratch[_SLOT_INTEGRAND]
    q_prof = array_scratch[_SLOT_AUX0]
    coeff_d = array_scratch[_SLOT_AUX1]
    coeff_y = array_scratch[_SLOT_AUX2]
    rhs = array_scratch[_SLOT_PNr]
    Y = array_scratch[_SLOT_Pr]
    Y_r = array_scratch[_SLOT_Fr]
    A = array_scratch[_SLOT_PQ_MATRIX : _SLOT_PQ_MATRIX + n, :]

    _fill_pq_q_profile(q_prof, driver_input, Kn, Ln_r, edge_weights, edge_F, Ip)
    _fill_pq_W_and_derivative(W, Y_r, Kn, Ln_r, q_prof, differentiator)

    # In r-coordinate PQ, solving for Y=F**2 keeps the strict edge condition
    # sign-safe; F is recovered only after the dense system succeeds.
    has_beta = not np.isnan(beta)
    pressure_scale = 1.0
    beta_C = 0.0
    if has_beta:
        copy_into(rhs, pprime_input)
        _compute_Pn_out(coeff_y, rhs, accumulator, weights)
        beta_den_pre = _r_beta_pressure_denominator(coeff_y, V_r, weights, scaled_p0)
        if not np.isfinite(beta_den_pre) or abs(beta_den_pre) <= 1.0e-14:
            raise ValueError("PQ/r strict beta solve produced invalid pressure integral")
        beta_C = 0.5 * beta * B0**2 * dot(V_r, weights) / beta_den_pre
        pressure_scale = beta_C

    pressure_factor = 1.0 / (2.0 * np.pi**2)
    for i in range(n):
        coeff_d[i] = W[i] + q_prof[i]
        coeff_y[i] = 2.0 * Y_r[i]
        rhs[i] = -pressure_factor * pressure_scale * V_r[i] * pprime_input[i] * q_prof[i] / Ln_r[i]
        if not np.isfinite(coeff_d[i]) or not np.isfinite(coeff_y[i]) or not np.isfinite(rhs[i]):
            raise ValueError("PQ/r strict solve assembled non-finite system")

    _fill_pq_linear_matrix(
        A,
        Y,
        accumulator,
        weights,
        coeff_d,
        coeff_y,
        rhs,
        edge_F * edge_F,
        n,
    )
    _dense_solve_one_rhs_inplace(A, Y, n, 1.0e-12)

    sign_F = 1.0
    if edge_F < 0.0:
        sign_F = -1.0
    for i in range(n):
        if not np.isfinite(Y[i]) or Y[i] <= 0.0:
            raise ValueError("PQ/r strict solve produced non-positive F squared")
        F_i = sign_F * np.sqrt(Y[i])
        out_psin_r[i] = F_i * Ln_r[i] / q_prof[i]
        if not np.isfinite(out_psin_r[i]):
            raise ValueError("PQ/r strict solve produced invalid psi_r")

    alpha2 = _normalize_pq_signed_psi_r(out_psin_r, weights, r, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)

    if has_beta:
        scaled_ratio_into(out_Pn_psin, pprime_input, out_psin_r, 1.0)
        alpha1 = beta_C / alpha2
    else:
        alpha1 = -dot(pprime_input, weights) / alpha2
        alpha1 = _ensure_pressure_alpha1(
            alpha1,
            pprime_input,
            coordinate_code,
            scaled_p0,
            alpha2,
            out_psin_r,
            accumulator,
            weights,
            array_scratch,
        )
        for i in range(n):
            denom = alpha1 * alpha2 * out_psin_r[i]
            if abs(denom) <= 1.0e-14:
                raise ValueError("PQ/r strict solve produced invalid pressure denominator")
            out_Pn_psin[i] = pprime_input[i] / denom
    _validate_pq_source_scalar(alpha1, 1)

    for i in range(n):
        forcing_i = (
            -pressure_factor * pressure_scale * V_r[i] * pprime_input[i] * q_prof[i] / Ln_r[i]
        )
        Y_r[i] = (forcing_i - coeff_y[i] * Y[i]) / coeff_d[i]
    _regularize_axis_linear(Y_r, r, n_axis_fix)
    for i in range(n):
        denom = alpha1 * alpha2 * out_psin_r[i]
        if abs(denom) <= 1.0e-14:
            raise ValueError("PQ/r strict solve produced invalid FFn denominator")
        out_FFn_psin[i] = 0.5 * Y_r[i] / denom
        if not np.isfinite(out_FFn_psin[i]) or not np.isfinite(out_Pn_psin[i]):
            raise ValueError("PQ/r strict solve produced non-finite normalized source")
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
def _update_fixed_point_psin_query_and_spline_uniform_inputs_impl(
    query: np.ndarray,
    psin: np.ndarray,
    max_residual: float,
    out_pprime_input: np.ndarray,
    out_driver_input: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
    pprime_spline_coeff: np.ndarray,
    driver_spline_coeff: np.ndarray,
) -> bool:
    max_abs_diff = 0.0
    for i in range(query.shape[0]):
        q = psin[i]
        diff = abs(q - query[i])
        if diff > max_abs_diff:
            max_abs_diff = diff
        query[i] = q

    _uniform_spline_interpolate_pair(
        out_pprime_input,
        out_driver_input,
        pprime_spline_coeff,
        driver_spline_coeff,
        query,
    )
    return max_abs_diff <= max_residual


@njit(cache=True, nogil=True)
def _update_fixed_point_psin_query_and_local_barycentric_inputs_impl(
    query: np.ndarray,
    psin: np.ndarray,
    max_residual: float,
    out_pprime_input: np.ndarray,
    out_driver_input: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
    weights: np.ndarray,
) -> bool:
    max_abs_diff = 0.0
    source_sample_count = pprime_input.shape[0]
    if source_sample_count == 1:
        pprime0 = pprime_input[0]
        driver0 = driver_input[0]
        for i in range(query.shape[0]):
            q = psin[i]
            diff = abs(q - query[i])
            if diff > max_abs_diff:
                max_abs_diff = diff
            query[i] = q
            out_pprime_input[i] = pprime0
            out_driver_input[i] = driver0
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
            out_pprime_input[i] = pprime_input[hit]
            out_driver_input[i] = driver_input[hit]
            continue

        denominator = 0.0
        numerator_pprime = 0.0
        numerator_driver = 0.0
        for local_j in range(local_size):
            j = start + local_j
            term = weights[local_j] / (q - j / denom_scale)
            denominator += term
            numerator_pprime += term * pprime_input[j]
            numerator_driver += term * driver_input[j]
        out_pprime_input[i] = numerator_pprime / denominator
        out_driver_input[i] = numerator_driver / denominator
    return max_abs_diff <= max_residual


@njit(cache=True, nogil=True)
def _materialize_profile_owned_psin_fields_impl(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_source_psin_query: np.ndarray,
    psin_fields: np.ndarray,
    grid_radial_fields: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    weights: np.ndarray,
    n_axis_fix: int,
) -> None:
    r = _source_grid_r(grid_radial_fields)
    # Copy only psin_r from optimized profile fields; psin and psin_rr are
    # reconstructed so all source paths share the same axis regularization and
    # integration conventions.
    for i in range(out_psin.shape[0]):
        out_psin_r[i] = psin_fields[1, i]

    _regularize_psin_r(out_psin_r, r, n_axis_fix)
    full_differentiation(out_psin_rr, out_psin_r, differentiator)
    _update_psin_coordinate(out_psin, out_psin_r, accumulator, weights)

    for i in range(out_psin.shape[0]):
        psin_value = out_psin[i]
        out_source_psin_query[i] = psin_value


@njit(cache=True, nogil=True)
def _materialize_profile_owned_psin_source_impl(
    out_psin: np.ndarray,
    out_psin_r: np.ndarray,
    out_psin_rr: np.ndarray,
    out_source_psin_query: np.ndarray,
    out_parameter_query: np.ndarray,
    out_pprime_input: np.ndarray,
    out_driver_input: np.ndarray,
    psin_fields: np.ndarray,
    pprime_input: np.ndarray,
    driver_input: np.ndarray,
    pprime_spline_coeff: np.ndarray,
    driver_spline_coeff: np.ndarray,
    parameterization_code: int,
    grid_radial_fields: np.ndarray,
    differentiator: np.ndarray,
    accumulator: np.ndarray,
    weights: np.ndarray,
    n_axis_fix: int,
    barycentric_weights: np.ndarray,
    use_barycentric: bool,
) -> None:
    _materialize_profile_owned_psin_fields_impl(
        out_psin,
        out_psin_r,
        out_psin_rr,
        out_source_psin_query,
        psin_fields,
        grid_radial_fields,
        differentiator,
        accumulator,
        weights,
        n_axis_fix,
    )
    for i in range(out_psin.shape[0]):
        psin_value = out_source_psin_query[i]
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
            out_pprime_input,
            out_driver_input,
            pprime_input,
            driver_input,
            out_parameter_query,
            barycentric_weights,
        )
    else:
        _uniform_spline_interpolate_pair(
            out_pprime_input,
            out_driver_input,
            pprime_spline_coeff,
            driver_spline_coeff,
            out_parameter_query,
        )


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
def _explicit_pchip_interpolate_pair_with_derivatives(
    out0: np.ndarray,
    out1: np.ndarray,
    out0_derivative: np.ndarray,
    out1_derivative: np.ndarray,
    source_nodes: np.ndarray,
    coeff0: np.ndarray,
    coeff1: np.ndarray,
    query: np.ndarray,
    evaluate_derivative0: bool,
    evaluate_derivative1: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate retained PCHIP values and selected derivatives in one lookup."""
    last_interval = source_nodes.shape[0] - 2
    for i in range(out0.shape[0]):
        q = query[i]
        if q <= source_nodes[0]:
            interval = 0
            q = source_nodes[0]
        elif q >= source_nodes[-1]:
            interval = last_interval
            q = source_nodes[-1]
        else:
            low = 0
            high = source_nodes.shape[0] - 1
            while high - low > 1:
                middle = (low + high) // 2
                if source_nodes[middle] <= q:
                    low = middle
                else:
                    high = middle
            interval = low
        dx = q - source_nodes[interval]
        out0[i] = (
            (coeff0[interval, 3] * dx + coeff0[interval, 2]) * dx + coeff0[interval, 1]
        ) * dx + coeff0[interval, 0]
        out1[i] = (
            (coeff1[interval, 3] * dx + coeff1[interval, 2]) * dx + coeff1[interval, 1]
        ) * dx + coeff1[interval, 0]
        if evaluate_derivative0:
            out0_derivative[i] = (
                3.0 * coeff0[interval, 3] * dx + 2.0 * coeff0[interval, 2]
            ) * dx + coeff0[interval, 1]
        if evaluate_derivative1:
            out1_derivative[i] = (
                3.0 * coeff1[interval, 3] * dx + 2.0 * coeff1[interval, 2]
            ) * dx + coeff1[interval, 1]
    return out0, out1


@njit(cache=True, nogil=True)
def _explicit_pchip_interpolate_pair(
    out0: np.ndarray,
    out1: np.ndarray,
    source_nodes: np.ndarray,
    coeff0: np.ndarray,
    coeff1: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate two retained arbitrary-node PCHIP representations in place."""
    return _explicit_pchip_interpolate_pair_with_derivatives(
        out0,
        out1,
        out0,
        out1,
        source_nodes,
        coeff0,
        coeff1,
        query,
        False,
        False,
    )


@njit(cache=True, nogil=True)
def _interpolate_retained_source_pair_impl(
    out0: np.ndarray,
    out1: np.ndarray,
    values0: np.ndarray,
    values1: np.ndarray,
    source_nodes: np.ndarray,
    source_weights: np.ndarray,
    coeff0: np.ndarray,
    coeff1: np.ndarray,
    local_weights: np.ndarray,
    query: np.ndarray,
    interpolation_code: int,
    differentiate_first: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one retained source representation without Python dispatch."""
    if interpolation_code == RETAINED_SOURCE_EXPLICIT_PCHIP:
        return _explicit_pchip_interpolate_pair_with_derivatives(
            out0,
            out1,
            out0,
            out1,
            source_nodes,
            coeff0,
            coeff1,
            query,
            differentiate_first,
            False,
        )
    if interpolation_code == RETAINED_SOURCE_GRID_BARYCENTRIC:
        return _global_barycentric_interpolate_pair(
            out0,
            out1,
            values0,
            values1,
            source_nodes,
            source_weights,
            query,
        )
    if interpolation_code == RETAINED_SOURCE_LOCAL_BARYCENTRIC:
        return _local_barycentric_interpolate_pair(
            out0,
            out1,
            values0,
            values1,
            query,
            local_weights,
        )
    return _uniform_spline_interpolate_pair(
        out0,
        out1,
        coeff0,
        coeff1,
        query,
    )


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
def _global_barycentric_interpolate_pair(
    out0: np.ndarray,
    out1: np.ndarray,
    values0: np.ndarray,
    values1: np.ndarray,
    source_nodes: np.ndarray,
    source_weights: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate two profiles on fixed nonuniform source-coordinate nodes."""
    for i in range(query.shape[0]):
        q = query[i]
        hit = -1
        for j in range(source_nodes.shape[0]):
            if abs(q - source_nodes[j]) <= 1.0e-14:
                hit = j
                break
        if hit >= 0:
            out0[i] = values0[hit]
            out1[i] = values1[hit]
            continue
        denominator = 0.0
        numerator0 = 0.0
        numerator1 = 0.0
        for j in range(source_nodes.shape[0]):
            term = source_weights[j] / (q - source_nodes[j])
            denominator += term
            numerator0 += term * values0[j]
            numerator1 += term * values1[j]
        out0[i] = numerator0 / denominator
        out1[i] = numerator1 / denominator
    return out0, out1


@njit(cache=True, nogil=True)
def _prepare_rho_r_inputs(
    out_pprime_r: np.ndarray,
    out_driver_r: np.ndarray,
    sampled_pprime_s: np.ndarray,
    sampled_driver_s: np.ndarray,
    rho_r: np.ndarray,
    transform_driver_derivative: bool,
) -> None:
    """Apply ds/dr to derivative-valued rho source inputs."""
    for i in range(out_pprime_r.shape[0]):
        jacobian = rho_r[i]
        out_pprime_r[i] = sampled_pprime_s[i] * jacobian
        if transform_driver_derivative:
            out_driver_r[i] = sampled_driver_s[i] * jacobian
        else:
            out_driver_r[i] = sampled_driver_s[i]


@njit(cache=True, nogil=True)
def _update_rho_from_source(
    out_s: np.ndarray,
    out_s_r: np.ndarray,
    out_f: np.ndarray,
    out_f2: np.ndarray,
    FFn_psin: np.ndarray,
    psin_r: np.ndarray,
    alpha1: float,
    alpha2: float,
    edge_f: float,
    Ln_r: np.ndarray,
    accumulator: np.ndarray,
    weights: np.ndarray,
) -> int:
    """Rebuild sqrt(Phi_N) from the source-produced F and current geometry."""
    edge_integral = 0.0
    source_scale = alpha1 * alpha2
    for i in range(out_f.shape[0]):
        # Reuse out_f as physical F*dF/dr until F itself is recovered below.
        out_f[i] = source_scale * FFn_psin[i] * psin_r[i]
        edge_integral += weights[i] * out_f[i]
    edge_f2 = edge_f * edge_f
    sign_f = -1.0 if edge_f < 0.0 else 1.0
    for i in range(out_f.shape[0]):
        prefix = 0.0
        for k in range(out_f.shape[0]):
            prefix += accumulator[i, k] * out_f[k]
        value = edge_f2 + 2.0 * (prefix - edge_integral)
        if not np.isfinite(value):
            return i + 1
        out_f2[i] = value
    for i in range(out_f.shape[0]):
        out_f[i] = sign_f * np.sqrt(out_f2[i])

    phi_edge = 0.0
    for i in range(out_f.shape[0]):
        phi_edge += weights[i] * out_f[i] * Ln_r[i]
    if not np.isfinite(phi_edge) or abs(phi_edge) <= 1.0e-14:
        return -1
    previous = -1.0
    for i in range(out_s.shape[0]):
        phi = 0.0
        for k in range(out_s.shape[0]):
            phi += accumulator[i, k] * out_f[k] * Ln_r[k]
        phin = phi / phi_edge
        if not np.isfinite(phin) or phin <= 0.0 or phin >= 1.0:
            return i + 1
        s = np.sqrt(phin)
        s_r = out_f[i] * Ln_r[i] / (2.0 * s * phi_edge)
        if not np.isfinite(s_r) or s_r <= 0.0 or s <= previous:
            return i + 1
        out_s[i] = s
        out_s_r[i] = s_r
        previous = s
    return 0


@njit(cache=True, nogil=True)
def _update_rho_from_u(
    out_s: np.ndarray,
    out_s_r: np.ndarray,
    out_f: np.ndarray,
    u: np.ndarray,
    edge_f: float,
    Ln_r: np.ndarray,
    accumulator: np.ndarray,
    weights: np.ndarray,
) -> int:
    """Rebuild sqrt(Phi_N) directly from the PJ2/PJ3 ``u=log(F^2/Fedge^2)`` state."""
    for i in range(out_f.shape[0]):
        value = edge_f * np.exp(0.5 * u[i])
        if not np.isfinite(value):
            return i + 1
        out_f[i] = value
    phi_edge = 0.0
    for i in range(out_f.shape[0]):
        phi_edge += weights[i] * out_f[i] * Ln_r[i]
    if not np.isfinite(phi_edge) or abs(phi_edge) <= 1.0e-14:
        return -1
    previous = -1.0
    for i in range(out_s.shape[0]):
        phi = 0.0
        for k in range(out_s.shape[0]):
            phi += accumulator[i, k] * out_f[k] * Ln_r[k]
        phin = phi / phi_edge
        if not np.isfinite(phin) or phin <= 0.0 or phin >= 1.0:
            return i + 1
        s = np.sqrt(phin)
        s_r = out_f[i] * Ln_r[i] / (2.0 * s * phi_edge)
        if not np.isfinite(s_r) or s_r <= 0.0 or s <= previous:
            return i + 1
        out_s[i] = s
        out_s_r[i] = s_r
        previous = s
    return 0


@njit(cache=True, nogil=True)
def _rho_fixed_point_defect(
    current_s: np.ndarray,
    current_s_r: np.ndarray,
    next_s: np.ndarray,
    next_s_r: np.ndarray,
) -> tuple[float, float, float]:
    value_defect = 0.0
    derivative_defect = 0.0
    for i in range(current_s.shape[0]):
        value_error = abs(next_s[i] - current_s[i])
        derivative_error = abs(next_s_r[i] - current_s_r[i]) / (1.0 + abs(next_s_r[i]))
        if value_error > value_defect:
            value_defect = value_error
        if derivative_error > derivative_defect:
            derivative_defect = derivative_error
    return max(value_defect, derivative_defect), value_defect, derivative_defect


@njit(cache=True, nogil=True)
def uniform_barycentric_weights(source_sample_count: int) -> np.ndarray:
    weights = np.empty(source_sample_count, dtype=np.float64)
    weights[0] = 1.0
    for j in range(1, source_sample_count):
        weights[j] = -weights[j - 1] * (source_sample_count - j) / j
    return weights


@njit(cache=True, nogil=True)
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
