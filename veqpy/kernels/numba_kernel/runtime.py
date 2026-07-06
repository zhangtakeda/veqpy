"""Direct runtime assembly for the Numba backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from veqlib.facade import (
    KernelBoundary,
    KernelSource,
    KernelTopology,
)
from veqlib.facade.source_semantics import materialize_kernel_source
from veqpy.kernels.numba_kernel.workspace import allocate_runtime_state
from veqpy.model import Grid
from veqpy.model.numerics import (
    SOURCE_INTERP_DEFAULT,
    GridWorkspace,
    normalize_source_interpolation_kind,
)

from . import backend_abi
from .initialize import build_boundary_slope_initial_state
from .layout import KernelLayout
from .layout_binding import build_kernel_layout
from .numba_source import source_parameterization_for_route_key, validate_route
from .packed_layout import (
    PROFILE_OFFSET_SPECS,
    PROFILE_STATIC_KWARGS,
    build_active_profile_metadata,
    build_fourier_profile_names,
    build_profile_index,
    build_profile_layout,
    build_profile_names,
    build_residual_block_metadata,
    build_residual_block_radial_powers,
    build_shape_profile_names,
    get_prefix_profile_names,
    packed_size,
)
from .profile_runtime import (
    refresh_fourier_family_metadata,
    refresh_profile_runtime,
    refresh_stage_a_runtime,
)
from .snapshot import snapshot_equilibrium_from_kernel_runtime
from .source_plan import SourcePlan
from .source_runtime import refresh_source_runtime

_AUTO_CURVE_STRAIN_THRESHOLD = 0.20
_AUTO_CURVE_STRAIN_SAMPLES = 32
_AUTO_CURVE_STRAIN_MAX_ORDER = 32
_AUTO_CURVE_STRAIN_THETA = np.linspace(
    0.0,
    2.0 * np.pi,
    _AUTO_CURVE_STRAIN_SAMPLES,
    endpoint=False,
    dtype=np.float64,
)
_AUTO_CURVE_STRAIN_SIN_THETA = np.sin(_AUTO_CURVE_STRAIN_THETA)
_AUTO_CURVE_STRAIN_COS_THETA = np.cos(_AUTO_CURVE_STRAIN_THETA)
_AUTO_CURVE_STRAIN_ORDERS = np.arange(
    1,
    _AUTO_CURVE_STRAIN_MAX_ORDER + 1,
    dtype=np.float64,
)
_AUTO_CURVE_STRAIN_ORDER_THETA = (
    _AUTO_CURVE_STRAIN_ORDERS[:, np.newaxis] * _AUTO_CURVE_STRAIN_THETA[np.newaxis, :]
)
_AUTO_CURVE_STRAIN_SIN_ORDER_THETA = np.sin(_AUTO_CURVE_STRAIN_ORDER_THETA)
_AUTO_CURVE_STRAIN_COS_ORDER_THETA = np.cos(_AUTO_CURVE_STRAIN_ORDER_THETA)


@dataclass(frozen=True, slots=True)
class KernelResidualBindingLayout:
    """Read-only residual packing metadata for direct kernel runtime."""

    active_profile_names: tuple[str, ...]
    active_residual_block_codes: np.ndarray
    active_residual_block_orders: np.ndarray
    active_residual_block_radial_powers: np.ndarray


@dataclass(slots=True)
class KernelRuntimePlan:
    """Static topology and refreshed case data consumed by Numba layout binders."""

    grid_workspace: GridWorkspace
    prefix_profile_names: tuple[str, ...]
    shape_profile_names: tuple[str, ...]
    profile_names: tuple[str, ...]
    profile_index: dict[str, int]
    c_profile_names: tuple[str, ...]
    s_profile_names: tuple[str, ...]
    profile_L: np.ndarray
    coeff_index: np.ndarray
    order_offsets: np.ndarray
    active_profile_mask: np.ndarray
    active_profile_ids: np.ndarray
    x_size: int
    source_route_spec: object
    source_plan: SourcePlan
    source_execution: backend_abi.SourceExecutionABI
    residual_binding_layout: KernelResidualBindingLayout
    profile_static_kwargs_by_name: dict[str, dict[str, float | int]]
    profile_offset_specs: dict[str, float | str]
    profile_offsets: np.ndarray
    profile_scales: np.ndarray
    profile_powers: np.ndarray
    profile_envelope_powers: np.ndarray
    profile_amplitude_powers: np.ndarray


@dataclass(frozen=True, slots=True)
class KernelRuntimeCase:
    """Runtime case object with the scalar attributes layout binders need."""

    topology: KernelTopology
    boundary: KernelBoundary
    source: KernelSource

    @property
    def route(self) -> str:
        return self.topology.route

    @property
    def coordinate(self) -> str:
        return self.topology.coordinate

    @property
    def nodes(self) -> str:
        return self.topology.nodes

    @property
    def active_profiles(self) -> dict[str, int]:
        return dict(self.topology.active_profiles)

    @property
    def a(self) -> float:
        return self.boundary.a

    @property
    def R0(self) -> float:
        return self.boundary.R0

    @property
    def Z0(self) -> float:
        return self.boundary.Z0

    @property
    def B0(self) -> float:
        return self.boundary.B0

    @property
    def ka(self) -> float:
        return self.boundary.ka

    @property
    def c_offsets(self) -> np.ndarray:
        return self.boundary.c_offsets

    @property
    def s_offsets(self) -> np.ndarray:
        return self.boundary.s_offsets

    @property
    def heat_input(self) -> np.ndarray:
        return self.source.heat_profile

    @property
    def current_input(self) -> np.ndarray:
        return self.source.current_profile


class NumbaRuntime:
    """Topology-native residual runtime for the Numba backend."""

    def __init__(
        self,
        topology: KernelTopology,
        *,
        fix_rho: float = 0.05,
        source_interpolation_kind: str = SOURCE_INTERP_DEFAULT,
    ) -> None:
        self.topology = topology
        self.fix_rho = float(fix_rho)
        self.source_interpolation_kind = source_interpolation_kind
        self.plan = _build_kernel_runtime_plan(
            topology,
            source_interpolation_kind=source_interpolation_kind,
        )
        self.layout = KernelLayout.empty(self.plan.x_size)
        (
            self.profile_workspace,
            self.geometry_workspace,
            self.source_workspace,
            self.residual_workspace,
        ) = allocate_runtime_state(
            grid_workspace=self.plan.grid_workspace,
            source_execution=self.plan.source_execution,
            profile_names=self.plan.profile_names,
            profile_index=self.plan.profile_index,
            active_profile_ids=self.plan.active_profile_ids,
            profile_L=self.plan.profile_L,
            x_size=self.plan.x_size,
        )
        self.c_effective_order = 0
        self.s_effective_order = 0
        self._case: KernelRuntimeCase | None = None

    @property
    def x_size(self) -> int:
        return self.plan.x_size

    @property
    def alpha(self) -> np.ndarray:
        return self.source_workspace.alpha_state

    def zero_state(self) -> np.ndarray:
        return np.zeros(self.plan.x_size, dtype=np.float64)

    def coerce_x(self, x: np.ndarray) -> np.ndarray:
        if isinstance(x, np.ndarray) and x.dtype == np.float64:
            if x.ndim != 1 or x.shape[0] != self.plan.x_size:
                raise ValueError(f"Expected x to have shape ({self.plan.x_size},), got {x.shape}")
            if x.flags.c_contiguous:
                return x
            return np.ascontiguousarray(x)
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != self.plan.x_size:
            raise ValueError(f"Expected x to have shape ({self.plan.x_size},), got {arr.shape}")
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr, dtype=np.float64)
        return arr

    def residual_block_lengths(self) -> np.ndarray:
        return self.profile_workspace.residual_block_lengths()

    def active_profile_blocks(self) -> tuple[tuple[int, str, np.ndarray, float, float], ...]:
        return self.profile_workspace.active_profile_blocks()

    def set_case(self, boundary: KernelBoundary, source: KernelSource) -> None:
        case = KernelRuntimeCase(self.topology, boundary, source)
        self.plan.source_plan = _build_kernel_source_plan(
            self.topology,
            source,
            source_interpolation_kind=self.source_interpolation_kind,
        )
        self.plan.source_execution = backend_abi.build_source_execution_abi(
            source_plan=self.plan.source_plan,
            profile_index=self.plan.profile_index,
            profile_L=self.plan.profile_L,
            coeff_index=self.plan.coeff_index,
            active_profile_ids=self.plan.active_profile_ids,
        )
        self._case = case
        self._refresh_runtime_state(case)

    def residual_into(
        self,
        out: np.ndarray,
        x: np.ndarray,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> None:
        self.set_case(boundary, source)
        x_eval = self.coerce_x(x)
        self.layout.run_fused_residual_into(x_eval, out)

    def residual(
        self,
        x: np.ndarray,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> np.ndarray:
        out = np.empty(self.plan.x_size, dtype=np.float64)
        self.residual_into(out, x, boundary, source)
        return out

    def residual_into_for_current_case(self, out: np.ndarray, x: np.ndarray) -> None:
        self._require_case()
        self.layout.run_fused_residual_into(self.coerce_x(x), out)

    def residual_for_current_case(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(self.plan.x_size, dtype=np.float64)
        self.residual_into_for_current_case(out, x)
        return out

    def initial_state(
        self,
        boundary: KernelBoundary,
        source: KernelSource,
        *,
        initial: str,
        x0: np.ndarray | None,
    ) -> np.ndarray:
        self.set_case(boundary, source)
        if x0 is not None:
            return self.coerce_x(x0).copy()
        if initial == "cold-zeros":
            return self.zero_state()
        if initial == "cold":
            if _boundary_curve_strain(boundary) >= _AUTO_CURVE_STRAIN_THRESHOLD:
                return self._build_geometric_initial_state()
            return self.zero_state()
        if initial == "cold-geometric":
            return self._build_geometric_initial_state()
        raise NotImplementedError(
            f"Numba backend does not support KernelConfig.initial={initial!r}"
        )

    def build_equilibrium(
        self,
        x: np.ndarray,
        boundary: KernelBoundary,
        source: KernelSource,
    ):
        self.set_case(boundary, source)
        x_eval = self.coerce_x(x)
        out = np.empty(self.plan.x_size, dtype=np.float64)
        self.layout.run_fused_residual_into(x_eval, out)
        root_fields = self.residual_workspace.root_fields
        case = self._require_case()
        return snapshot_equilibrium_from_kernel_runtime(
            x=x_eval,
            a=case.a,
            R0=case.R0,
            Z0=case.Z0,
            B0=case.B0,
            grid=self.plan.grid_workspace.to_grid(),
            profile_L=self.plan.profile_L,
            coeff_index=self.plan.coeff_index,
            profile_names=self.plan.profile_names,
            shape_profile_names=self.plan.shape_profile_names,
            profile_index=self.plan.profile_index,
            profile_offsets=self.plan.profile_offsets,
            profile_scales=self.plan.profile_scales,
            profile_powers=self.plan.profile_powers,
            profile_envelope_powers=self.plan.profile_envelope_powers,
            profile_amplitude_powers=self.plan.profile_amplitude_powers,
            psin=root_fields[0],
            FFn_psin=root_fields[3],
            Pn_psin=root_fields[4],
            psin_r=root_fields[1],
            psin_rr=root_fields[2],
            alpha1=float(self.source_workspace.alpha_state[0]),
            alpha2=float(self.source_workspace.alpha_state[1]),
        )

    def _refresh_runtime_state(self, case: KernelRuntimeCase) -> None:
        refresh_profile_runtime(
            case=case,
            operator_grid=self.plan.grid_workspace,
            profile_names=self.plan.profile_names,
            profile_workspace=self.profile_workspace,
            profile_offsets=self.plan.profile_offsets,
            profile_scales=self.plan.profile_scales,
            profile_powers=self.plan.profile_powers,
            profile_envelope_powers=self.plan.profile_envelope_powers,
            profile_amplitude_powers=self.plan.profile_amplitude_powers,
            profile_static_kwargs_by_name=self.plan.profile_static_kwargs_by_name,
            profile_offset_specs=self.plan.profile_offset_specs,
        )
        self.c_effective_order, self.s_effective_order = refresh_fourier_family_metadata(
            c_profile_names=self.plan.c_profile_names,
            s_profile_names=self.plan.s_profile_names,
            profile_L=self.plan.profile_L,
            profile_index=self.plan.profile_index,
            c_offsets=case.c_offsets,
            s_offsets=case.s_offsets,
            c_family_fields=self.profile_workspace.c_family_fields,
            s_family_fields=self.profile_workspace.s_family_fields,
        )
        refresh_source_runtime(
            grid_rho=self.plan.grid_workspace.rho,
            source_plan=self.plan.source_plan,
            source_execution=self.plan.source_execution,
            source_workspace=self.source_workspace,
            psin=self.residual_workspace.root_fields[0],
        )
        refresh_stage_a_runtime(
            active_profile_ids=self.plan.active_profile_ids,
            profile_L=self.plan.profile_L,
            coeff_index=self.plan.coeff_index,
            profile_offsets=self.plan.profile_offsets,
            profile_scales=self.plan.profile_scales,
            profile_amplitude_powers=self.plan.profile_amplitude_powers,
            active_offsets=self.profile_workspace.active_offsets,
            active_scales=self.profile_workspace.active_scales,
            active_amplitude_powers=self.profile_workspace.active_amplitude_powers,
            active_lengths=self.profile_workspace.active_lengths,
            active_coeff_index_rows=self.profile_workspace.active_coeff_index_rows,
        )
        self._refresh_runtime_bindings(case)

    def _refresh_runtime_bindings(self, case: KernelRuntimeCase) -> None:
        self.layout = build_kernel_layout(
            plan=self.plan,
            case=case,
            profile_workspace=self.profile_workspace,
            geometry_workspace=self.geometry_workspace,
            source_workspace=self.source_workspace,
            residual_workspace=self.residual_workspace,
            grid_workspace=self.plan.grid_workspace,
            residual_binding_layout=self.plan.residual_binding_layout,
            c_effective_order=self.c_effective_order,
            s_effective_order=self.s_effective_order,
            fix_rho=self.fix_rho,
            psin_profile_fields_available=self.profile_workspace.has_fields_for("psin"),
        )
        fixed_profile_ids = np.flatnonzero(~self.plan.active_profile_mask).astype(
            np.int64, copy=False
        )
        for profile_id in fixed_profile_ids:
            p = int(profile_id)
            self.profile_workspace.refresh_profile_fields(
                profile_id=p,
                offset=float(self.plan.profile_offsets[p]),
                scale=float(self.plan.profile_scales[p]),
                amplitude_power=float(self.plan.profile_amplitude_powers[p]),
                coeff=None,
                grid_workspace=self.plan.grid_workspace,
            )

    def _build_geometric_initial_state(self) -> np.ndarray:
        x = build_boundary_slope_initial_state(
            case=self._require_case(),
            plan=self.plan,
            profile_workspace=self.profile_workspace,
            source_psin_target=self._source_psin_target_for_initial_state,
        )
        self.invalidate_source_state()
        return x

    def _source_psin_target_for_initial_state(self, x: np.ndarray) -> np.ndarray | None:
        if not bool(self.plan.source_execution.requires_optimized_psin_profile):
            return None
        target_root_fields = self.source_workspace.target_root_fields
        if target_root_fields.shape[1] == 0:
            return None
        self.layout.run_profile(x)
        self.layout.run_geometry()
        self.layout.run_source()
        return target_root_fields[0]

    def invalidate_source_state(self) -> None:
        if tuple(self.plan.source_execution.route_key) == ("PJ2", "psin", "uniform"):
            self.source_workspace.psin_query.fill(-1.0)

    def _require_case(self) -> KernelRuntimeCase:
        if self._case is None:
            raise RuntimeError("NumbaRuntime requires a bound runtime case")
        return self._case


def _build_kernel_runtime_plan(
    topology: KernelTopology,
    *,
    source_interpolation_kind: str,
) -> KernelRuntimePlan:
    grid_workspace = GridWorkspace.from_grid(
        Grid(
            Nr=topology.Nr,
            Nt=topology.Nt,
            L_max=topology.L_max,
            M_max=topology.M_max,
            K_max=topology.K_max,
            quadrature_scheme=topology.quadrature,
            calculus_scheme=topology.calculus,
        )
    )
    prefix_profile_names = get_prefix_profile_names()
    shape_profile_names = build_shape_profile_names(grid_workspace.M_max)
    profile_names = build_profile_names(grid_workspace.M_max)
    profile_index = build_profile_index(profile_names)
    fourier_profile_names = build_fourier_profile_names(grid_workspace.M_max)
    c_profile_names = tuple(name for name in fourier_profile_names if name.startswith("c"))
    s_profile_names = tuple(name for name in fourier_profile_names if name.startswith("s"))
    profile_L, coeff_index, order_offsets = build_profile_layout(
        dict(topology.active_profiles),
        profile_names=profile_names,
        prefix_profile_names=prefix_profile_names,
    )
    active_profile_mask, active_profile_ids = build_active_profile_metadata(
        profile_L,
        profile_names=profile_names,
    )
    x_size = packed_size(coeff_index)
    if x_size != topology.x_size:
        raise ValueError(
            f"KernelTopology x_size={topology.x_size} disagrees with packed layout x_size={x_size}"
        )
    source_route_spec = validate_route(topology.route, topology.coordinate, topology.nodes)
    source_plan = _placeholder_source_plan(
        topology,
        source_route_spec=source_route_spec,
        source_interpolation_kind=source_interpolation_kind,
    )
    source_execution = backend_abi.build_source_execution_abi(
        source_plan=source_plan,
        profile_index=profile_index,
        profile_L=profile_L,
        coeff_index=coeff_index,
        active_profile_ids=active_profile_ids,
    )
    profile_static_kwargs_by_name, profile_offset_specs = _build_profile_config(
        grid_workspace=grid_workspace,
        c_profile_names=c_profile_names,
        s_profile_names=s_profile_names,
    )
    profile_count = len(profile_names)
    profile_offsets = np.zeros(profile_count, dtype=np.float64)
    profile_scales = np.ones(profile_count, dtype=np.float64)
    profile_powers = np.zeros(profile_count, dtype=np.int64)
    profile_envelope_powers = np.ones(profile_count, dtype=np.int64)
    profile_amplitude_powers = np.ones(profile_count, dtype=np.float64)
    return KernelRuntimePlan(
        grid_workspace=grid_workspace,
        prefix_profile_names=prefix_profile_names,
        shape_profile_names=shape_profile_names,
        profile_names=profile_names,
        profile_index=profile_index,
        c_profile_names=c_profile_names,
        s_profile_names=s_profile_names,
        profile_L=profile_L,
        coeff_index=coeff_index,
        order_offsets=order_offsets,
        active_profile_mask=active_profile_mask,
        active_profile_ids=active_profile_ids,
        x_size=x_size,
        source_route_spec=source_route_spec,
        source_plan=source_plan,
        source_execution=source_execution,
        residual_binding_layout=_build_kernel_residual_binding_layout(
            profile_names=profile_names,
            active_profile_ids=active_profile_ids,
            K_values=grid_workspace.K_values,
        ),
        profile_static_kwargs_by_name=profile_static_kwargs_by_name,
        profile_offset_specs=profile_offset_specs,
        profile_offsets=profile_offsets,
        profile_scales=profile_scales,
        profile_powers=profile_powers,
        profile_envelope_powers=profile_envelope_powers,
        profile_amplitude_powers=profile_amplitude_powers,
    )


def _build_kernel_source_plan(
    topology: KernelTopology,
    source: KernelSource,
    *,
    source_interpolation_kind: str,
) -> SourcePlan:
    materialized = materialize_kernel_source(topology, source)
    source_route_spec = validate_route(topology.route, topology.coordinate, topology.nodes)
    return SourcePlan(
        route=topology.route,
        kernel=source_route_spec.implementation,
        coordinate=topology.coordinate,
        nodes=topology.nodes,
        parameterization=topology.source_parameterization,
        source_sample_count=int(materialized.scaled_heat.shape[0]),
        scaled_heat=materialized.scaled_heat,
        scaled_current=materialized.scaled_current,
        scaled_Ip=materialized.scaled_Ip,
        beta=materialized.beta,
        interpolation_kind=_interpolation_kind_for(topology, source_interpolation_kind),
    )


def _placeholder_source_plan(
    topology: KernelTopology,
    *,
    source_route_spec: object,
    source_interpolation_kind: str,
) -> SourcePlan:
    samples = int(topology.sample_count)
    placeholder = np.ones(samples, dtype=np.float64)
    placeholder.setflags(write=False)
    return SourcePlan(
        route=topology.route,
        kernel=source_route_spec.implementation,
        coordinate=topology.coordinate,
        nodes=topology.nodes,
        parameterization=source_parameterization_for_route_key(topology.source_route_key),
        source_sample_count=samples,
        scaled_heat=placeholder,
        scaled_current=placeholder,
        scaled_Ip=np.nan,
        beta=np.nan,
        interpolation_kind=_interpolation_kind_for(topology, source_interpolation_kind),
    )


def _interpolation_kind_for(topology: KernelTopology, kind: str) -> str:
    if topology.nodes == "grid":
        return ""
    return normalize_source_interpolation_kind(kind)


def _build_kernel_residual_binding_layout(
    *,
    profile_names: tuple[str, ...],
    active_profile_ids: np.ndarray,
    K_values: np.ndarray,
) -> KernelResidualBindingLayout:
    active_profile_names = tuple(profile_names[int(p)] for p in active_profile_ids)
    active_residual_block_codes, active_residual_block_orders = build_residual_block_metadata(
        active_profile_names
    )
    active_residual_block_radial_powers = build_residual_block_radial_powers(
        active_profile_names,
        K_values=K_values,
    )
    return KernelResidualBindingLayout(
        active_profile_names=active_profile_names,
        active_residual_block_codes=active_residual_block_codes,
        active_residual_block_orders=active_residual_block_orders,
        active_residual_block_radial_powers=active_residual_block_radial_powers,
    )


def _build_profile_config(
    *,
    grid_workspace: GridWorkspace,
    c_profile_names: tuple[str, ...],
    s_profile_names: tuple[str, ...],
) -> tuple[dict[str, dict[str, float | int]], dict[str, float | str]]:
    profile_static_kwargs_by_name = {
        name: dict(kwargs) for name, kwargs in PROFILE_STATIC_KWARGS.items()
    }
    for name in c_profile_names + s_profile_names:
        order = int(name[1:])
        profile_static_kwargs_by_name[name] = (
            {} if order == 0 else {"power": int(grid_workspace.K_values[order])}
        )
    return profile_static_kwargs_by_name, dict(PROFILE_OFFSET_SPECS)


def _boundary_curve_strain(boundary: KernelBoundary) -> float:
    c_offsets = _boundary_offset_array(boundary.c_offsets)
    s_offsets = _boundary_offset_array(boundary.s_offsets)
    if c_offsets is None or s_offsets is None:
        return float("inf")
    has_c_shape = c_offsets.size > 0 and bool(np.any(c_offsets != 0.0))
    has_s_shape = s_offsets.size > 1 and bool(np.any(s_offsets[1:] != 0.0))
    if not has_c_shape and not has_s_shape:
        return 0.0
    kappa = abs(float(boundary.ka))
    if not np.isfinite(kappa):
        return float("inf")

    theta = _AUTO_CURVE_STRAIN_THETA
    eta = np.zeros_like(_AUTO_CURVE_STRAIN_THETA)
    eta_prime = np.zeros_like(_AUTO_CURVE_STRAIN_THETA)
    if c_offsets.size:
        eta += c_offsets[0]

    c_fast_count = min(max(c_offsets.size - 1, 0), _AUTO_CURVE_STRAIN_MAX_ORDER)
    if c_fast_count:
        c_tail = c_offsets[1 : c_fast_count + 1]
        eta += c_tail @ _AUTO_CURVE_STRAIN_COS_ORDER_THETA[:c_fast_count]
        eta_prime -= (
            _AUTO_CURVE_STRAIN_ORDERS[:c_fast_count] * c_tail
        ) @ _AUTO_CURVE_STRAIN_SIN_ORDER_THETA[:c_fast_count]
    for order in range(c_fast_count + 1, c_offsets.size):
        order_theta = float(order) * theta
        eta += c_offsets[order] * np.cos(order_theta)
        eta_prime -= float(order) * c_offsets[order] * np.sin(order_theta)

    s_fast_count = min(max(s_offsets.size - 1, 0), _AUTO_CURVE_STRAIN_MAX_ORDER)
    if s_fast_count:
        s_tail = s_offsets[1 : s_fast_count + 1]
        eta += s_tail @ _AUTO_CURVE_STRAIN_SIN_ORDER_THETA[:s_fast_count]
        eta_prime += (
            _AUTO_CURVE_STRAIN_ORDERS[:s_fast_count] * s_tail
        ) @ _AUTO_CURVE_STRAIN_COS_ORDER_THETA[:s_fast_count]
    for order in range(s_fast_count + 1, s_offsets.size):
        order_theta = float(order) * theta
        eta += s_offsets[order] * np.sin(order_theta)
        eta_prime += float(order) * s_offsets[order] * np.cos(order_theta)

    speed_boundary = np.sqrt(
        (np.sin(theta + eta) * (1.0 + eta_prime)) ** 2
        + (kappa * _AUTO_CURVE_STRAIN_COS_THETA) ** 2
    )
    speed_ellipse = np.sqrt(
        _AUTO_CURVE_STRAIN_SIN_THETA**2 + (kappa * _AUTO_CURVE_STRAIN_COS_THETA) ** 2
    )
    strain = (speed_boundary - speed_ellipse) / np.maximum(speed_ellipse, 1.0e-12)
    return float(np.sqrt(np.mean(strain * strain)))


def _boundary_offset_array(value: object) -> np.ndarray | None:
    if value is None:
        return np.zeros(1, dtype=np.float64)
    try:
        offsets = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if offsets.size == 0 or not bool(np.all(np.isfinite(offsets))):
        return None
    return offsets
