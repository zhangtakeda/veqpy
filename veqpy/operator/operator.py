"""
Module: operator.operator

Role:
- Connect case, grid, model runtime, engine kernels, and packed layout.
- Expose stable residual evaluation interfaces.

Public API:
- Operator

Notes:
- `Operator` is the default fused operator.
- Does not own solver iteration policy, backend selection, or benchmark orchestration.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

import numpy as np

from veqpy.layout.binding import build_operator_layout
from veqpy.layout.runtime import OperatorLayout
from veqpy.math.interpolate import SOURCE_INTERP_DEFAULT
from veqpy.model.equilibrium import Equilibrium
from veqpy.model.grid import Grid
from veqpy.model.profile import Profile
from veqpy.operator.build_plan import (
    OperatorBuildPlan,
    build_operator_plan,
    refresh_operator_plan_for_case,
)
from veqpy.operator.operator_case import OperatorCase
from veqpy.operator.packed_layout import (
    decode_packed_blocks,
    encode_packed_state,
)
from veqpy.operator.profile_runtime import (
    make_profile,
    refresh_fourier_family_metadata,
    refresh_profile_runtime,
    refresh_stage_a_runtime,
    validate_case_compatibility,
)
from veqpy.operator.snapshot import snapshot_equilibrium_from_runtime
from veqpy.operator.source_plan import (
    validate_source_inputs,
    validate_source_plan_profile_support,
)
from veqpy.operator.source_runtime import refresh_source_runtime
from veqpy.workspace import allocate_runtime_state
from veqpy.workspace.geometry_workspace import GeometryWorkspace
from veqpy.workspace.profile_workspace import ProfileWorkspace
from veqpy.workspace.residual_workspace import ResidualWorkspace
from veqpy.workspace.source_workspace import SourceWorkspace


@dataclass(slots=True)
class Operator:
    """Encapsulate the residual evaluator for a fixed case, grid, and runtime."""

    grid: InitVar[Grid]
    case: OperatorCase = field(repr=False)
    fix_rho: float = 0.05
    source_interpolation_kind: str = SOURCE_INTERP_DEFAULT
    plan: OperatorBuildPlan = field(init=False, repr=False)
    profile_workspace: ProfileWorkspace = field(init=False, repr=False)
    geometry_workspace: GeometryWorkspace = field(init=False, repr=False)
    source_workspace: SourceWorkspace = field(init=False, repr=False)
    residual_workspace: ResidualWorkspace = field(init=False, repr=False)
    layout: OperatorLayout = field(init=False, repr=False)

    h_profile: Profile = field(init=False)
    v_profile: Profile = field(init=False)
    k_profile: Profile = field(init=False)
    psin_profile: Profile = field(init=False)
    F_profile: Profile = field(init=False)
    profiles_by_name: dict[str, Profile] = field(init=False, repr=False)

    c_effective_order: int = field(init=False, repr=False)
    s_effective_order: int = field(init=False, repr=False)

    def __post_init__(self, grid: Grid) -> None:
        """Build layouts, allocate runtime buffers, and bind the case.

        The input grid is lowered to a GridWorkspace snapshot at construction time;
        Operator does not read live Grid state afterwards.
        """
        self._apply_plan(
            build_operator_plan(
                grid=grid,
                case=self.case,
                source_interpolation_kind=self.source_interpolation_kind,
            )
        )
        self._validate_runtime_profile_support()

        self.layout = OperatorLayout.empty(self.plan.x_size)
        self._setup_runtime_state()
        self._refresh_runtime_state()

    def _apply_plan(self, plan: OperatorBuildPlan) -> None:
        """Install the operator topology/configuration plan."""

        self.plan = plan

    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Call the main variational residual evaluation entrypoint."""
        return self.residual_var(x, *args, **kwargs)

    @property
    def alpha1(self) -> float:
        """Current source normalization owned by the source runtime state."""
        return float(self.source_workspace.alpha_state[0])

    @alpha1.setter
    def alpha1(self, value: float) -> None:
        """Update the current source normalization in place."""
        self.source_workspace.alpha_state[0] = float(value)

    @property
    def alpha2(self) -> float:
        """Flux/source normalization owned by the source runtime state."""
        return float(self.source_workspace.alpha_state[1])

    @alpha2.setter
    def alpha2(self, value: float) -> None:
        """Update the flux/source normalization in place."""
        self.source_workspace.alpha_state[1] = float(value)

    # Solver-facing plan accessors kept as the public facade; Operator does not
    # mirror these fields as mutable state.
    @property
    def x_size(self) -> int:
        """Length of the packed unknown vector accepted by this operator."""
        return self.plan.x_size

    @property
    def profile_names(self) -> tuple[str, ...]:
        """All profile names in packed-layout order."""
        return self.plan.profile_names

    @property
    def active_profile_ids(self) -> np.ndarray:
        """Integer ids of profiles that are active in the packed state."""
        return self.plan.active_profile_ids

    def residual_block_lengths(self) -> np.ndarray:
        """Return packed residual block lengths for solver normalization.

        This is a narrow solver-facing view; raw workspace indexing arrays remain
        owned by ``ProfileWorkspace``.
        """
        return self.profile_workspace.residual_block_lengths()

    def active_profile_blocks(self) -> tuple[tuple[int, str, np.ndarray, float, float], ...]:
        """Return solver-scale metadata for active packed profile blocks.

        Coefficient index arrays are copies so callers do not depend on workspace
        storage layout.
        """

        return self.profile_workspace.active_profile_blocks()

    def build_boundary_slope_initial_state(self) -> np.ndarray:
        """Build a geometrically-motivated packed x0 in a single pass.

        c/s Fourier profiles use ``-offset / (2*p + 1)``.
        ``h`` uses the Shafranov-shift estimate ``0.66 * a / R0``.
        For uniform profiles (Solovev-like) all coefficients are left
        at zero — the toroidal correction is small and boundary-based
        shaping estimates are unreliable for analytic equilibria.
        """

        pw = self.profile_workspace
        x = np.zeros(self.plan.x_size, dtype=np.float64)

        h0_est = _estimate_h0_from_case(self.case)
        if h0_est == 0.0:
            return x  # uniform → pure zeros, skip c/s estimates

        for slot, profile_id in enumerate(pw.active_profile_ids):
            length = int(pw.active_lengths[slot])
            if length <= 0:
                continue
            name = pw.profile_names[int(profile_id)]
            idx0 = int(pw.active_coeff_index_rows[slot, 0])
            if name.startswith("c") or name.startswith("s"):
                profile = self.profiles_by_name[name]
                offset = float(profile.offset)
                if abs(offset) > 1.0e-14:
                    x[idx0] = -offset / float(2 * int(profile.power) + 1)
            elif name == "h":
                x[idx0] = h0_est
        return x

    def _validate_runtime_profile_support(self) -> None:
        """Validate psin profile ownership requirements for the current source route."""
        validate_source_plan_profile_support(
            source_plan=self.plan.source_plan,
            source_execution=self.plan.source_execution,
            case=self.case,
        )
        return None

    def replace_case(self, case: OperatorCase) -> None:
        """Replace the current case without changing the packed layout."""
        validate_case_compatibility(
            case,
            profile_names=self.plan.profile_names,
            prefix_profile_names=self.plan.prefix_profile_names,
            profile_L=self.plan.profile_L,
            coeff_index=self.plan.coeff_index,
            order_offsets=self.plan.order_offsets,
            validate_source_inputs=lambda next_case: validate_source_inputs(
                next_case, self.plan.grid_workspace.Nr
            ),
        )
        self.case = case
        self._refresh_runtime_state()

    def encode_initial_state(self) -> np.ndarray:
        """Encode profile coefficients from the current case into the packed initial state."""
        return encode_packed_state(
            self.case.profile_coeffs,
            self.plan.profile_L,
            self.plan.coeff_index,
            profile_names=self.plan.profile_names,
        )

    def residual_var(
        self,
        x: np.ndarray,
        *,
        check: bool = True,
    ) -> np.ndarray:
        """Return the variational/Galerkin residual vector."""
        out = np.empty(self.plan.x_size, dtype=np.float64)
        self.residual_var_into(x, out, check=check)
        return out

    def residual_var_into(
        self,
        x: np.ndarray,
        out: np.ndarray,
        *,
        check: bool = True,
    ) -> None:
        """Write the variational/Galerkin residual into caller-provided ``out``."""
        if check:
            x_eval = self.coerce_x(x)
            if not isinstance(out, np.ndarray):
                raise TypeError("Expected out to be a numpy.ndarray")
            out_eval = out
            if out_eval.dtype != np.float64:
                raise TypeError(f"Expected out dtype float64, got {out_eval.dtype}")
            if out_eval.ndim != 1 or out_eval.shape[0] != self.plan.x_size:
                raise ValueError(
                    f"Expected out to have shape ({self.plan.x_size},), got {out_eval.shape}"
                )
            if not out_eval.flags.c_contiguous:
                raise ValueError("Expected out to be C-contiguous")
        else:
            x_eval = x
            out_eval = out
        self.layout.run_fused_residual_into(x_eval, out_eval)

    def residual_collocation(self, x: np.ndarray) -> np.ndarray:
        """Return the quadrature-scaled pointwise collocation residual.

        This residual does not append a Galerkin/weak-form residual to an external
        objective. Instead, it directly constrains ``R/J * G`` at every collocation
        node, where ``G = J/R * GS_residual``. Square-root radial/poloidal quadrature
        weights provide the discrete least-squares scaling. The returned vector has
        shape ``(Nr * Nt,)``.
        """
        out = np.empty(self.plan.grid_workspace.Nr * self.plan.grid_workspace.Nt, dtype=np.float64)
        self.residual_collocation_into(x, out)
        return out

    def residual_collocation_into(self, x: np.ndarray, out: np.ndarray) -> None:
        """Write the quadrature-scaled collocation residual into caller-provided ``out``."""
        x_eval = self.coerce_x(x)
        expected_size = self.plan.grid_workspace.Nr * self.plan.grid_workspace.Nt
        if not isinstance(out, np.ndarray):
            raise TypeError("Expected out to be a numpy.ndarray")
        out_eval = out
        if out_eval.dtype != np.float64:
            raise TypeError(f"Expected out dtype float64, got {out_eval.dtype}")
        if out_eval.ndim != 1 or out_eval.shape[0] != expected_size:
            raise ValueError(f"Expected out to have shape ({expected_size},), got {out_eval.shape}")
        if not out_eval.flags.c_contiguous:
            raise ValueError("Expected out to be C-contiguous")
        self.layout.run_collocation_into(x_eval, out_eval)

    def build_coeffs(
        self, x: np.ndarray, *, include_none: bool = True
    ) -> dict[str, list[float] | None]:
        """Decode a packed state vector into a profile-coefficient dictionary."""
        blocks = decode_packed_blocks(
            x, self.plan.profile_L, self.plan.coeff_index, profile_names=self.plan.profile_names
        )
        coeffs: dict[str, list[float] | None] = {}
        for name, block in zip(self.plan.profile_names, blocks, strict=True):
            if include_none or block is not None:
                coeffs[name] = None if block is None else block.tolist()
        return coeffs

    def build_equilibrium(self, x: np.ndarray) -> Equilibrium:
        """Build a complete Equilibrium snapshot from a packed state vector."""
        x_eval = self.coerce_x(x)
        # Snapshotting intentionally runs the residual pipeline first: source
        # routes own alpha1/alpha2 and root fields, so decoding coefficients alone
        # is not enough to build a consistent Equilibrium.
        self.residual_var(x_eval)
        return self._snapshot_equilibrium_from_runtime(x_eval)

    def stage_a_profile(self, x: np.ndarray) -> None:
        """Run the profile stage and refresh active profile fields."""
        self.layout.run_profile(x)

    def stage_b_geometry(self) -> None:
        """Run the geometry stage and refresh geometry fields."""
        self.layout.run_geometry()

    def stage_c_source(self) -> None:
        """Run the source stage and refresh root fields and scale factors."""
        self.layout.run_source()

    def stage_d_residual(self) -> np.ndarray:
        """Run the residual stage and return the packed residual."""
        out = np.empty(self.plan.x_size, dtype=np.float64)
        self.stage_d_residual_into(out)
        return out

    def stage_d_residual_into(self, out: np.ndarray) -> None:
        """Run the residual stage and write the packed residual into ``out``."""
        if not isinstance(out, np.ndarray):
            raise TypeError("Expected out to be a numpy.ndarray")
        out_eval = out
        if out_eval.dtype != np.float64:
            raise TypeError(f"Expected out dtype float64, got {out_eval.dtype}")
        if out_eval.ndim != 1 or out_eval.shape[0] != self.plan.x_size:
            raise ValueError(
                f"Expected out to have shape ({self.plan.x_size},), got {out_eval.shape}"
            )
        if not out_eval.flags.c_contiguous:
            raise ValueError("Expected out to be C-contiguous")
        self.layout.run_residual_into(out_eval)

    def coerce_x(self, x: np.ndarray) -> np.ndarray:
        """Return a C-contiguous float64 packed state vector."""
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

    def _setup_runtime_state(self) -> None:
        (
            profiles_by_name,
            profile_workspace,
            geometry_workspace,
            source_workspace,
            residual_workspace,
        ) = allocate_runtime_state(
            grid_workspace=self.plan.grid_workspace,
            source_execution=self.plan.source_execution,
            profile_names=self.plan.profile_names,
            profile_index=self.plan.profile_index,
            active_profile_ids=self.plan.active_profile_ids,
            profile_L=self.plan.profile_L,
            x_size=self.plan.x_size,
            make_profile=lambda name: make_profile(
                case=self.case,
                operator_grid=self.plan.grid_workspace,
                name=name,
                profile_L=self.plan.profile_L,
                profile_names=self.plan.profile_names,
                profile_index=self.plan.profile_index,
                profile_static_kwargs_by_name=self.plan.profile_static_kwargs_by_name,
                profile_offset_specs=self.plan.profile_offset_specs,
            ),
        )
        self.profiles_by_name = profiles_by_name
        for name, profile in self.profiles_by_name.items():
            if hasattr(type(self), f"{name}_profile"):
                # Keep legacy attribute access (h_profile, psin_profile, ...)
                # as views of the runtime profile objects, not duplicated state.
                setattr(self, f"{name}_profile", profile)
        self.profile_workspace = profile_workspace
        self.geometry_workspace = geometry_workspace
        self.source_workspace = source_workspace
        self.residual_workspace = residual_workspace

    def _refresh_runtime_state(self) -> None:
        # Case replacement may change source route semantics but must preserve
        # the packed topology; compatibility was already checked by replace_case.
        self._apply_plan(
            refresh_operator_plan_for_case(
                self.plan,
                case=self.case,
                source_interpolation_kind=self.source_interpolation_kind,
            )
        )
        self._validate_runtime_profile_support()
        self._refresh_profile_runtime()
        self._refresh_fourier_family_metadata()
        # Source runtime refresh happens after profile metadata because some
        # routes decide whether psin is source-owned or optimized-profile-owned.
        refresh_source_runtime(
            case=self.case,
            grid_rho=self.plan.grid_workspace.rho,
            source_plan=self.plan.source_plan,
            source_execution=self.plan.source_execution,
            source_workspace=self.source_workspace,
            psin=self.residual_workspace.root_fields[0],
        )
        self._refresh_stage_a_runtime()
        self._refresh_runtime_bindings()

    def _refresh_profile_runtime(self) -> None:
        # Profiles are mutable runtime views over immutable plan topology.  The
        # refresh updates offsets, passive fields, and active metadata in place.
        refresh_profile_runtime(
            case=self.case,
            operator_grid=self.plan.grid_workspace,
            profile_names=self.plan.profile_names,
            profile_index=self.plan.profile_index,
            profile_L=self.plan.profile_L,
            profiles_by_name=self.profiles_by_name,
            profile_workspace=self.profile_workspace,
            profile_static_kwargs_by_name=self.plan.profile_static_kwargs_by_name,
            profile_offset_specs=self.plan.profile_offset_specs,
        )

    def _refresh_runtime_bindings(self) -> None:
        self.layout = build_operator_layout(
            plan=self.plan,
            case=self.case,
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
        # Build all passive profile fields after binding so geometry/source
        # stages can read a complete profile table before the first residual.
        for p in fixed_profile_ids:
            # Passive profiles are not refreshed by Stage A because they have no
            # packed coefficients, so materialize them immediately after binding.
            self.profile_workspace.refresh_profile_fields(
                profile_id=int(p),
                profile=self.profiles_by_name[self.plan.profile_names[int(p)]],
                grid_workspace=self.plan.grid_workspace,
            )
        f_profile_id = self.plan.profile_index.get("F", -1)
        if f_profile_id >= 0 and not bool(self.plan.active_profile_mask[f_profile_id]):
            # Passive F is stored as F**2 for the same postprocess contract as
            # active F; convert once so source kernels always see F fields.
            self.layout.profile.run_postprocess()

    def _refresh_stage_a_runtime(self) -> None:
        profile_workspace = self.profile_workspace
        refresh_stage_a_runtime(
            active_profile_ids=self.plan.active_profile_ids,
            profile_names=self.plan.profile_names,
            profiles_by_name=self.profiles_by_name,
            profile_L=self.plan.profile_L,
            coeff_index=self.plan.coeff_index,
            active_offsets=profile_workspace.active_offsets,
            active_scales=profile_workspace.active_scales,
            active_lengths=profile_workspace.active_lengths,
            active_coeff_index_rows=profile_workspace.active_coeff_index_rows,
        )

    def _refresh_fourier_family_metadata(self) -> None:
        # Effective order is case-dependent: a coefficient-free high-order
        # profile with nonzero boundary offset still contributes to geometry.
        self.c_effective_order, self.s_effective_order = refresh_fourier_family_metadata(
            c_profile_names=self.plan.c_profile_names,
            s_profile_names=self.plan.s_profile_names,
            profile_coeffs=self.case.profile_coeffs,
            c_offsets=self.case.c_offsets,
            s_offsets=self.case.s_offsets,
            c_family_fields=self.profile_workspace.c_family_fields,
            s_family_fields=self.profile_workspace.s_family_fields,
        )

    def invalidate_source_state(self) -> None:
        """Invalidate cached source state when a route requires fixed-point psin."""
        if tuple(self.plan.source_execution.route_key) == ("PJ2", "psin", "uniform"):
            # Negative sentinel forces the next PJ2 fixed-point runner to seed
            # its query from the current psin profile instead of stale psin.
            self.source_workspace.psin_query.fill(-1.0)

    def _snapshot_equilibrium_from_runtime(self, x: np.ndarray) -> Equilibrium:
        root_fields = self.residual_workspace.root_fields
        return snapshot_equilibrium_from_runtime(
            x=x,
            case=self.case,
            grid=self.plan.grid_workspace.to_grid(),
            profile_L=self.plan.profile_L,
            coeff_index=self.plan.coeff_index,
            profile_names=self.plan.profile_names,
            shape_profile_names=self.plan.shape_profile_names,
            profile_index=self.plan.profile_index,
            profiles_by_name=self.profiles_by_name,
            psin=root_fields[0],
            FFn_psin=root_fields[3],
            Pn_psin=root_fields[4],
            psin_r=root_fields[1],
            psin_rr=root_fields[2],
            alpha1=self.alpha1,
            alpha2=self.alpha2,
        )


def _estimate_h0_from_case(case: "OperatorCase") -> float:
    """Return Shafranov-shift estimate h0, or 0.0 for uniform profiles.

    An H-mode pedestal is detected when ``d|heat_input|/dpsi`` changes
    from negative to positive in the outer 30 % of the profile (the
    pedestal rise).  If no pedestal is found but the profile has
    significant radial structure it is treated as core-peaked H-mode
    and the toroidal estimate is kept.  Only perfectly uniform
    profiles (Solovev-like, range / mean < 1e-4) return zero.
    """
    try:
        a = float(case.boundary.a)
        R0 = float(case.boundary.R0)
    except (AttributeError, TypeError, ValueError):
        return 0.0
    heat = getattr(case, "heat_input", None)
    if heat is None or not hasattr(heat, "__len__") or len(heat) < 6:
        return 0.66 * a / R0
    try:
        n = len(heat)
        # single-pass: min, max, mean of |heat|, plus edge-derivative scan
        v0 = float(heat[0])
        h_min = v0 if v0 >= 0.0 else -v0
        h_max = h_min
        total = h_min
        edge_start = int(n * 0.7)
        if edge_start < 3:
            edge_start = 3
        prev_edge = 0.0
        prev_edge_diff = 0.0
        edge_mean = 0.0
        has_pedestal = False
        for i in range(1, n):
            v = float(heat[i])
            av = v if v >= 0.0 else -v
            total += av
            if av < h_min:
                h_min = av
            if av > h_max:
                h_max = av
            if i >= edge_start:
                # A sign change in the outer derivative is a coarse pedestal
                # detector; it only affects the initial guess, not physics.
                edge_mean += av
                cur_diff = av - prev_edge
                if prev_edge_diff < 0.0 and cur_diff > 0.0:
                    has_pedestal = True
                prev_edge = av
                prev_edge_diff = cur_diff
        h_mean = total / float(n)
        if h_mean < 1.0e-30:
            return 0.0
        # uniform (Solovev-like) → zero
        if (h_max - h_min) / h_mean < 1.0e-4:
            return 0.0
        # pedestal or structured → toroidal estimate
        if has_pedestal:
            edge_mean /= float(n - edge_start)
            if edge_mean > 1.0e-30:
                return 0.66 * a / R0
        # Even without a resolved pedestal, any nonuniform source benefits from
        # the same small Shafranov-shift seed used for shaped cases.
        return 0.66 * a / R0  # structured, no clear pedestal
    except (TypeError, ValueError):
        return 0.66 * a / R0
