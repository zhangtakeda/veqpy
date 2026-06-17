"""
Module: operator.operator

Role:
- Connect problem, grid, model runtime, engine kernels, and packed layout.
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

from veqpy.engine.backend import JaxBackendOptions, UnsupportedBackendFeature, normalize_backend
from veqpy.layout.binding import build_operator_layout
from veqpy.layout.runtime import OperatorLayout
from veqpy.math.interpolate import SOURCE_INTERP_DEFAULT
from veqpy.model.equilibrium import Equilibrium
from veqpy.model.grid import Grid
from veqpy.model.problem import Problem
from veqpy.operator.build_plan import (
    OperatorBuildPlan,
    build_operator_plan,
    refresh_operator_plan_for_problem,
)
from veqpy.operator.initialize import (
    build_boundary_slope_initial_state as build_operator_initial_state,
)
from veqpy.operator.initialize import (
    build_legacy_boundary_slope_initial_state as build_legacy_operator_initial_state,
)
from veqpy.operator.packed_layout import (
    decode_packed_blocks,
    encode_packed_state,
)
from veqpy.operator.profile_runtime import (
    refresh_fourier_family_metadata,
    refresh_profile_runtime,
    refresh_stage_a_runtime,
    validate_problem_compatibility,
)
from veqpy.operator.snapshot import snapshot_equilibrium_from_runtime
from veqpy.operator.source_plan import (
    validate_source_inputs,
)
from veqpy.operator.source_runtime import refresh_source_runtime
from veqpy.workspace import allocate_runtime_state
from veqpy.workspace.geometry_workspace import GeometryWorkspace
from veqpy.workspace.profile_workspace import ProfileWorkspace
from veqpy.workspace.residual_workspace import ResidualWorkspace
from veqpy.workspace.source_workspace import SourceWorkspace


@dataclass(slots=True, init=False)
class Operator:
    """Encapsulate the residual evaluator for a fixed problem, grid, and runtime."""

    grid: InitVar[Grid]
    problem: Problem = field(repr=False)
    backend: str = field(init=False)
    backend_options: JaxBackendOptions | None = field(init=False, repr=False)
    fix_rho: float = 0.05
    source_interpolation_kind: str = SOURCE_INTERP_DEFAULT
    plan: OperatorBuildPlan = field(init=False, repr=False)
    profile_workspace: ProfileWorkspace = field(init=False, repr=False)
    geometry_workspace: GeometryWorkspace = field(init=False, repr=False)
    source_workspace: SourceWorkspace = field(init=False, repr=False)
    residual_workspace: ResidualWorkspace = field(init=False, repr=False)
    layout: OperatorLayout = field(init=False, repr=False)

    c_effective_order: int = field(init=False, repr=False)
    s_effective_order: int = field(init=False, repr=False)

    def __init__(
        self,
        grid: Grid,
        problem: Problem | None = None,
        *,
        case: Problem | None = None,
        backend: str = "numba",
        backend_options: JaxBackendOptions | None = None,
        fix_rho: float = 0.05,
        source_interpolation_kind: str = SOURCE_INTERP_DEFAULT,
    ) -> None:
        """Bind an operator problem.

        ``case`` is kept as a compatibility alias for existing callers while
        the public spelling migrates to ``problem``.
        """
        if problem is None:
            if case is None:
                raise TypeError("Operator requires a problem")
            problem = case
        elif case is not None:
            raise TypeError("Pass either problem or case, not both")
        self.problem = problem
        self.backend = normalize_backend(backend)
        self.backend_options = backend_options
        self.fix_rho = float(fix_rho)
        self.source_interpolation_kind = source_interpolation_kind
        self.__post_init__(grid)

    def __post_init__(self, grid: Grid) -> None:
        """Build layouts, allocate runtime buffers, and bind the problem.

        The input grid is lowered to a GridWorkspace snapshot at construction time;
        Operator does not read live Grid state afterwards.
        """
        self.plan = build_operator_plan(
            grid=grid,
            problem=self.problem,
            source_interpolation_kind=self.source_interpolation_kind,
        )
        self.layout = OperatorLayout.empty(self.plan.x_size)
        self._setup_runtime_state()
        self._refresh_runtime_state()

    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Call the main variational residual evaluation entrypoint."""
        return self.residual_var(x, *args, **kwargs)

    @property
    def alpha1(self) -> float:
        """Current source normalization owned by the source runtime state."""
        self._raise_if_jax_public_state("alpha1")
        return float(self.source_workspace.alpha_state[0])

    @alpha1.setter
    def alpha1(self, value: float) -> None:
        """Update the current source normalization in place."""
        self._raise_if_jax_public_state("alpha1")
        self.source_workspace.alpha_state[0] = float(value)

    @property
    def alpha2(self) -> float:
        """Flux/source normalization owned by the source runtime state."""
        self._raise_if_jax_public_state("alpha2")
        return float(self.source_workspace.alpha_state[1])

    @alpha2.setter
    def alpha2(self, value: float) -> None:
        """Update the flux/source normalization in place."""
        self._raise_if_jax_public_state("alpha2")
        self.source_workspace.alpha_state[1] = float(value)

    @property
    def case(self) -> Problem:
        """Compatibility alias for ``problem``."""
        return self.problem

    @case.setter
    def case(self, problem: Problem) -> None:
        self.replace_problem(problem)

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

    def build_boundary_slope_initial_state(self, *, include_active_psin: bool = True) -> np.ndarray:
        """Build a geometrically-motivated packed x0 in a single pass."""

        x = build_operator_initial_state(
            problem=self.problem,
            plan=self.plan,
            profile_workspace=self.profile_workspace,
            source_psin_target=(
                self._source_psin_target_for_initial_state if include_active_psin else None
            ),
        )
        self.invalidate_source_state()
        return x

    def build_legacy_boundary_slope_initial_state(self) -> np.ndarray:
        """Build the legacy geometrically-motivated packed x0."""

        x = build_legacy_operator_initial_state(
            problem=self.problem,
            plan=self.plan,
            profile_workspace=self.profile_workspace,
        )
        self.invalidate_source_state()
        return x

    def replace_problem(self, problem: Problem) -> None:
        """Replace the current problem without changing the packed layout."""
        validate_problem_compatibility(
            problem,
            profile_names=self.plan.profile_names,
            prefix_profile_names=self.plan.prefix_profile_names,
            profile_L=self.plan.profile_L,
            coeff_index=self.plan.coeff_index,
            order_offsets=self.plan.order_offsets,
            validate_source_inputs=lambda next_problem: validate_source_inputs(
                next_problem, self.plan.grid_workspace.Nr
            ),
        )
        self.problem = problem
        self._refresh_runtime_state()

    def replace_case(self, case: Problem) -> None:
        """Compatibility alias for ``replace_problem``."""
        self.replace_problem(case)

    def zero_state(self) -> np.ndarray:
        """Return the zero packed state for this operator topology."""
        return np.zeros(self.plan.x_size, dtype=np.float64)

    def pack_coefficients(self, coefficients: dict[str, object]) -> np.ndarray:
        """Pack named active profile coefficients into the operator state vector."""
        return encode_packed_state(
            coefficients,
            self.plan.profile_L,
            self.plan.coeff_index,
            profile_names=self.plan.profile_names,
        )

    def unpack_coefficients(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Decode a packed state vector into active profile coefficient arrays."""
        blocks = decode_packed_blocks(
            x, self.plan.profile_L, self.plan.coeff_index, profile_names=self.plan.profile_names
        )
        return {
            name: block
            for name, block in zip(self.plan.profile_names, blocks, strict=True)
            if block is not None
        }

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

    def residual_collocation(
        self,
        x: np.ndarray,
        *,
        check: bool = True,
    ) -> np.ndarray:
        """Return the quadrature-scaled pointwise collocation residual.

        This residual does not append a Galerkin/weak-form residual to an external
        objective. Instead, it directly constrains ``R/J * G`` at every collocation
        node, where ``G = J/R * GS_residual``. Square-root radial/poloidal quadrature
        weights provide the discrete least-squares scaling. The returned vector has
        shape ``(Nr * Nt,)``.
        """
        out = np.empty(self.plan.grid_workspace.Nr * self.plan.grid_workspace.Nt, dtype=np.float64)
        self.residual_collocation_into(x, out, check=check)
        return out

    def residual_collocation_into(
        self,
        x: np.ndarray,
        out: np.ndarray,
        *,
        check: bool = True,
    ) -> None:
        """Write the quadrature-scaled collocation residual into caller-provided ``out``."""
        if check:
            x_eval = self.coerce_x(x)
            expected_size = self.plan.grid_workspace.Nr * self.plan.grid_workspace.Nt
            if not isinstance(out, np.ndarray):
                raise TypeError("Expected out to be a numpy.ndarray")
            out_eval = out
            if out_eval.dtype != np.float64:
                raise TypeError(f"Expected out dtype float64, got {out_eval.dtype}")
            if out_eval.ndim != 1 or out_eval.shape[0] != expected_size:
                raise ValueError(
                    f"Expected out to have shape ({expected_size},), got {out_eval.shape}"
                )
            if not out_eval.flags.c_contiguous:
                raise ValueError("Expected out to be C-contiguous")
        else:
            x_eval = x
            out_eval = out
        self.layout.run_collocation_into(x_eval, out_eval)

    def build_coeffs(self, x: np.ndarray, *, include_none: bool = False) -> dict[str, list[float]]:
        """Decode a packed state vector into a profile-coefficient dictionary."""
        del include_none
        return {name: block.tolist() for name, block in self.unpack_coefficients(x).items()}

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
        )
        self.profile_workspace = profile_workspace
        self.geometry_workspace = geometry_workspace
        self.source_workspace = source_workspace
        self.residual_workspace = residual_workspace

    def _refresh_runtime_state(self) -> None:
        # Problem replacement may change source route semantics but must preserve
        # the packed topology; compatibility was already checked by replace_problem.
        self.plan = refresh_operator_plan_for_problem(
            self.plan,
            problem=self.problem,
            source_interpolation_kind=self.source_interpolation_kind,
        )
        self._refresh_profile_runtime()
        self._refresh_fourier_family_metadata()
        # Source runtime refresh happens after profile metadata because some
        # routes decide whether psin is source-owned or optimized-profile-owned.
        refresh_source_runtime(
            grid_rho=self.plan.grid_workspace.rho,
            source_plan=self.plan.source_plan,
            source_execution=self.plan.source_execution,
            source_workspace=self.source_workspace,
            psin=self.residual_workspace.root_fields[0],
        )
        self._refresh_stage_a_runtime()
        self._refresh_runtime_bindings()

    def _refresh_profile_runtime(self) -> None:
        # Profile metadata is flat plan state.  Refresh updates offsets, passive
        # fields, and active metadata in place without constructing Profile objects.
        refresh_profile_runtime(
            problem=self.problem,
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

    def _refresh_runtime_bindings(self) -> None:
        self.layout = build_operator_layout(
            plan=self.plan,
            problem=self.problem,
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
            backend=self.backend,
            backend_options=self.backend_options,
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
                offset=float(self.plan.profile_offsets[int(p)]),
                scale=float(self.plan.profile_scales[int(p)]),
                amplitude_power=float(self.plan.profile_amplitude_powers[int(p)]),
                coeff=None,
                grid_workspace=self.plan.grid_workspace,
            )

    def _refresh_stage_a_runtime(self) -> None:
        profile_workspace = self.profile_workspace
        refresh_stage_a_runtime(
            active_profile_ids=self.plan.active_profile_ids,
            profile_L=self.plan.profile_L,
            coeff_index=self.plan.coeff_index,
            profile_offsets=self.plan.profile_offsets,
            profile_scales=self.plan.profile_scales,
            profile_amplitude_powers=self.plan.profile_amplitude_powers,
            active_offsets=profile_workspace.active_offsets,
            active_scales=profile_workspace.active_scales,
            active_amplitude_powers=profile_workspace.active_amplitude_powers,
            active_lengths=profile_workspace.active_lengths,
            active_coeff_index_rows=profile_workspace.active_coeff_index_rows,
        )

    def _refresh_fourier_family_metadata(self) -> None:
        # Effective order is problem-dependent: a coefficient-free high-order
        # profile with nonzero boundary offset still contributes to geometry.
        self.c_effective_order, self.s_effective_order = refresh_fourier_family_metadata(
            c_profile_names=self.plan.c_profile_names,
            s_profile_names=self.plan.s_profile_names,
            profile_L=self.plan.profile_L,
            profile_index=self.plan.profile_index,
            c_offsets=self.problem.c_offsets,
            s_offsets=self.problem.s_offsets,
            c_family_fields=self.profile_workspace.c_family_fields,
            s_family_fields=self.profile_workspace.s_family_fields,
        )

    def invalidate_source_state(self) -> None:
        """Invalidate cached source state when a route requires fixed-point psin."""
        if tuple(self.plan.source_execution.route_key) == ("PJ2", "psin", "uniform"):
            # Negative sentinel forces the next PJ2 fixed-point runner to seed
            # its query from the current psin profile instead of stale psin.
            self.source_workspace.psin_query.fill(-1.0)

    def _source_psin_target_for_initial_state(self, x: np.ndarray) -> np.ndarray | None:
        """Return the source-owned psin target produced by one residual refresh."""

        if not bool(self.plan.source_execution.requires_optimized_psin_profile):
            return None
        target_root_fields = self.source_workspace.target_root_fields
        if target_root_fields.shape[1] == 0:
            return None
        self.layout.run_profile(x)
        self.layout.run_geometry()
        self.layout.run_source()
        return target_root_fields[0]

    def _raise_if_jax_public_state(self, name: str) -> None:
        if self.backend == "jax":
            raise UnsupportedBackendFeature(
                f"backend='jax' does not expose public state {name!r} until "
                "explicit host publication is implemented."
            )

    def _snapshot_equilibrium_from_runtime(self, x: np.ndarray) -> Equilibrium:
        root_fields = self.residual_workspace.root_fields
        return snapshot_equilibrium_from_runtime(
            x=x,
            problem=self.problem,
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
