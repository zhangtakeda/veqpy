"""
Module: veqpy.kernels.numba_kernel.kernel

Role:
- Implement the private Numba backend behind the public ``Kernel`` wrapper.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import fields, replace
from time import perf_counter
from typing import Any

import numpy as np
from numpy.linalg import norm

from veqpy.kernels.boundary_materialization import materialize_kernel_boundary
from veqpy.kernels.initial import KernelInitial, materialize_initial_state
from veqpy.kernels.pareto import (
    KernelParetoSignature,
    ParetoResult,
    ParetoSample,
    normalize_pareto_candidates,
    normalize_pareto_metric,
    normalize_pareto_target,
    pareto_frontier,
    pareto_sample_complexity,
    pareto_shape_error,
    topology_from_pareto_signature,
)
from veqpy.kernels.types import (
    KernelBoundary,
    KernelConfig,
    KernelPrepareResult,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    SolveResult,
)
from veqpy.kernels.variant import build_kernel_variant_topology
from veqpy.model import Equilibrium, Grid

from .pareto_runtime import sample_r_surface
from .solver import NumbaSolver

_JVP_EPS_SCALE = float(np.sqrt(1.0e-12))
_JACOBIAN_REL_STEP = 1.0e-7
_ParetoProgressCallback = Callable[[str, int, int], None]


class _NumbaKernelImpl:
    """Stateful Numba-backed Kernel API handle using direct Numba runtime."""

    def __init__(
        self,
        *,
        topology: KernelTopology,
        recipe: KernelRecipe | None = None,
        config: KernelConfig | None = None,
    ) -> None:
        if not isinstance(topology, KernelTopology):
            raise TypeError(f"topology must be KernelTopology, got {type(topology).__name__}")
        self.topology = topology
        self.recipe = KernelRecipe(backend="numba", layout="degree") if recipe is None else recipe
        if not isinstance(self.recipe, KernelRecipe):
            raise TypeError(f"recipe must be KernelRecipe, got {type(self.recipe).__name__}")
        self._validate_numba_recipe(self.recipe)
        self.config = KernelConfig() if config is None else self._kernel_config(config)
        self._solver = NumbaSolver(topology)
        self.history: list[SolveResult] = []
        self.result: SolveResult | None = None
        self._last_boundary: KernelBoundary | None = None
        self._last_source: KernelSource | None = None
        self._prepare_result: KernelPrepareResult | None = None

    @property
    def x_size(self) -> int:
        return self.topology.x_size

    def prepare(self, *, force: bool = False, dry_run: bool = False) -> KernelPrepareResult:
        self._validate_numba_recipe(self.recipe)
        if self._prepare_result is not None and not force and not dry_run:
            return self._prepare_result
        if dry_run:
            return KernelPrepareResult(
                backend=self.recipe.backend,
                topology=self.topology,
                recipe=self.recipe,
                x_size=self.x_size,
                residual_size=self.x_size,
                prepared=False,
                dry_run=True,
                artifact=None,
                warmed=False,
                raw_norm=float("nan"),
            )
        boundary = _prepare_boundary(self.topology)
        source = _prepare_source(self.topology)
        _, raw = self._solver.prepare(boundary, source, self.config)
        prepared = KernelPrepareResult(
            backend=self.recipe.backend,
            topology=self.topology,
            recipe=self.recipe,
            x_size=self.x_size,
            residual_size=int(raw.size),
            prepared=True,
            dry_run=False,
            artifact=None,
            warmed=True,
            raw_norm=float(norm(raw)),
        )
        self._prepare_result = prepared
        return prepared

    def variant(
        self,
        *,
        h_count: int | None = None,
        v_count: int | None = None,
        kappa_count: int | None = None,
        psin_count: int | None = None,
        F_count: int | None = None,
        c_counts: tuple[int, ...] | None = None,
        s_counts: tuple[int, ...] | None = None,
    ) -> "_NumbaKernelImpl":
        plan = build_kernel_variant_topology(
            self.topology,
            h_count=h_count,
            v_count=v_count,
            kappa_count=kappa_count,
            psin_count=psin_count,
            F_count=F_count,
            c_counts=c_counts,
            s_counts=s_counts,
        )
        if not plan.contained:
            # The topology constructor normally catches this first because
            # variant preserves L_max/M_max/K_max. Keep the guard local so the
            # runtime cannot silently accept an over-capacity active topology.
            raise ValueError("variant counts exceed current Numba kernel capacity")
        if not self._solver.variant(plan.topology):
            self._solver = NumbaSolver(plan.topology)
        self.topology = plan.topology
        self._clear_variant_state()
        return self

    def solve(
        self,
        boundary: KernelBoundary,
        source: KernelSource,
        *,
        config: KernelConfig | None = None,
        case_name: str | None = None,
        x0: KernelInitial | None = None,
        **config_overrides: Any,
    ) -> SolveResult:
        elapsed_started = perf_counter()
        kernel_config = self._runtime_config(config, config_overrides)
        kernel_boundary = self._kernel_boundary(boundary)
        kernel_source = self._kernel_source(source, case_name=case_name)
        self._last_boundary = kernel_boundary
        self._last_source = kernel_source
        initial_x = (
            materialize_initial_state(x0, self.topology, self.recipe)
            if x0 is not None
            else self._warm_start_x(kernel_config)
        )
        preprocess_ms = (perf_counter() - elapsed_started) * 1000.0
        result = self._solver.solve(
            kernel_boundary,
            kernel_source,
            kernel_config,
            x0=initial_x,
            preprocess_ms=preprocess_ms,
            elapsed_started=elapsed_started,
        )
        append_started = perf_counter()
        self.history.append(result)
        append_ms = (perf_counter() - append_started) * 1000.0
        self.result = replace(
            result,
            elapsed_ms=(perf_counter() - elapsed_started) * 1000.0,
            postprocess_ms=result.postprocess_ms + append_ms,
        )
        self.history[-1] = self.result
        return self.result

    def pareto(
        self,
        boundary: KernelBoundary,
        source: KernelSource,
        *,
        candidates: Sequence[object] | object,
        config: KernelConfig | None = None,
        case_name: str | None = None,
        reference: SolveResult | None = None,
        target: str = "counts",
        metric: str = "rms",
        **config_overrides: Any,
    ) -> ParetoResult:
        progress_callback = config_overrides.pop("_progress_callback", None)
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("_progress_callback must be callable")
        target_name = normalize_pareto_target(target)
        metric_name = normalize_pareto_metric(metric)
        kernel_config = self._runtime_config(config, config_overrides)
        kernel_boundary = self._kernel_boundary(boundary)
        kernel_source = self._kernel_source(source, case_name=case_name)

        saved_topology = self.topology
        saved_result = self.result
        saved_history = list(self.history)
        saved_last_boundary = self._last_boundary
        saved_last_source = self._last_source
        saved_prepare_result = self._prepare_result

        try:
            candidate_signatures = normalize_pareto_candidates(saved_topology, candidates)
            self._activate_pareto_topology(saved_topology)
            if reference is None:
                self._emit_pareto_progress(progress_callback, "ref", 0, 0)
                reference_result = self._solve_pareto_case(
                    kernel_boundary,
                    kernel_source,
                    kernel_config,
                )
            else:
                reference_result = reference
                if int(reference_result.x.size) != int(saved_topology.x_size):
                    raise ValueError(
                        "reference result packed state does not match the active topology"
                    )
            if not reference_result.success:
                raise RuntimeError("Kernel.pareto reference solve did not converge")
            reference_R = sample_r_surface(
                self._solver.runtime,
                reference_result.x,
                kernel_boundary,
                kernel_source,
            )
            reference_signature = KernelParetoSignature.from_topology(saved_topology)
            reference_sample = self._pareto_sample_from_result(
                saved_topology,
                reference_signature,
                reference_result,
                shape_error=0.0,
            )
            self._emit_pareto_progress(
                progress_callback,
                "run",
                0,
                len(candidate_signatures),
            )
            samples = self._solve_pareto_signatures(
                candidate_signatures,
                capacity_topology=saved_topology,
                boundary=kernel_boundary,
                source=kernel_source,
                config=kernel_config,
                reference_R=reference_R,
                metric=metric_name,
                progress_callback=progress_callback,
                progress_phase="run",
                progress_completed=0,
                progress_total=len(candidate_signatures),
            )

            frontier = pareto_frontier((reference_sample, *samples), target=target_name)
            return ParetoResult(
                reference=reference_sample,
                samples=tuple(samples),
                frontier=frontier,
                target=target_name,
                metric=metric_name,
            )
        finally:
            self._activate_pareto_topology(saved_topology)
            self.result = saved_result
            self.history[:] = saved_history
            self._last_boundary = saved_last_boundary
            self._last_source = saved_last_source
            self._prepare_result = saved_prepare_result

    def _solve_pareto_signatures(
        self,
        signatures: Sequence[KernelParetoSignature],
        *,
        capacity_topology: KernelTopology,
        boundary: KernelBoundary,
        source: KernelSource,
        config: KernelConfig,
        reference_R: np.ndarray,
        metric: str,
        progress_callback: _ParetoProgressCallback | None = None,
        progress_phase: str = "run",
        progress_completed: int = 0,
        progress_total: int | None = None,
    ) -> list[ParetoSample]:
        samples: list[ParetoSample] = []
        total = len(signatures) if progress_total is None else int(progress_total)
        for signature in signatures:
            candidate_topology = topology_from_pareto_signature(capacity_topology, signature)
            self._activate_pareto_topology(candidate_topology)
            result = self._solve_pareto_case(
                boundary,
                source,
                config,
            )
            candidate_R = sample_r_surface(
                self._solver.runtime,
                result.x,
                boundary,
                source,
            )
            samples.append(
                self._pareto_sample_from_result(
                    candidate_topology,
                    signature,
                    result,
                    shape_error=pareto_shape_error(
                        reference_R,
                        candidate_R,
                        metric=metric,
                    ),
                )
            )
            self._emit_pareto_progress(
                progress_callback,
                progress_phase,
                progress_completed + len(samples),
                total,
            )
        return samples

    @staticmethod
    def _emit_pareto_progress(
        progress_callback: object,
        phase: str,
        completed: int,
        total: int,
    ) -> None:
        if progress_callback is None:
            return
        assert callable(progress_callback)
        progress_callback(phase, int(completed), max(int(total), 0))

    def residual(self, x: Any, boundary: KernelBoundary, source: KernelSource) -> np.ndarray:
        out = np.empty(self.x_size, dtype=np.float64)
        self.residual_into(out, x, boundary, source)
        return out

    def residual_into(
        self,
        out: np.ndarray,
        x: Any,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> None:
        packed_out = self._packed_output(out, (self.x_size,), "out")
        packed_x = self._packed_input(x, "x")
        kernel_boundary = self._kernel_boundary(boundary)
        kernel_source = self._kernel_source(source, case_name=None)
        self._last_boundary = kernel_boundary
        self._last_source = kernel_source
        self._solver.residual_into(packed_out, packed_x, kernel_boundary, kernel_source)

    def jvp(self, x: Any, v: Any, boundary: KernelBoundary, source: KernelSource) -> np.ndarray:
        out = np.empty(self.x_size, dtype=np.float64)
        self.jvp_into(out, x, v, boundary, source)
        return out

    def jvp_into(
        self,
        out: np.ndarray,
        x: Any,
        v: Any,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> None:
        packed_out = self._packed_output(out, (self.x_size,), "out")
        packed_x = self._packed_input(x, "x")
        packed_v = self._packed_input(v, "v")
        kernel_boundary = self._kernel_boundary(boundary)
        kernel_source = self._kernel_source(source, case_name=None)
        self._last_boundary = kernel_boundary
        self._last_source = kernel_source
        _jvp_into(packed_out, packed_x, packed_v, kernel_boundary, kernel_source, self._solver)

    def jacobian(self, x: Any, boundary: KernelBoundary, source: KernelSource) -> np.ndarray:
        out = np.empty((self.x_size, self.x_size), dtype=np.float64)
        self.jacobian_into(out, x, boundary, source)
        return out

    def jacobian_into(
        self,
        out: np.ndarray,
        x: Any,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> None:
        matrix_out = self._packed_output(out, (self.x_size, self.x_size), "out")
        packed_x = self._packed_input(x, "x")
        kernel_boundary = self._kernel_boundary(boundary)
        kernel_source = self._kernel_source(source, case_name=None)
        self._last_boundary = kernel_boundary
        self._last_source = kernel_source
        _jacobian_into(matrix_out, packed_x, kernel_boundary, kernel_source, self._solver)

    def build_equilibrium(
        self,
        x: Any | None = None,
        *,
        grid: Grid | None = None,
    ) -> Equilibrium:
        if self._last_boundary is None or self._last_source is None:
            raise RuntimeError("build_equilibrium requires a previous Kernel runtime case")
        if x is None:
            if self.result is None:
                raise RuntimeError("build_equilibrium(x=None) requires a previous solve result")
            packed_x = self.result.x
        else:
            packed_x = self._packed_input(x, "x")
        return self._solver.build_equilibrium(
            packed_x,
            self._last_boundary,
            self._last_source,
            grid=grid,
        )

    def clear(self) -> None:
        self.history.clear()
        self.result = None
        self._last_boundary = None
        self._last_source = None
        self._prepare_result = None

    def _clear_variant_state(self) -> None:
        self.result = None
        self._last_boundary = None
        self._last_source = None
        self._prepare_result = None

    def _activate_pareto_topology(self, topology: KernelTopology) -> None:
        if not self._solver.variant(topology):
            self._solver = NumbaSolver(topology)
        self.topology = topology

    def _solve_pareto_case(
        self,
        boundary: KernelBoundary,
        source: KernelSource,
        config: KernelConfig,
    ) -> SolveResult:
        return self._solver.solve(
            boundary,
            source,
            config,
            x0=None,
            elapsed_started=perf_counter(),
        )

    @staticmethod
    def _pareto_sample_from_result(
        topology: KernelTopology,
        signature: KernelParetoSignature,
        result: SolveResult,
        *,
        shape_error: float,
    ) -> ParetoSample:
        return ParetoSample(
            topology=topology,
            signature=signature,
            counts=int(topology.x_size),
            time=float(result.elapsed_ms),
            complexity=pareto_sample_complexity(result, topology),
            shape_error=float(shape_error),
            result=result,
        )

    def close(self) -> None:
        return None

    def _warm_start_x(self, config: KernelConfig) -> np.ndarray | None:
        if self.result is None or not config.continuation.startswith("warm"):
            return None
        if self.result.x.shape != (self.x_size,):
            return None
        return self.result.x.copy()

    def _runtime_config(
        self,
        config: KernelConfig | None,
        overrides: dict[str, Any],
    ) -> KernelConfig:
        kernel_config = self.config if config is None else self._kernel_config(config)
        if overrides:
            kernel_config = _config_with_overrides(kernel_config, **overrides)
        return kernel_config

    @staticmethod
    def _validate_numba_recipe(recipe: KernelRecipe) -> None:
        if recipe.backend != "numba":
            raise ValueError("Numba backend requires KernelRecipe backend='numba'")
        if recipe.layout != "degree":
            raise ValueError("Numba backend only supports KernelRecipe layout='degree'")

    @staticmethod
    def _kernel_config(config: KernelConfig) -> KernelConfig:
        if not isinstance(config, KernelConfig):
            raise TypeError(f"config must be KernelConfig, got {type(config).__name__}")
        return config

    @staticmethod
    def _kernel_boundary(boundary: KernelBoundary) -> KernelBoundary:
        if not isinstance(boundary, KernelBoundary):
            raise TypeError(f"boundary must be KernelBoundary, got {type(boundary).__name__}")
        return materialize_kernel_boundary(boundary, fit_backend="numba").boundary

    @staticmethod
    def _kernel_source(source: KernelSource, *, case_name: str | None) -> KernelSource:
        if not isinstance(source, KernelSource):
            raise TypeError(f"source must be KernelSource, got {type(source).__name__}")
        if case_name is None:
            return source
        return KernelSource(
            heat_profile=source.heat_profile,
            current_profile=source.current_profile,
            Ip=source.Ip,
            beta=source.beta,
            case_name=case_name,
        )

    def _packed_input(self, value: Any, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (self.x_size,):
            raise ValueError(f"{name} must have shape ({self.x_size},), got {array.shape}")
        return np.ascontiguousarray(array, dtype=np.float64)

    @staticmethod
    def _packed_output(out: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
        if not isinstance(out, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if out.dtype != np.float64:
            raise TypeError(f"{name} must have dtype float64")
        if out.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
        if not out.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        return out


def _config_with_overrides(config: KernelConfig, **overrides: Any) -> KernelConfig:
    field_names = {item.name for item in fields(config) if item.init}
    unknown = sorted(name for name in overrides if name not in field_names)
    if unknown:
        names = ", ".join(unknown)
        raise TypeError(f"Unsupported KernelConfig override(s): {names}")
    return replace(config, **overrides)


def _prepare_boundary(topology: KernelTopology) -> KernelBoundary:
    offset_size = max(1, int(topology.M_max) + 1)
    sine_offset_count = max(0, int(topology.M_max))
    return KernelBoundary(
        a=0.5,
        R0=1.0,
        Z0=0.0,
        B0=3.0,
        ka=1.0,
        c_offsets=np.zeros(offset_size, dtype=np.float64),
        s_offsets=np.zeros(sine_offset_count, dtype=np.float64),
    )


def _prepare_source(topology: KernelTopology) -> KernelSource:
    sample_count = int(topology.sample_count)
    heat_profile = np.full(sample_count, 1.0e6, dtype=np.float64)
    current_value = 1.0e6 if topology.route in {"PI", "PJ1", "PJ2"} else 1.0
    current_profile = np.full(sample_count, current_value, dtype=np.float64)
    return KernelSource(
        heat_profile=heat_profile,
        current_profile=current_profile,
        Ip=1.0e6,
        beta=0.5 if topology.source_uses_beta_constraint else np.nan,
    )


def _jvp_into(
    out: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    boundary: KernelBoundary,
    source: KernelSource,
    solver: NumbaSolver,
) -> None:
    v_norm = float(norm(v))
    if v_norm <= 0.0:
        out.fill(0.0)
        return
    eps = _JVP_EPS_SCALE * (1.0 + float(norm(x))) / v_norm
    x_plus = x + eps * v
    f_base = _raw_residual(x, boundary, source, solver)
    f_plus = _raw_residual(x_plus, boundary, source, solver)
    np.subtract(f_plus, f_base, out=out)
    out /= eps


def _jacobian_into(
    out: np.ndarray,
    x: np.ndarray,
    boundary: KernelBoundary,
    source: KernelSource,
    solver: NumbaSolver,
) -> None:
    x_plus = x.copy()
    f_base = _raw_residual(x, boundary, source, solver)
    f_plus = np.empty_like(f_base)
    for col, saved in enumerate(x):
        step = _JACOBIAN_REL_STEP * max(1.0, abs(float(saved)))
        x_plus[col] = saved + step
        solver.residual_into(f_plus, x_plus, boundary, source)
        x_plus[col] = saved
        out[:, col] = (f_plus - f_base) / step


def _raw_residual(
    x: np.ndarray,
    boundary: KernelBoundary,
    source: KernelSource,
    solver: NumbaSolver,
) -> np.ndarray:
    raw = np.empty(x.size, dtype=np.float64)
    solver.residual_into(raw, x, boundary, source)
    return raw
