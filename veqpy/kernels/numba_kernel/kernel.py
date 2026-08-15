"""
Module: veqpy.kernels.numba_kernel.kernel

Role:
- Implement the private Numba backend behind the public ``Kernel`` wrapper.
"""

from __future__ import annotations

from dataclasses import fields, replace
from time import perf_counter
from typing import Any

import numpy as np
from fusionprime_base import Equilibrium
from numpy.linalg import norm

from veqpy.kernels.abi.enums import (
    PRESSURE_DERIVATIVE_BY_COORDINATE,
    source_driver_for,
)
from veqpy.kernels.boundary_materialization import materialize_kernel_boundary
from veqpy.kernels.types import (
    KernelTopology,
    _BackendConfig,
    _BoundaryCase,
    _BuildPolicy,
    _PrepareDiagnostics,
    _SolveSnapshot,
    _SourceCase,
)
from veqpy.numerics.grid import Grid

from .solver import NumbaSolver
from .state import coerce_initial_state

_JVP_EPS_SCALE = float(np.sqrt(1.0e-12))
_JACOBIAN_REL_STEP = 1.0e-7
class _NumbaKernelImpl:
    """Stateful Numba-backed Kernel API handle using direct Numba runtime."""

    def __init__(
        self,
        *,
        topology: KernelTopology,
        recipe: _BuildPolicy | None = None,
        config: _BackendConfig | None = None,
    ) -> None:
        if not isinstance(topology, KernelTopology):
            raise TypeError(f"topology must be KernelTopology, got {type(topology).__name__}")
        self.topology = topology
        self.recipe = (
            _BuildPolicy(backend="numba", layout="degree", build="numba")
            if recipe is None
            else recipe
        )
        if not isinstance(self.recipe, _BuildPolicy):
            raise TypeError(f"recipe must be _BuildPolicy, got {type(self.recipe).__name__}")
        self._validate_numba_recipe(self.recipe)
        self.config = _BackendConfig() if config is None else self._kernel_config(config)
        self._solver = NumbaSolver(topology)
        self._last_snapshot: _SolveSnapshot | None = None
        self._last_boundary: _BoundaryCase | None = None
        self._last_source: _SourceCase | None = None
        self._prepare_result: _PrepareDiagnostics | None = None

    @property
    def x_size(self) -> int:
        return self.topology.x_size

    def prepare(self, *, force: bool = False, dry_run: bool = False) -> _PrepareDiagnostics:
        self._validate_numba_recipe(self.recipe)
        if self._prepare_result is not None and not force and not dry_run:
            return self._prepare_result
        if dry_run:
            return _PrepareDiagnostics(
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
        prepared = _PrepareDiagnostics(
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

    def solve(
        self,
        boundary: _BoundaryCase,
        source: _SourceCase,
        *,
        config: _BackendConfig | None = None,
        case_name: str | None = None,
        x0: np.ndarray | None = None,
        **config_overrides: Any,
    ) -> _SolveSnapshot:
        elapsed_started = perf_counter()
        kernel_config = self._runtime_config(config, config_overrides)
        kernel_boundary = self._kernel_boundary(boundary)
        kernel_source = self._kernel_source(source, case_name=case_name)
        self._last_boundary = kernel_boundary
        self._last_source = kernel_source
        initial_x = (
            coerce_initial_state(x0, self.x_size)
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
        self._last_snapshot = replace(
            result,
            elapsed_ms=(perf_counter() - elapsed_started) * 1000.0,
            postprocess_ms=result.postprocess_ms,
        )
        return self._last_snapshot

    def residual(self, x: Any, boundary: _BoundaryCase, source: _SourceCase) -> np.ndarray:
        out = np.empty(self.x_size, dtype=np.float64)
        self.residual_into(out, x, boundary, source)
        return out

    def residual_into(
        self,
        out: np.ndarray,
        x: Any,
        boundary: _BoundaryCase,
        source: _SourceCase,
    ) -> None:
        packed_out = self._packed_output(out, (self.x_size,), "out")
        packed_x = self._packed_input(x, "x")
        kernel_boundary = self._kernel_boundary(boundary)
        kernel_source = self._kernel_source(source, case_name=None)
        self._last_boundary = kernel_boundary
        self._last_source = kernel_source
        self._solver.residual_into(packed_out, packed_x, kernel_boundary, kernel_source)

    def jvp(self, x: Any, v: Any, boundary: _BoundaryCase, source: _SourceCase) -> np.ndarray:
        out = np.empty(self.x_size, dtype=np.float64)
        self.jvp_into(out, x, v, boundary, source)
        return out

    def jvp_into(
        self,
        out: np.ndarray,
        x: Any,
        v: Any,
        boundary: _BoundaryCase,
        source: _SourceCase,
    ) -> None:
        packed_out = self._packed_output(out, (self.x_size,), "out")
        packed_x = self._packed_input(x, "x")
        packed_v = self._packed_input(v, "v")
        kernel_boundary = self._kernel_boundary(boundary)
        kernel_source = self._kernel_source(source, case_name=None)
        self._last_boundary = kernel_boundary
        self._last_source = kernel_source
        _jvp_into(packed_out, packed_x, packed_v, kernel_boundary, kernel_source, self._solver)

    def jacobian(self, x: Any, boundary: _BoundaryCase, source: _SourceCase) -> np.ndarray:
        out = np.empty((self.x_size, self.x_size), dtype=np.float64)
        self.jacobian_into(out, x, boundary, source)
        return out

    def jacobian_into(
        self,
        out: np.ndarray,
        x: Any,
        boundary: _BoundaryCase,
        source: _SourceCase,
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
            if self._last_snapshot is None:
                raise RuntimeError("build_equilibrium(x=None) requires a previous solve result")
            packed_x = self._last_snapshot.x
        else:
            packed_x = self._packed_input(x, "x")
        return self._solver.build_equilibrium(
            packed_x,
            self._last_boundary,
            self._last_source,
            grid=grid,
        )

    def clear(self) -> None:
        self._last_snapshot = None
        self._last_boundary = None
        self._last_source = None
        self._prepare_result = None

    def close(self) -> None:
        return None

    def _warm_start_x(self, config: _BackendConfig) -> np.ndarray | None:
        if self._last_snapshot is None or not config.continuation.startswith("warm"):
            return None
        if self._last_snapshot.x.shape != (self.x_size,):
            return None
        return self._last_snapshot.x.copy()

    def _runtime_config(
        self,
        config: _BackendConfig | None,
        overrides: dict[str, Any],
    ) -> _BackendConfig:
        kernel_config = self.config if config is None else self._kernel_config(config)
        if overrides:
            kernel_config = _config_with_overrides(kernel_config, **overrides)
        return kernel_config

    @staticmethod
    def _validate_numba_recipe(recipe: _BuildPolicy) -> None:
        if recipe.backend != "numba":
            raise ValueError("Numba backend requires _BuildPolicy backend='numba'")
        if recipe.layout != "degree":
            raise ValueError("Numba backend only supports _BuildPolicy layout='degree'")

    @staticmethod
    def _kernel_config(config: _BackendConfig) -> _BackendConfig:
        if not isinstance(config, _BackendConfig):
            raise TypeError(f"config must be _BackendConfig, got {type(config).__name__}")
        return config

    @staticmethod
    def _kernel_boundary(boundary: _BoundaryCase) -> _BoundaryCase:
        if not isinstance(boundary, _BoundaryCase):
            raise TypeError(f"boundary must be _BoundaryCase, got {type(boundary).__name__}")
        return materialize_kernel_boundary(boundary, fit_backend="numba").boundary

    @staticmethod
    def _kernel_source(source: _SourceCase, *, case_name: str | None) -> _SourceCase:
        if not isinstance(source, _SourceCase):
            raise TypeError(f"source must be _SourceCase, got {type(source).__name__}")
        if case_name is None:
            return source
        return replace(source, case_name=case_name)

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


def _config_with_overrides(config: _BackendConfig, **overrides: Any) -> _BackendConfig:
    field_names = {item.name for item in fields(config) if item.init}
    unknown = sorted(name for name in overrides if name not in field_names)
    if unknown:
        names = ", ".join(unknown)
        raise TypeError(f"Unsupported _BackendConfig override(s): {names}")
    return replace(config, **overrides)


def _prepare_boundary(topology: KernelTopology) -> _BoundaryCase:
    offset_size = max(1, int(topology.M_max) + 1)
    sine_offset_count = max(0, int(topology.M_max))
    return _BoundaryCase(
        a=0.5,
        R0=1.0,
        Z0=0.0,
        B0=3.0,
        ka=1.0,
        c_offsets=np.zeros(offset_size, dtype=np.float64),
        s_offsets=np.zeros(sine_offset_count, dtype=np.float64),
    )


def _prepare_source(topology: KernelTopology) -> _SourceCase:
    sample_count = 2
    pressure_derivative = np.full(sample_count, -1.0e6, dtype=np.float64)
    driver_value = 1.0e6 if topology.route in {"PI", "PJ1", "PJ2", "PJ3"} else 1.0
    driver = np.full(sample_count, driver_value, dtype=np.float64)
    return _SourceCase(
        **{
            PRESSURE_DERIVATIVE_BY_COORDINATE[topology.coordinate]: pressure_derivative,
            source_driver_for(topology.route, topology.coordinate): driver,
        },
        source_nodes=np.linspace(0.0, 1.0, sample_count, dtype=np.float64),
        Ip=1.0e6,
        beta=0.5 if topology.source_uses_beta_constraint else np.nan,
    )


def _jvp_into(
    out: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    boundary: _BoundaryCase,
    source: _SourceCase,
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
    boundary: _BoundaryCase,
    source: _SourceCase,
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
    boundary: _BoundaryCase,
    source: _SourceCase,
    solver: NumbaSolver,
) -> np.ndarray:
    raw = np.empty(x.size, dtype=np.float64)
    solver.residual_into(raw, x, boundary, source)
    return raw
