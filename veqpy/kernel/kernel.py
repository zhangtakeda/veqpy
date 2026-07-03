"""Public explicit Numba Kernel facade."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import numpy as np

from veqlib.facade import (
    KernelBoundary,
    KernelConfig,
    KernelRecipe,
    KernelSource,
    KernelTopology,
    SolveResult,
)
from veqpy.model import Equilibrium

from .solver import NumbaSolver


class NumbaKernel:
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

    @property
    def x_size(self) -> int:
        return self.topology.x_size

    def prepare(self, *, force: bool = False, dry_run: bool = False) -> None:
        del force, dry_run
        self._validate_numba_recipe(self.recipe)

    def solve(
        self,
        boundary: KernelBoundary,
        source: KernelSource,
        *,
        config: KernelConfig | None = None,
        case_name: str | None = None,
        **config_overrides: Any,
    ) -> SolveResult:
        kernel_config = self._runtime_config(config, config_overrides)
        kernel_boundary = self._kernel_boundary(boundary)
        kernel_source = self._kernel_source(source, case_name=case_name)
        self._last_boundary = kernel_boundary
        self._last_source = kernel_source
        x0 = self._warm_start_x(kernel_config)
        self.result = self._solver.solve(kernel_boundary, kernel_source, kernel_config, x0=x0)
        self.history.append(self.result)
        return self.result

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
        del x, v, boundary, source
        raise NotImplementedError("NumbaKernel.jvp is not implemented")

    def jvp_into(
        self,
        out: np.ndarray,
        x: Any,
        v: Any,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> None:
        del out, x, v, boundary, source
        raise NotImplementedError("NumbaKernel.jvp_into is not implemented")

    def jacobian(self, x: Any, boundary: KernelBoundary, source: KernelSource) -> np.ndarray:
        del x, boundary, source
        raise NotImplementedError("NumbaKernel.jacobian is not implemented")

    def jacobian_into(
        self,
        out: np.ndarray,
        x: Any,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> None:
        del out, x, boundary, source
        raise NotImplementedError("NumbaKernel.jacobian_into is not implemented")

    def build_equilibrium(self, x: Any | None = None) -> Equilibrium:
        if self._last_boundary is None or self._last_source is None:
            raise RuntimeError("build_equilibrium requires a previous NumbaKernel runtime case")
        if x is None:
            if self.result is None:
                raise RuntimeError("build_equilibrium(x=None) requires a previous solve result")
            packed_x = self.result.x
        else:
            packed_x = self._packed_input(x, "x")
        return self._solver.build_equilibrium(packed_x, self._last_boundary, self._last_source)

    def clear(self) -> None:
        self.history.clear()
        self.result = None
        self._last_boundary = None
        self._last_source = None

    def close(self) -> None:
        return None

    def _warm_start_x(self, config: KernelConfig) -> np.ndarray | None:
        if self.result is not None and config.continuation.startswith("warm"):
            return self.result.x.copy()
        return None

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
            raise ValueError("NumbaKernel requires KernelRecipe backend='numba'")
        if recipe.layout != "degree":
            raise ValueError("NumbaKernel only supports KernelRecipe layout='degree'")

    @staticmethod
    def _kernel_config(config: KernelConfig) -> KernelConfig:
        if not isinstance(config, KernelConfig):
            raise TypeError(f"config must be KernelConfig, got {type(config).__name__}")
        return config

    @staticmethod
    def _kernel_boundary(boundary: KernelBoundary) -> KernelBoundary:
        if not isinstance(boundary, KernelBoundary):
            raise TypeError(f"boundary must be KernelBoundary, got {type(boundary).__name__}")
        return boundary

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
