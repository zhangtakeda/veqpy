"""Low-level NumbaKernel solve/residual facade backed by legacy VEQPy."""

from __future__ import annotations

import numpy as np

from veqlib.facade import KernelBoundary, KernelConfig, KernelSource, KernelTopology, SolveResult
from veqpy.model import Equilibrium
from veqpy.solver import Solver, SolverConfig

from ._compat_lowering import build_legacy_operator
from .result import solve_result_from_legacy


class NumbaSolver:
    """Temporary bridge from KernelTypes to legacy Operator/Solver."""

    def __init__(self, topology: KernelTopology) -> None:
        self.topology = topology

    def residual_into(
        self,
        out: np.ndarray,
        x: np.ndarray,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> None:
        operator = build_legacy_operator(self.topology, boundary, source)
        operator.residual_var_into(x, out)

    def solve(
        self,
        boundary: KernelBoundary,
        source: KernelSource,
        config: KernelConfig,
        *,
        x0: np.ndarray | None,
    ) -> SolveResult:
        operator = build_legacy_operator(self.topology, boundary, source)
        legacy_config = _legacy_solver_config(config, x_size=operator.x_size)
        solver = Solver(operator=operator, config=legacy_config)
        solver.solve(x0=x0)
        if solver.result is None:
            raise RuntimeError("legacy Solver did not produce a result")
        return solve_result_from_legacy(solver.result, operator, config)

    def build_equilibrium(
        self,
        x: np.ndarray,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> Equilibrium:
        operator = build_legacy_operator(self.topology, boundary, source)
        return operator.build_equilibrium(x)


def _legacy_solver_config(config: KernelConfig, *, x_size: int) -> SolverConfig:
    method = _legacy_method(config.method)
    return SolverConfig(
        method=method,
        max_residual=config.max_residual,
        max_evaluations=(
            x_size * x_size if config.max_evaluations is None else config.max_evaluations
        ),
        initial_policy=_legacy_initial_policy(config.initial),
        enable_fallback=method == "hybr",
        residual_normalization=config.norm,
        residual_normalization_floor=config.residual_normalization_floor,
        residual_normalization_max_ratio=config.residual_normalization_max_ratio,
        residual_normalization_huber_tau=config.residual_normalization_huber_tau,
        residual_normalization_probe_count=config.residual_normalization_probe_count,
        residual_normalization_probe_step=config.residual_normalization_probe_step,
        residual_normalization_sensitivity_lambda=config.residual_normalization_sensitivity_lambda,
    )


def _legacy_method(method: str) -> str:
    if method == "powell":
        return "hybr"
    if method == "levenberg-marquardt":
        return "lm"
    raise NotImplementedError(
        f"NumbaKernel legacy lowering does not support KernelConfig.method={method!r}"
    )


def _legacy_initial_policy(initial: str) -> str:
    if initial == "cold-zeros":
        return "zeros"
    if initial == "cold-geometric":
        return "geometric-refined"
    if initial == "cold":
        return "auto"
    raise NotImplementedError(
        f"NumbaKernel legacy lowering does not support KernelConfig.initial={initial!r}"
    )
