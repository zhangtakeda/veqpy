"""Nanobind ABI lowering helpers for Kernel dataclasses."""

from __future__ import annotations

from typing import Any

import numpy as np

from veqpy.kernels.abi.source_semantics import MaterializedKernelSource
from veqpy.types import KernelBoundary, KernelConfig, SolveResult


def boundary_runtime_args(boundary: KernelBoundary) -> tuple[Any, ...]:
    return (
        boundary.a,
        boundary.R0,
        boundary.Z0,
        boundary.B0,
        boundary.ka,
        np.ascontiguousarray(boundary.c_offsets, dtype=np.float64),
        np.ascontiguousarray(boundary.s_offsets, dtype=np.float64),
    )


def source_runtime_args(source: MaterializedKernelSource) -> tuple[Any, ...]:
    return (
        source.scaled_heat,
        source.scaled_current,
        source.scaled_Ip,
        source.beta,
    )


def config_runtime_args(config: KernelConfig, *, x_size: int) -> tuple[Any, ...]:
    max_evaluations = x_size * x_size if config.max_evaluations is None else config.max_evaluations
    return (
        config.method_code,
        config.max_residual,
        max_evaluations,
        config.accepted_residual_factor,
        config.accepted_residual_floor,
        config.initial_code,
        config.continuation_code,
        config.norm_code,
        config.residual_normalization_floor,
        config.residual_normalization_max_ratio,
        config.residual_normalization_huber_tau,
        config.residual_normalization_probe_count,
        config.residual_normalization_probe_step,
        config.residual_normalization_sensitivity_lambda,
    )


def solve_result_from_native(value: Any) -> SolveResult:
    (
        elapsed_ms,
        success,
        info,
        nfev,
        njev,
        callbacks,
        jacobian_component_evaluations,
        jvp_evaluations,
        linear_iterations,
        raw_norm,
        scaled_norm,
        x,
        raw,
        scaled,
        alpha,
    ) = value
    return SolveResult(
        elapsed_ms=float(elapsed_ms),
        success=bool(success),
        info=int(info),
        nfev=int(nfev),
        njev=int(njev),
        callbacks=int(callbacks),
        jacobian_component_evaluations=int(jacobian_component_evaluations),
        jvp_evaluations=int(jvp_evaluations),
        linear_iterations=int(linear_iterations),
        raw_norm=float(raw_norm),
        scaled_norm=float(scaled_norm),
        x=np.array(x, dtype=np.float64, copy=True),
        raw=np.array(raw, dtype=np.float64, copy=True),
        scaled=np.array(scaled, dtype=np.float64, copy=True),
        alpha=np.array(alpha, dtype=np.float64, copy=True),
    )
