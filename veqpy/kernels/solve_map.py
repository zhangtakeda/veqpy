"""Numerical derivatives of the complete VEQ Kernel solve map."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from veqpy.kernels.initial import KernelInitial
from veqpy.kernels.types import (
    KernelBoundary,
    KernelConfig,
    KernelSource,
    SolveResult,
)

if TYPE_CHECKING:
    from veqpy.kernels.kernel import Kernel

_BOUNDARY_TANGENT_NAMES = frozenset({"a", "R0", "Z0", "B0", "ka", "c_offsets", "s_offsets"})
_SOURCE_SCALAR_TANGENT_NAMES = frozenset({"p0", "Ip", "beta"})


def solve_jvp(
    kernel: Kernel,
    boundary: KernelBoundary,
    source: KernelSource,
    *,
    boundary_tangent: Mapping[str, Any] | None,
    source_tangent: Mapping[str, Any] | None,
    output: Callable[[Kernel, SolveResult], Any] | None,
    base_result: SolveResult | None,
    relative_step: float,
    config: KernelConfig | None,
    case_name: str | None,
    x0: KernelInitial | None,
    **config_overrides: Any,
) -> np.ndarray:
    """Evaluate a central finite-difference JVP without publishing trial solves."""

    if kernel.recipe.backend != "numba":
        raise NotImplementedError(
            "solve_jvp currently supports only backend='numba'; "
            "the Cxx continuation workspace cannot yet be transactionally restored"
        )

    boundary_direction = {} if boundary_tangent is None else dict(boundary_tangent)
    source_direction = {} if source_tangent is None else dict(source_tangent)
    step = _directional_step(
        boundary,
        source,
        boundary_direction,
        source_direction,
        relative_step=relative_step,
    )

    history_size = len(kernel._impl.history)
    saved_result = kernel._impl.result
    saved_boundary = kernel._impl._last_boundary
    saved_source = kernel._impl._last_source
    try:
        if base_result is None:
            base_result = kernel.solve(
                boundary,
                source,
                config=config,
                case_name=case_name,
                x0=x0,
                **config_overrides,
            )
        _require_differentiable_solve(base_result, label="base")
        if step is None:
            # Rebind the supplied base case before a materializing output
            # callback. A caller may provide a valid result after the handle
            # has evaluated another runtime case.
            kernel.residual(base_result.x, boundary, source)
            return np.zeros_like(_encode_solve_output(kernel, base_result, output))

        plus_boundary, plus_source = _perturbed_case(
            boundary,
            source,
            boundary_direction,
            source_direction,
            scale=step,
        )
        minus_boundary, minus_source = _perturbed_case(
            boundary,
            source,
            boundary_direction,
            source_direction,
            scale=-step,
        )
        plus = kernel.solve(
            plus_boundary,
            plus_source,
            config=config,
            case_name=case_name,
            x0=base_result.x,
            **config_overrides,
        )
        _require_differentiable_solve(plus, label="positive perturbation")
        plus_output = _encode_solve_output(kernel, plus, output)
        minus = kernel.solve(
            minus_boundary,
            minus_source,
            config=config,
            case_name=case_name,
            x0=base_result.x,
            **config_overrides,
        )
        _require_differentiable_solve(minus, label="negative perturbation")
        minus_output = _encode_solve_output(kernel, minus, output)
        if plus_output.shape != minus_output.shape:
            raise ValueError(
                "solve_jvp output shape changed between perturbations: "
                f"{plus_output.shape} != {minus_output.shape}"
            )
        return (plus_output - minus_output) / (2.0 * step)
    finally:
        del kernel._impl.history[history_size:]
        kernel._impl.result = saved_result
        kernel._impl._last_boundary = saved_boundary
        kernel._impl._last_source = saved_source


def _directional_step(
    boundary: KernelBoundary,
    source: KernelSource,
    boundary_tangent: Mapping[str, Any],
    source_tangent: Mapping[str, Any],
    *,
    relative_step: float,
) -> float | None:
    relative_step = float(relative_step)
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError(f"relative_step must be finite and positive, got {relative_step!r}")

    unknown_boundary = sorted(set(boundary_tangent) - _BOUNDARY_TANGENT_NAMES)
    if unknown_boundary:
        raise KeyError(f"unsupported KernelBoundary tangent fields: {unknown_boundary}")
    active_source_names = {
        source.pressure_name,
        source.driver_name,
        *_SOURCE_SCALAR_TANGENT_NAMES,
    }
    unknown_source = sorted(set(source_tangent) - active_source_names)
    if unknown_source:
        raise KeyError(f"unsupported or inactive KernelSource tangent fields: {unknown_source}")
    if source.pressure_name == "p" and "p0" in source_tangent:
        raise KeyError("p0 is derived and cannot be differentiated when pressure input is p")

    relative_rate = 0.0
    for name, direction in boundary_tangent.items():
        base = getattr(boundary, name)
        if base is None:
            raise ValueError(
                f"solve_jvp requires a parameterized KernelBoundary; field {name!r} is unavailable"
            )
        relative_rate = max(relative_rate, _relative_direction_rate(base, direction, name))
    for name, direction in source_tangent.items():
        if name == source.pressure_name:
            base = source.pressure_profile
        elif name == source.driver_name:
            base = source.driver_profile
        else:
            base = getattr(source, name)
        if base is None or (np.isscalar(base) and not np.isfinite(float(base))):
            raise ValueError(f"KernelSource field {name!r} is not an active finite input")
        relative_rate = max(relative_rate, _relative_direction_rate(base, direction, name))
    if relative_rate == 0.0:
        return None
    return relative_step / relative_rate


def _relative_direction_rate(base: Any, direction: Any, name: str) -> float:
    base_array = np.asarray(base, dtype=np.float64)
    direction_array = np.asarray(direction, dtype=np.float64)
    if direction_array.shape != base_array.shape:
        raise ValueError(
            f"tangent field {name!r} has shape {direction_array.shape}, expected {base_array.shape}"
        )
    if not np.all(np.isfinite(direction_array)):
        raise ValueError(f"tangent field {name!r} must contain only finite values")
    base_scale = max(1.0, float(np.max(np.abs(base_array), initial=0.0)))
    direction_scale = float(np.max(np.abs(direction_array), initial=0.0))
    return direction_scale / base_scale


def _perturbed_case(
    boundary: KernelBoundary,
    source: KernelSource,
    boundary_tangent: Mapping[str, Any],
    source_tangent: Mapping[str, Any],
    *,
    scale: float,
) -> tuple[KernelBoundary, KernelSource]:
    if boundary.a is None or boundary.R0 is None or boundary.Z0 is None or boundary.ka is None:
        raise ValueError(
            "solve_jvp requires a parameterized KernelBoundary; fit raw R/Z points first"
        )
    boundary_values: dict[str, Any] = {
        "a": boundary.a,
        "R0": boundary.R0,
        "Z0": boundary.Z0,
        "B0": boundary.B0,
        "ka": boundary.ka,
        "c_offsets": boundary.c_offsets,
        "s_offsets": boundary.s_offsets,
    }
    for name, direction in boundary_tangent.items():
        boundary_values[name] = _add_direction(boundary_values[name], direction, scale)
    perturbed_boundary = KernelBoundary(**boundary_values)

    source_values: dict[str, Any] = {
        source.pressure_name: source.pressure_profile,
        source.driver_name: source.driver_profile,
        "source_nodes": source.source_nodes,
        "Ip": source.Ip,
        "beta": source.beta,
        "case_name": source.case_name,
    }
    if source.pressure_name != "p":
        source_values["p0"] = source.p0
    for name, direction in source_tangent.items():
        source_values[name] = _add_direction(source_values[name], direction, scale)
    return perturbed_boundary, KernelSource(**source_values)


def _add_direction(base: Any, direction: Any, scale: float) -> Any:
    value = np.asarray(base, dtype=np.float64) + scale * np.asarray(direction, dtype=np.float64)
    if value.ndim == 0:
        return float(value)
    if isinstance(base, tuple):
        return tuple(float(item) for item in value)
    return value


def _encode_solve_output(
    kernel: Kernel,
    result: SolveResult,
    output: Callable[[Kernel, SolveResult], Any] | None,
) -> np.ndarray:
    value = result.x if output is None else output(kernel, result)
    encoded = np.asarray(value, dtype=np.float64)
    if encoded.ndim == 0:
        encoded = encoded.reshape(1)
    if not np.all(np.isfinite(encoded)):
        raise ValueError("solve_jvp output must contain only finite numeric values")
    return np.array(encoded, dtype=np.float64, copy=True, order="C")


def _require_differentiable_solve(result: SolveResult, *, label: str) -> None:
    if not result.success:
        raise RuntimeError(
            f"solve_jvp {label} did not converge "
            f"(info={result.info}, raw_norm={result.raw_norm:.6e})"
        )
