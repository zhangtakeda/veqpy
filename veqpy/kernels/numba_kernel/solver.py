"""Low-level solve/residual adapter backed by direct Numba runtime."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares, root

from veqpy.kernels.types import (
    KernelBoundary,
    KernelConfig,
    KernelSource,
    KernelTopology,
    SolveResult,
)
from veqpy.model.equilibrium import Equilibrium

from .residual_scale import make_residual_scale
from .result import solve_result_from_runtime
from .runtime import NumbaRuntime


@dataclass(frozen=True, slots=True)
class _SolveOutcome:
    method: str
    x: np.ndarray
    success: bool
    raw_norm: float
    nfev: int
    njev: int
    iterations: int
    message: str
    error: Exception | None = None


class NumbaSolver:
    """Direct KernelTypes-to-Numba runtime solve adapter."""

    def __init__(self, topology: KernelTopology) -> None:
        self.topology = topology
        self.runtime = NumbaRuntime(topology)

    def residual_into(
        self,
        out: np.ndarray,
        x: np.ndarray,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> None:
        self.runtime.residual_into(out, x, boundary, source)

    def prepare(
        self,
        boundary: KernelBoundary,
        source: KernelSource,
        config: KernelConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = self.runtime.initial_state(
            boundary,
            source,
            initial=config.initial,
            x0=None,
        )
        raw = self.runtime.residual_for_current_case(x)
        return x.copy(), raw.copy()

    def solve(
        self,
        boundary: KernelBoundary,
        source: KernelSource,
        config: KernelConfig,
        *,
        x0: np.ndarray | None,
    ) -> SolveResult:
        x_guess = self.runtime.initial_state(
            boundary,
            source,
            initial=config.initial,
            x0=x0,
        )
        started = perf_counter()
        outcome = self._solve_once(x_guess, config)
        x_final = self.runtime.coerce_x(outcome.x).copy()
        nfev = int(outcome.nfev)
        njev = int(outcome.njev)
        iterations = int(outcome.iterations)

        raw = self.runtime.residual_for_current_case(x_final)
        alpha = self.runtime.alpha.copy()
        raw_norm = float(np.linalg.norm(raw))
        success = _residual_within_acceptance(raw_norm, config)
        elapsed_ms = (perf_counter() - started) * 1000.0
        return solve_result_from_runtime(
            x0=x_guess,
            x=x_final,
            raw=raw,
            alpha=alpha,
            success=success,
            nfev=nfev,
            njev=njev,
            iterations=iterations,
            elapsed_ms=elapsed_ms,
            runtime=self.runtime,
            config=config,
        )

    def _solve_once(self, x_guess: np.ndarray, config: KernelConfig) -> _SolveOutcome:
        return self._try_solve_once(x_guess, config, method=_solver_method(config))

    def _try_solve_once(
        self,
        x_guess: np.ndarray,
        config: KernelConfig,
        *,
        method: str,
    ) -> _SolveOutcome:
        x_initial = self.runtime.coerce_x(x_guess).copy()
        try:
            residual_fun, optimizer_x0, decode_x = self._optimizer_problem(
                x_initial,
                config,
                method=method,
            )
            opt = _run_optimizer(residual_fun, optimizer_x0, config, method=method)
            opt_x = decode_x(opt.x) if decode_x is not None else opt.x
            x_final = self.runtime.coerce_x(opt_x).copy()
            raw = self.runtime.residual_for_current_case(x_final)
            raw_norm = float(np.linalg.norm(raw))
            accepted = _residual_within_acceptance(raw_norm, config)
            message = str(getattr(opt, "message", ""))
            if bool(getattr(opt, "success", False)) and not accepted:
                message = f"{message} [rejected by residual={raw_norm:.6e}]"
            elif not bool(getattr(opt, "success", False)) and accepted:
                message = f"{message} [accepted by residual]"
            return _SolveOutcome(
                method=method,
                x=x_final,
                success=accepted,
                raw_norm=raw_norm,
                nfev=_count_opt_attr(opt, "nfev"),
                njev=_count_opt_attr(opt, "njev"),
                iterations=_count_opt_attr(opt, "nit"),
                message=message,
            )
        except Exception as exc:
            raw_norm = _safe_raw_norm(self.runtime, x_initial)
            accepted = _residual_within_acceptance(raw_norm, config)
            return _SolveOutcome(
                method=method,
                x=x_initial,
                success=accepted,
                raw_norm=raw_norm,
                nfev=0,
                njev=0,
                iterations=0,
                message=f"{type(exc).__name__}: {exc}",
                error=None if accepted else exc,
            )

    def build_equilibrium(
        self,
        x: np.ndarray,
        boundary: KernelBoundary,
        source: KernelSource,
    ) -> Equilibrium:
        return self.runtime.build_equilibrium(x, boundary, source)

    def _residual_function(self, x_reference: np.ndarray, config: KernelConfig):
        if config.norm == "none":
            return self.runtime.residual_for_current_case

        reference_raw = self.runtime.residual_for_current_case(x_reference)
        scale = _reference_residual_scale(
            reference_raw,
            x_reference,
            self.runtime,
            config,
        )
        if scale is None:
            return self.runtime.residual_for_current_case

        scale_eval = np.asarray(scale, dtype=np.float64)

        def residual_fun(x: np.ndarray) -> np.ndarray:
            return self.runtime.residual_for_current_case(x) / scale_eval

        return residual_fun

    def _optimizer_problem(self, x_initial: np.ndarray, config: KernelConfig, *, method: str):
        residual_fun = self._residual_function(x_initial, config)
        if method != "hybr":
            return residual_fun, x_initial, None

        x_transform_fun, optimizer_x0, decode_x = _build_x_transform_wrapper(
            self.runtime,
            x_initial,
        )
        if x_transform_fun is None:
            return residual_fun, x_initial, None

        def transformed_residual_fun(z: np.ndarray) -> np.ndarray:
            return residual_fun(x_transform_fun(z))

        return transformed_residual_fun, optimizer_x0, decode_x


def _run_optimizer(residual_fun, x_guess: np.ndarray, config: KernelConfig, *, method: str):
    if method == "hybr":
        return root(
            residual_fun,
            x_guess,
            method="hybr",
            tol=float(config.max_residual),
            options=_root_options(config, default_max_evaluations=x_guess.size * x_guess.size),
        )
    return least_squares(
        residual_fun,
        x_guess,
        method=method,
        **_least_squares_kwargs(
            config,
            method=method,
            default_max_evaluations=x_guess.size * x_guess.size,
        ),
    )


def _solver_method(config: KernelConfig) -> str:
    if config.method == "powell":
        return "hybr"
    if config.method == "levenberg-marquardt":
        return "lm"
    raise NotImplementedError(
        f"Numba backend does not support KernelConfig.method={config.method!r}"
    )


def _root_options(config: KernelConfig, *, default_max_evaluations: int) -> dict[str, object]:
    options: dict[str, object] = {"eps": 1.0e-6}
    max_evaluations = _max_evaluations(config, default=default_max_evaluations)
    if max_evaluations > 0:
        options["maxfev"] = max_evaluations
    if config.norm != "none":
        options["factor"] = 1.0
    return options


def _least_squares_kwargs(
    config: KernelConfig,
    *,
    method: str,
    default_max_evaluations: int,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "ftol": float(config.max_residual),
        "xtol": float(config.max_residual),
        "gtol": float(config.max_residual),
    }
    max_evaluations = _max_evaluations(config, default=default_max_evaluations)
    if max_evaluations > 0:
        kwargs["max_nfev"] = max_evaluations
    if config.norm != "none" and method == "lm":
        kwargs["x_scale"] = 1.0
    return kwargs


def _max_evaluations(config: KernelConfig, *, default: int) -> int:
    if config.max_evaluations is None:
        return int(default)
    return int(config.max_evaluations)


def _count_opt_attr(opt, name: str) -> int:
    value = getattr(opt, name, 0)
    if value is None:
        return 0
    return int(value)


def _safe_raw_norm(runtime: NumbaRuntime, x: np.ndarray) -> float:
    try:
        return float(np.linalg.norm(runtime.residual_for_current_case(x)))
    except Exception:
        return float("inf")


def _residual_within_acceptance(residual_norm: float, config: KernelConfig) -> bool:
    accepted = max(
        float(config.max_residual) * float(config.accepted_residual_factor),
        float(config.accepted_residual_floor),
    )
    return bool(np.isfinite(residual_norm) and residual_norm <= accepted)


def _reference_residual_scale(
    reference_raw: np.ndarray,
    x_reference: np.ndarray,
    runtime: NumbaRuntime,
    config: KernelConfig,
) -> np.ndarray | None:
    params: dict[str, object] = {}
    if config.norm in {"balanced", "safe"}:
        params.update(
            floor=config.residual_normalization_floor,
            max_ratio=config.residual_normalization_max_ratio,
            huber_tau=config.residual_normalization_huber_tau,
        )
    if config.norm == "safe":
        params.update(
            residual_fun=runtime.residual_for_current_case,
            x_guess=x_reference,
            x_scale=_x_scale_for_reference(runtime, x_reference),
            probe_count=config.residual_normalization_probe_count,
            probe_step=config.residual_normalization_probe_step,
            sensitivity_lambda=config.residual_normalization_sensitivity_lambda,
        )
    scale = make_residual_scale(
        config.norm,
        reference_raw,
        runtime.residual_block_lengths(),
        **params,
    )
    if scale is None:
        return None
    return np.asarray(scale, dtype=np.float64)


def _x_scale_for_reference(runtime: NumbaRuntime, x_reference: np.ndarray) -> np.ndarray | None:
    x_eval = np.asarray(x_reference, dtype=np.float64)
    scale = np.ones_like(x_eval)
    for _, profile_name, coeff_indices, offset, profile_scale in runtime.active_profile_blocks():
        coeff_indices = np.asarray(coeff_indices, dtype=np.int64)
        length = int(coeff_indices.size)
        if length <= 0:
            continue
        if np.any(coeff_indices < 0) or np.any(coeff_indices >= x_eval.size):
            return None
        block_guess = x_eval[coeff_indices]
        guess_rms = float(np.linalg.norm(block_guess) / np.sqrt(length))
        offset_scale = 0.0 if profile_name in {"h", "v", "psin"} else abs(float(offset))
        family_scale = abs(float(profile_scale))
        family_prior = _x_scale_profile_prior(profile_name)
        if abs(family_scale - 1.0) <= 1.0e-12:
            family_scale = family_prior
        scale[coeff_indices] = max(offset_scale, family_scale, family_prior, guess_rms, 1.0e-2)
    return scale


def _build_x_transform_wrapper(
    runtime: NumbaRuntime,
    x_guess: np.ndarray,
):
    x_eval = runtime.coerce_x(x_guess)
    x_scale = _x_scale_for_reference(runtime, x_eval)
    if x_scale is None:
        return None, x_eval, None

    inv_scale = 1.0 / x_scale
    x_buffer = np.empty_like(x_eval)

    def map_z_to_x(z: np.ndarray) -> np.ndarray:
        z_eval = np.asarray(z, dtype=np.float64)
        if z_eval.ndim != 1 or z_eval.shape[0] != x_eval.shape[0]:
            raise ValueError(f"Expected z to have shape {x_eval.shape}, got {z_eval.shape}")
        np.multiply(z_eval, x_scale, out=x_buffer)
        return x_buffer

    return map_z_to_x, x_eval * inv_scale, map_z_to_x


def _x_scale_profile_prior(name: str) -> float:
    if name in {"h", "v", "psin"}:
        return 1.5e-1
    if name == "k":
        return 1.0
    if name.startswith(("c", "s")):
        return 5.0e-2
    if name == "F":
        return 2.5e-1
    return 5.0e-2
