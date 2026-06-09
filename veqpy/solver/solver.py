"""
Module: solver.solver

Role:
- Execute the nonlinear solve lifecycle.
- Manage x0, history, and SolverResult packaging.

Public API:
- Solver

Notes:
- `Solver` is the solver-layer facade.
- Does not own packed layout/codecs, backend selection, or Stage A/B/C/D numerical kernels.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import replace
from time import perf_counter

import numpy as np
from rich.console import Console

from veqpy.model.equilibrium import Equilibrium
from veqpy.operator.operator import Operator
from veqpy.operator.operator_case import OperatorCase
from veqpy.solver.residual_scale import (
    _block_rms_values,
    _build_block_rms_scale,
    _mode_is_block_rms,
    _residual_rms,
    make_residual_scale,
)
from veqpy.solver.solver_config import (
    LEAST_SQUARES_METHODS,
    ROOT_METHODS,
    SUPPORTED_METHODS,
    OptimizeMethod,
    SolverConfig,
)
from veqpy.solver.solver_record import SolverRecord
from veqpy.solver.solver_result import SolverResult


class Solver:
    """Solve facade for a fixed packed layout."""

    def __init__(
        self,
        *,
        operator: Operator,
        config: SolverConfig | None = None,
    ) -> None:
        """Bind an Operator and one default solve configuration."""

        self.operator = operator
        self.config = SolverConfig() if config is None else config
        self.result: SolverResult | None = None
        self.history: list[SolverRecord] = []
        self.x0 = self.operator.encode_initial_state()

    def reset(self) -> None:
        """Zero the solver-owned x0 in place."""

        self.x0.fill(0.0)

    def clear(self) -> None:
        """Clear solve history without changing the current x0."""

        self.history.clear()

    def replace_case(self, case: OperatorCase) -> None:
        """Replace the case with a compatible one."""

        self.operator.replace_case(case)

    def solve(
        self,
        x0: np.ndarray | None = None,
        *,
        method: str | None = None,
        max_residual: float | None = None,
        max_evaluations: int | None = None,
        enable_warmstart: bool | None = None,
        initial_policy: str | None = None,
        initial_homothetic_lambda: float | None = None,
        enable_fallback: bool | None = None,
        fallback_methods: tuple[str, ...] | list[str] | None = None,
        enable_verbose: bool | None = None,
        enable_history: bool | None = None,
        residual_normalization: str | None = None,
        residual_normalization_floor: float | None = None,
        residual_normalization_max_ratio: float | None = None,
        residual_normalization_huber_tau: float | None = None,
        residual_normalization_probe_count: int | None = None,
        residual_normalization_probe_step: float | None = None,
        residual_normalization_sensitivity_lambda: float | None = None,
        enable_collocation: bool | None = None,
        collocation_method: str | None = None,
        collocation_weight: float | None = None,
        collocation_max_residual: float | None = None,
        collocation_max_evaluations: int | None = None,
    ) -> np.ndarray:
        """Execute one solve and return the converged packed x."""

        solve_config = self._resolve_solve_config(
            method=method,
            max_residual=max_residual,
            max_evaluations=max_evaluations,
            enable_warmstart=enable_warmstart,
            initial_policy=initial_policy,
            initial_homothetic_lambda=initial_homothetic_lambda,
            enable_fallback=enable_fallback,
            fallback_methods=fallback_methods,
            enable_verbose=enable_verbose,
            enable_history=enable_history,
            residual_normalization=residual_normalization,
            residual_normalization_floor=residual_normalization_floor,
            residual_normalization_max_ratio=residual_normalization_max_ratio,
            residual_normalization_huber_tau=residual_normalization_huber_tau,
            residual_normalization_probe_count=residual_normalization_probe_count,
            residual_normalization_probe_step=residual_normalization_probe_step,
            residual_normalization_sensitivity_lambda=residual_normalization_sensitivity_lambda,
            enable_collocation=enable_collocation,
            collocation_method=collocation_method,
            collocation_weight=collocation_weight,
            collocation_max_residual=collocation_max_residual,
            collocation_max_evaluations=collocation_max_evaluations,
        )
        _validate_stage_solve_config(solve_config, residual_kind="variational")

        if x0 is not None:
            self.x0 = self.operator.coerce_x(x0).copy()
        elif solve_config.initial_policy == "warm":
            self.x0 = self.x0.copy()
        else:
            self.x0 = _build_initial_state(self.operator, solve_config).copy()
        if x0 is not None or solve_config.initial_policy != "warm":
            self.operator.invalidate_source_state()

        x_guess = self.x0.copy()

        started = perf_counter()
        if solve_config.enable_collocation:
            (
                x_opt,
                success,
                message,
                function_evaluations,
                jacobian_evaluations,
                iterations,
                residual_norm_final,
            ) = self._solve_with_collocation_polish(
                x_guess,
                solve_config=solve_config,
                x0_was_provided=x0 is not None,
            )
        else:
            (
                x_opt,
                success,
                message,
                function_evaluations,
                jacobian_evaluations,
                iterations,
                residual_norm_final,
            ) = self._solve_with_fallbacks(
                x_guess,
                solve_config=solve_config,
                residual_kind="variational",
                x0_was_provided=x0 is not None,
            )
        elapsed = (perf_counter() - started) * 1e6

        x_final = self.operator.coerce_x(x_opt)
        residual_final_exc = None
        if not bool(success) and not np.isfinite(residual_norm_final):
            residual_norm_final, residual_final_exc = self._safe_residual_norm(
                x_final,
                solve_config=self._final_residual_config(solve_config),
                residual_kind=self._final_residual_kind(solve_config),
            )
        if residual_final_exc is not None:
            success = False
            message = (
                f"{message} [final residual evaluation failed: "
                f"{type(residual_final_exc).__name__}: {residual_final_exc}]"
            )

        self.result = SolverResult(
            x0=x_guess,
            x=x_final,
            success=bool(success),
            message=str(message),
            residual_norm_final=residual_norm_final,
            function_evaluations=int(function_evaluations),
            jacobian_evaluations=int(jacobian_evaluations),
            iterations=int(iterations),
            elapsed=elapsed,
        )

        record = SolverRecord(
            case_snapshot=self.operator.case.copy(),
            config_snapshot=solve_config,
            result_snapshot=self.result,
        )

        if solve_config.enable_verbose:
            Console().print(record)

        if solve_config.enable_history:
            self.history.append(record)

        self.x0 = x_final.copy()
        return x_final

    def build_coeffs(
        self,
        *,
        include_none: bool = True,
    ) -> dict[str, list[float] | None]:
        """Rebuild a profile-coefficient dictionary from the current solver-owned x0."""

        return self.operator.build_coeffs(self.x0, include_none=include_none)

    def build_equilibrium(self) -> Equilibrium:
        """Materialize an Equilibrium snapshot from the current solver-owned x0."""

        return self.operator.build_equilibrium(self.x0)

    def _resolve_solve_config(
        self,
        *,
        method: str | None,
        max_residual: float | None,
        max_evaluations: int | None,
        enable_warmstart: bool | None,
        initial_policy: str | None,
        initial_homothetic_lambda: float | None,
        enable_fallback: bool | None,
        fallback_methods: tuple[str, ...] | list[str] | None,
        enable_verbose: bool | None,
        enable_history: bool | None,
        residual_normalization: str | None,
        residual_normalization_floor: float | None,
        residual_normalization_max_ratio: float | None,
        residual_normalization_huber_tau: float | None,
        residual_normalization_probe_count: int | None,
        residual_normalization_probe_step: float | None,
        residual_normalization_sensitivity_lambda: float | None,
        enable_collocation: bool | None,
        collocation_method: str | None,
        collocation_weight: float | None,
        collocation_max_residual: float | None,
        collocation_max_evaluations: int | None,
    ) -> SolverConfig:
        """Build a temporary per-solve configuration snapshot from defaults."""

        overrides: dict[str, object] = {}
        if method is not None:
            overrides["method"] = str(method)
        if max_residual is not None:
            overrides["max_residual"] = float(max_residual)
        if max_evaluations is not None:
            overrides["max_evaluations"] = int(max_evaluations)
        if enable_warmstart is not None:
            overrides["enable_warmstart"] = bool(enable_warmstart)
        if initial_policy is not None:
            overrides["initial_policy"] = str(initial_policy)
        if initial_homothetic_lambda is not None:
            overrides["initial_homothetic_lambda"] = float(initial_homothetic_lambda)
        if enable_fallback is not None:
            overrides["enable_fallback"] = bool(enable_fallback)
        if fallback_methods is not None:
            overrides["fallback_methods"] = tuple(
                str(method_name) for method_name in fallback_methods
            )
        if enable_verbose is not None:
            overrides["enable_verbose"] = bool(enable_verbose)
        if enable_history is not None:
            overrides["enable_history"] = bool(enable_history)
        if residual_normalization is not None:
            overrides["residual_normalization"] = residual_normalization
        if residual_normalization_floor is not None:
            overrides["residual_normalization_floor"] = float(residual_normalization_floor)
        if residual_normalization_max_ratio is not None:
            overrides["residual_normalization_max_ratio"] = float(residual_normalization_max_ratio)
        if residual_normalization_huber_tau is not None:
            overrides["residual_normalization_huber_tau"] = float(residual_normalization_huber_tau)
        if residual_normalization_probe_count is not None:
            overrides["residual_normalization_probe_count"] = int(
                residual_normalization_probe_count
            )
        if residual_normalization_probe_step is not None:
            overrides["residual_normalization_probe_step"] = float(
                residual_normalization_probe_step
            )
        if residual_normalization_sensitivity_lambda is not None:
            overrides["residual_normalization_sensitivity_lambda"] = float(
                residual_normalization_sensitivity_lambda
            )
        if enable_collocation is not None:
            overrides["enable_collocation"] = bool(enable_collocation)
        if collocation_method is not None:
            overrides["collocation_method"] = str(collocation_method)
        if collocation_weight is not None:
            overrides["collocation_weight"] = float(collocation_weight)
        if collocation_max_residual is not None:
            overrides["collocation_max_residual"] = float(collocation_max_residual)
        if collocation_max_evaluations is not None:
            overrides["collocation_max_evaluations"] = int(collocation_max_evaluations)
        if not overrides:
            return self.config
        return replace(self.config, **overrides)

    def _solve_with_collocation_polish(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        x0_was_provided: bool,
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        """Run a variational solve first, then warm-start collocation polish from that result."""

        variational_config = self._variational_stage_config(solve_config)
        collocation_config = self._collocation_stage_config(solve_config)
        collocation_residual_kind = self._collocation_residual_kind(solve_config)
        _validate_stage_solve_config(variational_config, residual_kind="variational")
        _validate_stage_solve_config(collocation_config, residual_kind=collocation_residual_kind)

        variational_result = self._solve_with_fallbacks(
            x_guess,
            solve_config=variational_config,
            residual_kind="variational",
            x0_was_provided=x0_was_provided,
        )
        if float(solve_config.collocation_weight) <= 0.0:
            return self._combine_variational_collocation_results(
                variational_result=variational_result,
                collocation_result=(
                    variational_result[0].copy(),
                    variational_result[1],
                    "skipped because collocation_weight=0",
                    0,
                    0,
                    0,
                    variational_result[6],
                ),
                collocation_error=None,
            )
        collocation_result, collocation_error = self._try_solve_attempt(
            variational_result[0],
            solve_config=collocation_config,
            residual_kind=collocation_residual_kind,
        )
        if collocation_result is None:
            if collocation_error is not None:
                raise RuntimeError(
                    "Collocation polish failed without a usable result"
                ) from collocation_error
            raise RuntimeError("Collocation polish failed without a usable result")

        return self._combine_variational_collocation_results(
            variational_result=variational_result,
            collocation_result=collocation_result,
            collocation_error=collocation_error,
        )

    def _variational_stage_config(self, solve_config: SolverConfig) -> SolverConfig:
        """Return the variational-stage configuration for the two-stage workflow."""

        return replace(solve_config, enable_collocation=False)

    def _collocation_stage_config(self, solve_config: SolverConfig) -> SolverConfig:
        """Return the collocation-polish configuration for the two-stage workflow."""

        max_residual = (
            solve_config.max_residual
            if solve_config.collocation_max_residual is None
            else solve_config.collocation_max_residual
        )
        max_evaluations = (
            solve_config.max_evaluations
            if solve_config.collocation_max_evaluations is None
            else solve_config.collocation_max_evaluations
        )
        return replace(
            solve_config,
            method=solve_config.collocation_method,
            max_residual=max_residual,
            max_evaluations=max_evaluations,
            enable_collocation=False,
            enable_fallback=False,
            fallback_methods=(),
        )

    def _collocation_residual_kind(self, solve_config: SolverConfig) -> str:
        """Return the residual objective used by the collocation-polish stage."""

        collocation_weight = float(solve_config.collocation_weight)
        if collocation_weight >= 1.0:
            return "collocation"
        if collocation_weight <= 0.0:
            return "variational"
        return "blended_collocation"

    def _final_residual_config(self, solve_config: SolverConfig) -> SolverConfig:
        """Return the residual evaluation configuration used for the final SolverResult x."""

        if solve_config.enable_collocation:
            return self._collocation_stage_config(solve_config)
        return solve_config

    def _final_residual_kind(self, solve_config: SolverConfig) -> str:
        """Return the residual kind used for the final SolverResult x."""

        if solve_config.enable_collocation:
            return self._collocation_residual_kind(solve_config)
        return "variational"

    def _combine_variational_collocation_results(
        self,
        *,
        variational_result: tuple[np.ndarray, bool, str, int, int, int, float],
        collocation_result: tuple[np.ndarray, bool, str, int, int, int, float],
        collocation_error: Exception | None,
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        """Merge two-stage counters with collocation owning success and final x."""

        variational_status = "succeeded" if bool(variational_result[1]) else "failed"
        collocation_status = (
            "succeeded"
            if self._attempt_succeeded(collocation_result, collocation_error)
            else "failed"
        )
        collocation_failure = self._format_attempt_failure(
            method="collocation-polish",
            result=collocation_result,
            error=collocation_error,
        )
        message = (
            f"variational stage {variational_status}: {variational_result[2]}; "
            f"collocation polish {collocation_status}: {collocation_failure}"
        )
        return (
            collocation_result[0],
            self._attempt_succeeded(collocation_result, collocation_error),
            message,
            int(variational_result[3]) + int(collocation_result[3]),
            int(variational_result[4]) + int(collocation_result[4]),
            int(variational_result[5]) + int(collocation_result[5]),
            float(collocation_result[6]),
        )

    def _solve_with_fallbacks(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        residual_kind: str,
        x0_was_provided: bool,
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        """Solve with the primary method and fall back to configured backup methods if needed."""

        attempts: list[
            tuple[str, tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]
        ] = []

        attempt_plans = self._build_attempt_plans(
            x_guess,
            solve_config=solve_config,
            residual_kind=residual_kind,
            x0_was_provided=x0_was_provided,
        )

        for idx, attempt_plan in enumerate(attempt_plans):
            label, x_attempt_guess, attempt_config = attempt_plan
            result, error = self._try_solve_attempt(
                x_attempt_guess,
                solve_config=attempt_config,
                residual_kind=residual_kind,
            )
            attempts.append((label, result, error))
            if self._attempt_succeeded(result, error):
                if result is None:
                    raise RuntimeError("Solve attempt succeeded without a result")
                if idx == 0:
                    return result
                return self._finalize_attempts(attempts)

            if idx + 1 >= len(attempt_plans):
                break

            next_label = attempt_plans[idx + 1][0]
            failure = self._format_attempt_failure(
                method=label,
                result=result,
                error=error,
            )
            if solve_config.enable_verbose:
                warnings.warn(
                    (
                        f"Solve with method={label!r} failed ({failure}). "
                        f"Retrying with {next_label!r}."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )

        return self._finalize_attempts(attempts)

    def _build_attempt_plans(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        residual_kind: str,
        x0_was_provided: bool,
    ) -> list[tuple[str, np.ndarray, SolverConfig]]:
        x_initial = self.operator.coerce_x(x_guess).copy()
        attempt_plans = [
            (
                self._display_attempt_label(
                    solve_config,
                    start_kind=self._display_start_kind(
                        x_initial,
                        solve_config=solve_config,
                        x0_was_provided=x0_was_provided,
                    ),
                ),
                x_initial,
                solve_config,
            )
        ]

        seen_methods = {solve_config.method}
        for fallback_method in self._ordered_fallback_methods(solve_config):
            if fallback_method in seen_methods:
                continue
            seen_methods.add(fallback_method)
            fallback_config = replace(solve_config, method=fallback_method)
            attempt_plans.append(
                (
                    self._display_attempt_label(fallback_config, start_kind="warm-fallback"),
                    x_initial.copy(),
                    fallback_config,
                )
            )
        return attempt_plans

    def _ordered_fallback_methods(self, solve_config: SolverConfig) -> tuple[str, ...]:
        if not solve_config.enable_fallback:
            return ()
        return tuple(solve_config.fallback_methods)

    def _try_solve_attempt(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        residual_kind: str,
    ) -> tuple[tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]:
        """Wrap one solve stage so the fallback flow can also handle numerical exceptions."""

        x_guess_eval = self.operator.coerce_x(x_guess).copy()
        try:
            return self._solve_opt_problem(
                x_guess_eval,
                solve_config=solve_config,
                residual_kind=residual_kind,
            ), None
        except Exception as exc:
            residual_norm_x0, residual_exc = self._safe_residual_norm(
                x_guess_eval,
                solve_config=solve_config,
                residual_kind=residual_kind,
            )
            if residual_exc is None and _residual_within_acceptance(residual_norm_x0, solve_config):
                return (
                    (
                        x_guess_eval.copy(),
                        True,
                        f"{type(exc).__name__}: {exc} "
                        f"[accepted by x0 residual={residual_norm_x0:.6e}]",
                        0,
                        0,
                        0,
                        float(residual_norm_x0),
                    ),
                    None,
                )
            return (
                (
                    x_guess_eval.copy(),
                    False,
                    f"{type(exc).__name__}: {exc}",
                    0,
                    0,
                    0,
                    float("nan") if residual_exc is not None else float(residual_norm_x0),
                ),
                exc,
            )

    def _format_attempt_failure(
        self,
        *,
        method: str,
        result: tuple[np.ndarray, bool, str, int, int, int, float] | None,
        error: Exception | None,
    ) -> str:
        if error is not None:
            return f"{type(error).__name__}: {error}"
        if result is None:
            return f"method={method} produced no result"
        return result[2]

    def _attempt_residual_norm(
        self,
        attempt: tuple[np.ndarray, bool, str, int, int, int, float] | None,
    ) -> float:
        if attempt is None:
            return float("inf")
        residual_norm = float(attempt[6])
        if not np.isfinite(residual_norm):
            return float("inf")
        return residual_norm

    def _safe_residual_norm(
        self,
        x: np.ndarray,
        *,
        solve_config: SolverConfig | None = None,
        residual_kind: str = "variational",
    ) -> tuple[float, Exception | None]:
        try:
            config_eval = self.config if solve_config is None else solve_config
            residual_fun = self._residual_function_for(
                residual_kind,
                solve_config=config_eval,
                x_reference=x,
            )
            return _residual_array_norm(residual_fun(x)), None
        except Exception as exc:
            return float("inf"), exc

    def _attempt_succeeded(
        self,
        attempt: tuple[np.ndarray, bool, str, int, int, int, float] | None,
        error: Exception | None,
    ) -> bool:
        return bool(
            error is None
            and attempt is not None
            and bool(attempt[1])
            and np.isfinite(float(attempt[6]))
        )

    def _display_attempt_label(self, solve_config: SolverConfig, *, start_kind: str) -> str:
        return f"{self._display_method_label(solve_config)} [{start_kind}]"

    def _display_method_label(self, solve_config: SolverConfig) -> str:
        if _uses_least_squares_api(solve_config):
            return f"least_squares/{solve_config.method}"
        return f"root/{solve_config.method}"

    def _display_start_kind(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        x0_was_provided: bool,
    ) -> str:
        if x0_was_provided or solve_config.initial_policy == "warm":
            return "warm-start"
        if solve_config.initial_policy == "homothetic":
            return "homothetic-start"
        if solve_config.initial_policy == "zeros":
            return "zero-start"
        return "encoded-start" if self._is_warm_initial_guess(x_guess, False) else "zero-start"

    def _is_warm_initial_guess(self, x_guess: np.ndarray, x0_was_provided: bool) -> bool:
        return bool(x0_was_provided or not self._is_zero_guess(x_guess))

    def _is_zero_guess(self, x_guess: np.ndarray) -> bool:
        x_eval = self.operator.coerce_x(x_guess)
        return bool(np.all(x_eval == 0.0))

    def _finalize_attempts(
        self,
        attempts: list[
            tuple[str, tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]
        ],
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        for label, result, error in reversed(attempts):
            if self._attempt_succeeded(result, error):
                return self._build_attempts_result(
                    attempts, selected_label=label, selected_result=result
                )

        candidate_idx = self._best_attempt_index(attempts)
        if candidate_idx is None:
            tail_label, _, tail_exc = attempts[-1]
            if tail_exc is not None:
                raise RuntimeError(
                    f"All solve attempts failed; last method={tail_label}"
                ) from tail_exc
            raise RuntimeError("All solve attempts failed without a usable result")

        selected_label, selected_result, _ = attempts[candidate_idx]
        if selected_result is None:
            raise RuntimeError("Selected solve attempt has no result")
        return self._build_attempts_result(
            attempts, selected_label=selected_label, selected_result=selected_result
        )

    def _build_attempts_result(
        self,
        attempts: list[
            tuple[str, tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]
        ],
        *,
        selected_label: str,
        selected_result: tuple[np.ndarray, bool, str, int, int, int, float],
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        message = "; ".join(
            f"attempt(method={label}) "
            f"{'succeeded' if self._attempt_succeeded(res, err) else 'failed'}: "
            f"{self._format_attempt_failure(method=label, result=res, error=err)}"
            for label, res, err in attempts
        )
        return (
            selected_result[0],
            self._attempt_succeeded(selected_result, None),
            f"{message}; selected method={selected_label}",
            sum(int(result[3]) for _, result, _ in attempts if result is not None),
            sum(int(result[4]) for _, result, _ in attempts if result is not None),
            sum(int(result[5]) for _, result, _ in attempts if result is not None),
            float(selected_result[6]),
        )

    def _best_attempt_index(
        self,
        attempts: list[
            tuple[str, tuple[np.ndarray, bool, str, int, int, int, float] | None, Exception | None]
        ],
    ) -> int | None:
        candidate_indices = [
            idx
            for idx, (_, result, error) in enumerate(attempts)
            if result is not None and error is None
        ]
        if not candidate_indices:
            candidate_indices = [
                idx for idx, (_, result, _) in enumerate(attempts) if result is not None
            ]
        if not candidate_indices:
            return None
        return min(candidate_indices, key=lambda idx: self._attempt_residual_norm(attempts[idx][1]))

    def _solve_opt_problem(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        residual_kind: str,
    ) -> tuple[np.ndarray, bool, str, int, int, int, float]:
        """Execute one complete nonlinear solve."""

        opt = self._run_solve_full(x_guess, solve_config=solve_config, residual_kind=residual_kind)
        x_opt = self.operator.coerce_x(opt.x)
        residual_norm = self._optimizer_residual_norm(opt)
        if residual_norm is None or not np.isfinite(residual_norm):
            residual_norm, _ = self._safe_residual_norm(
                x_opt,
                solve_config=solve_config,
                residual_kind=residual_kind,
            )
        accepted_by_residual = _residual_within_acceptance(residual_norm, solve_config)
        accepted = bool(
            accepted_by_residual
            or (
                bool(opt.success)
                and residual_norm is not None
                and np.isfinite(residual_norm)
                and not _requires_strict_residual_acceptance(
                    solve_config, residual_kind=residual_kind
                )
            )
        )
        message = str(opt.message)
        if not bool(opt.success) and accepted:
            message = f"{message} [accepted by residual]"
        if (
            bool(opt.success)
            and not accepted
            and _requires_strict_residual_acceptance(
                solve_config,
                residual_kind=residual_kind,
            )
        ):
            message = f"{message} [rejected by residual={residual_norm:.6e}]"
        return (
            x_opt,
            accepted,
            message,
            _count_opt_attr(opt, "nfev"),
            _count_opt_attr(opt, "njev"),
            _count_opt_attr(opt, "nit"),
            float("nan") if residual_norm is None else float(residual_norm),
        )

    def _run_solve_full(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        residual_kind: str,
    ):
        _validate_stage_method(solve_config, residual_kind=residual_kind)
        optimize_method = _registered_method_for(solve_config)
        if solve_config.method in ROOT_METHODS:
            return self._run_root_full(
                x_guess, solve_config=solve_config, optimize_method=optimize_method
            )
        return self._run_least_squares_full(
            x_guess,
            solve_config=solve_config,
            optimize_method=optimize_method,
            residual_kind=residual_kind,
        )

    def _residual_function_for(
        self,
        residual_kind: str,
        *,
        solve_config: SolverConfig | None = None,
        x_reference: np.ndarray | None = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        if residual_kind == "blended_collocation":
            return self._blended_collocation_residual_function(
                solve_config=solve_config,
                x_reference=x_reference,
            )

        def residual_fun(x: np.ndarray) -> np.ndarray:
            x_eval = self.operator.coerce_x(x)
            if residual_kind == "variational":
                return self.operator.residual_var(x_eval)
            if residual_kind == "collocation":
                return self.operator.residual_collocation(x_eval)
            raise ValueError(f"Unsupported residual kind {residual_kind!r}.")

        return residual_fun

    def _blended_collocation_residual_function(
        self,
        *,
        solve_config: SolverConfig | None,
        x_reference: np.ndarray | None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Build a variational-state-anchored collocation-polish residual.

        The blend minimizes a convex combination of RMS-normalized distance
        from the variational warm start and the point-collocation force-balance
        residual:

        ``(1-w) * rms((x - x_var) / x_scale)^2 + w * rms(R_col)^2``.

        The reference state is the converged variational solution that warm-starts
        the polish.  This keeps the post-process local in coefficient space
        unless the collocation part has enough weight to justify moving away
        from the weak-form equilibrium.
        """

        config_eval = self.config if solve_config is None else solve_config
        x_ref: np.ndarray | None = (
            None if x_reference is None else self.operator.coerce_x(x_reference).copy()
        )
        x_scale: np.ndarray | None = None
        collocation_scale: float | None = None
        collocation_weight = float(config_eval.collocation_weight)
        anchor_weight = 1.0 - collocation_weight

        def residual_fun(x: np.ndarray) -> np.ndarray:
            nonlocal x_ref, x_scale, collocation_scale
            x_eval = self.operator.coerce_x(x)
            if x_ref is None:
                x_ref = x_eval.copy()
            if x_scale is None:
                x_scale = _reference_x_scale_vector(self.operator, x_ref)
            if collocation_scale is None:
                collocation_scale = _reference_rms_scale(
                    self.operator.residual_collocation(x_ref),
                    floor=float(config_eval.max_residual),
                )
            anchor_delta = x_eval - x_ref
            collocation_residual = np.asarray(
                self.operator.residual_collocation(x_eval), dtype=np.float64
            )
            return np.concatenate(
                (
                    _weighted_rms_vector(
                        anchor_delta,
                        scale=x_scale,
                        weight=anchor_weight,
                    ),
                    _weighted_rms_block(
                        collocation_residual,
                        scale=collocation_scale,
                        weight=collocation_weight,
                    ),
                )
            )

        return residual_fun

    def _optimizer_residual_norm(self, opt) -> float | None:
        fun = getattr(opt, "fun", None)
        if fun is None:
            return None
        return _residual_array_norm(fun)

    def _run_root_once(
        self,
        root_fun,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        optimize_method: OptimizeMethod,
        options: dict[str, object],
        get_raw_residual: Callable[[np.ndarray], np.ndarray] | None = None,
        decode_x: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        opt = optimize_method(
            root_fun,
            x_guess,
            tol=solve_config.max_residual,
            options=options,
        )
        if decode_x is not None:
            opt.x = decode_x(opt.x)
        if get_raw_residual is not None:
            x_opt = self.operator.coerce_x(opt.x)
            opt.fun = get_raw_residual(x_opt)
        return opt

    def _run_root_full(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        optimize_method: OptimizeMethod,
    ):
        """Call `scipy.optimize.root` once on the full packed x."""

        root_fun = self.operator
        get_raw_residual: Callable[[np.ndarray], np.ndarray] | None = None
        options = _root_options_for(solve_config)
        balanced_scope = "block"
        initial_residual: np.ndarray | None = None
        scaled_fun, get_raw_residual = self._build_normalized_residual_wrapper(
            x_guess,
            solve_config=solve_config,
            residual_kind="variational",
            legacy_transform="linear",
            balanced_scope=balanced_scope,
            initial_residual=initial_residual if balanced_scope == "block" else None,
        )
        x_root_guess = x_guess
        decode_x: Callable[[np.ndarray], np.ndarray] | None = None
        x_transform_fun, x_root_guess, decode_x = self._build_x_transform_wrapper(x_guess)
        if x_transform_fun is not None:
            if scaled_fun is not None:

                def root_fun(z_eval: np.ndarray) -> np.ndarray:
                    return scaled_fun(x_transform_fun(z_eval))
            else:

                def root_fun(z_eval: np.ndarray) -> np.ndarray:
                    return self.operator(x_transform_fun(z_eval))
        elif scaled_fun is not None:
            root_fun = scaled_fun
        if scaled_fun is not None and solve_config.method == "hybr":
            normalization_mode = getattr(solve_config, "residual_normalization", "block_huber")
            if normalization_mode != "none":
                options = {**options, "factor": 1.0}

        return self._run_root_once(
            root_fun,
            x_root_guess,
            solve_config=solve_config,
            optimize_method=optimize_method,
            options=options,
            get_raw_residual=get_raw_residual,
            decode_x=decode_x,
        )

    def _build_normalized_residual_wrapper(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        residual_kind: str,
        legacy_transform: str = "linear",
        balanced_scope: str = "block",
        initial_residual: np.ndarray | None = None,
    ) -> tuple[
        Callable[[np.ndarray], np.ndarray] | None, Callable[[np.ndarray], np.ndarray] | None
    ]:
        """Build the solver-layer residual normalization wrapper."""

        mode = getattr(solve_config, "residual_normalization", "block_huber")
        if mode == "none":
            return None, None
        if _mode_is_block_rms(mode):
            return self._build_legacy_residual_transform_wrapper(
                x_guess, transform=legacy_transform
            )
        return self._build_balanced_residual_transform_wrapper(
            x_guess,
            solve_config=solve_config,
            residual_kind=residual_kind,
            scope=balanced_scope,
            initial_x=x_guess,
            initial_residual=initial_residual,
            mode=mode,
        )

    def _build_legacy_residual_transform_wrapper(
        self,
        x_guess: np.ndarray,
        *,
        transform: str,
    ) -> tuple[
        Callable[[np.ndarray], np.ndarray] | None, Callable[[np.ndarray], np.ndarray] | None
    ]:
        """Legacy block-RMS residual transform wrapper for comparison mode."""

        block_lengths = self.operator.residual_block_lengths()
        if block_lengths is None:
            return None, None

        try:
            self.operator.coerce_x(x_guess)
        except Exception:
            return None, None
        block_lengths_eval = np.asarray(block_lengths, dtype=np.int64)
        scale: np.ndarray | None = None
        last_x: np.ndarray | None = None
        last_raw_residual: np.ndarray | None = None

        def wrapped(x: np.ndarray) -> np.ndarray:
            nonlocal scale, last_x, last_raw_residual
            x_eval = self.operator.coerce_x(x)
            raw_residual = np.asarray(self.operator(x_eval), dtype=np.float64)
            last_x = x_eval.copy()
            last_raw_residual = raw_residual.copy()
            if scale is None:
                scale = _build_block_rms_scale(raw_residual, block_lengths_eval)
                if scale is None:
                    scale = np.ones_like(raw_residual)
            scaled_residual = raw_residual / scale
            if transform == "asinh":
                return np.arcsinh(scaled_residual)
            return scaled_residual

        def get_raw_residual(x: np.ndarray) -> np.ndarray:
            x_eval = self.operator.coerce_x(x)
            if (
                last_x is not None
                and last_raw_residual is not None
                and np.array_equal(last_x, x_eval)
            ):
                return last_raw_residual.copy()
            return np.asarray(self.operator(x_eval), dtype=np.float64)

        return wrapped, get_raw_residual

    def _build_balanced_residual_transform_wrapper(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        residual_kind: str,
        scope: str,
        initial_x: np.ndarray | None = None,
        initial_residual: np.ndarray | None = None,
        mode: str = "block_huber",
    ) -> tuple[
        Callable[[np.ndarray], np.ndarray] | None, Callable[[np.ndarray], np.ndarray] | None
    ]:
        """O(n)-modeled linear left preconditioner for residuals."""

        try:
            self.operator.coerce_x(x_guess)
        except Exception:
            return None, None

        residual_fun = self._residual_function_for(residual_kind)
        if scope not in {"block", "global"}:
            raise ValueError(f"Unsupported balanced residual scope {scope!r}.")
        block_lengths = (
            self.operator.residual_block_lengths()
            if residual_kind == "variational" and scope == "block"
            else None
        )
        block_lengths_eval = (
            None if block_lengths is None else np.asarray(block_lengths, dtype=np.int64)
        )
        floor = float(solve_config.residual_normalization_floor)
        max_ratio = float(solve_config.residual_normalization_max_ratio)
        huber_tau = float(solve_config.residual_normalization_huber_tau)
        scale: np.ndarray | None = None
        last_x: np.ndarray | None = None
        last_raw_residual: np.ndarray | None = None
        if initial_residual is None and not _mode_is_block_rms(mode):
            try:
                initial_x_eval = self.operator.coerce_x(x_guess)
                initial_residual = np.asarray(residual_fun(initial_x_eval), dtype=np.float64)
                initial_x = initial_x_eval
            except Exception:
                initial_residual = None
        if initial_residual is not None:
            initial_residual_eval = np.asarray(initial_residual, dtype=np.float64)
            scale = self._build_residual_scale_for_mode(
                initial_residual_eval,
                block_lengths_eval,
                solve_config=solve_config,
                residual_fun=residual_fun,
                x_guess=self.operator.coerce_x(x_guess),
                mode=mode,
                floor=floor,
                max_ratio=max_ratio,
                huber_tau=huber_tau,
            )
            if initial_x is not None:
                last_x = self.operator.coerce_x(initial_x).copy()
                last_raw_residual = initial_residual_eval.copy()

        def wrapped(x: np.ndarray) -> np.ndarray:
            nonlocal scale, last_x, last_raw_residual
            x_eval = self.operator.coerce_x(x)
            if (
                last_x is not None
                and last_raw_residual is not None
                and np.array_equal(last_x, x_eval)
            ):
                raw_residual = last_raw_residual.copy()
            else:
                raw_residual = np.asarray(residual_fun(x_eval), dtype=np.float64)
                last_x = x_eval.copy()
                last_raw_residual = raw_residual.copy()
            if scale is None:
                scale = self._build_residual_scale_for_mode(
                    raw_residual,
                    block_lengths_eval,
                    solve_config=solve_config,
                    residual_fun=residual_fun,
                    x_guess=x_eval,
                    mode=mode,
                    floor=floor,
                    max_ratio=max_ratio,
                    huber_tau=huber_tau,
                )
            return raw_residual / scale

        def get_raw_residual(x: np.ndarray) -> np.ndarray:
            x_eval = self.operator.coerce_x(x)
            if (
                last_x is not None
                and last_raw_residual is not None
                and np.array_equal(last_x, x_eval)
            ):
                return last_raw_residual.copy()
            return np.asarray(residual_fun(x_eval), dtype=np.float64)

        return wrapped, get_raw_residual

    def _build_residual_scale_for_mode(
        self,
        residual: np.ndarray,
        block_lengths: np.ndarray | None,
        *,
        solve_config: SolverConfig,
        residual_fun: Callable[[np.ndarray], np.ndarray],
        x_guess: np.ndarray,
        mode: str,
        floor: float,
        max_ratio: float,
        huber_tau: float,
    ) -> np.ndarray:
        return make_residual_scale(
            mode,
            residual,
            block_lengths,
            floor=floor,
            max_ratio=max_ratio,
            huber_tau=huber_tau,
            residual_fun=residual_fun,
            x_guess=x_guess,
            x_scale=_build_x_block_scale_vector(self.operator, x_guess),
            probe_count=int(solve_config.residual_normalization_probe_count),
            probe_step=float(solve_config.residual_normalization_probe_step),
            sensitivity_lambda=float(solve_config.residual_normalization_sensitivity_lambda),
        )

    def _build_x_transform_wrapper(
        self,
        x_guess: np.ndarray,
    ) -> tuple[
        Callable[[np.ndarray], np.ndarray] | None,
        np.ndarray,
        Callable[[np.ndarray], np.ndarray] | None,
    ]:
        x_eval = self.operator.coerce_x(x_guess)
        x_scale = _build_x_block_scale_vector(self.operator, x_eval)
        if x_scale is None:
            return None, x_eval, None

        inv_scale = 1.0 / x_scale

        def map_z_to_x(z: np.ndarray) -> np.ndarray:
            z_eval = np.asarray(z, dtype=np.float64)
            if z_eval.ndim != 1 or z_eval.shape[0] != x_eval.shape[0]:
                raise ValueError(f"Expected z to have shape {x_eval.shape}, got {z_eval.shape}")
            return self.operator.coerce_x(z_eval * x_scale)

        return map_z_to_x, x_eval * inv_scale, map_z_to_x

    def _initial_residual_stats(
        self,
        x_guess: np.ndarray,
        *,
        residual_kind: str,
    ) -> tuple[np.ndarray | None, float | None]:
        try:
            residual_fun = self._residual_function_for(residual_kind)
            residual = np.asarray(residual_fun(self.operator.coerce_x(x_guess)), dtype=np.float64)
        except Exception:
            return None, None
        return residual, _residual_array_norm(residual)

    def _run_least_squares_full(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        optimize_method: OptimizeMethod,
        residual_kind: str,
    ):
        """Call `scipy.optimize.least_squares` once on the full packed x."""

        least_squares_fun = self._residual_function_for(
            residual_kind,
            solve_config=solve_config,
            x_reference=x_guess,
        )
        get_raw_residual: Callable[[np.ndarray], np.ndarray] | None = None
        kwargs = _least_squares_kwargs_for(solve_config)
        normalizer_applied = False

        if residual_kind == "variational":
            legacy_transform = "asinh" if solve_config.method == "lm" else "linear"
            normalized_fun, get_raw_residual = self._build_normalized_residual_wrapper(
                x_guess,
                solve_config=solve_config,
                residual_kind=residual_kind,
                legacy_transform=legacy_transform,
            )
            if normalized_fun is not None:
                least_squares_fun = normalized_fun
                normalizer_applied = True
                if solve_config.method == "lm":
                    kwargs["x_scale"] = 1.0

        if (
            not normalizer_applied
            and solve_config.method == "trf"
            and residual_kind == "variational"
        ):
            residual0, _ = self._initial_residual_stats(x_guess, residual_kind=residual_kind)
            if _should_use_robust_trf_loss(
                residual0,
                self.operator.residual_block_lengths(),
            ):
                kwargs["loss"] = "cauchy"
                kwargs["f_scale"] = max(_residual_rms(residual0), 1.0)

        opt = optimize_method(
            least_squares_fun,
            x_guess,
            **kwargs,
        )
        if get_raw_residual is not None:
            x_opt = self.operator.coerce_x(opt.x)
            opt.fun = get_raw_residual(x_opt)
        return opt


def _validate_stage_solve_config(solve_config: SolverConfig, *, residual_kind: str) -> None:
    _validate_stage_method(solve_config, residual_kind=residual_kind)
    if residual_kind == "variational" or not solve_config.enable_fallback:
        return

    root_fallbacks = [
        method for method in solve_config.fallback_methods if method not in LEAST_SQUARES_METHODS
    ]
    if root_fallbacks:
        unsupported = ", ".join(repr(method) for method in root_fallbacks)
        raise ValueError(
            f"Collocation needs least_squares ('trf' or 'lm'); bad fallback(s): {unsupported}."
        )


def _validate_stage_method(solve_config: SolverConfig, *, residual_kind: str) -> None:
    if residual_kind in {"collocation", "blended_collocation"} and not _uses_least_squares_api(
        solve_config
    ):
        raise ValueError("Collocation needs least_squares ('trf' or 'lm').")
    if residual_kind not in {"variational", "collocation", "blended_collocation"}:
        raise ValueError(f"Unsupported residual kind {residual_kind!r}.")


def _registered_method_for(solve_config: SolverConfig) -> OptimizeMethod:
    try:
        return SUPPORTED_METHODS[solve_config.method]
    except KeyError as exc:
        raise ValueError(f"Unsupported solver method {solve_config.method!r}.") from exc


def _root_options_for(solve_config: SolverConfig) -> dict[str, object]:
    """Map `SolverConfig` to `scipy.optimize.root(..., options=...)`."""

    options: dict[str, object] = {}
    method = solve_config.method

    if method in {"hybr", "df-sane"}:
        if solve_config.max_evaluations > 0:
            options["maxfev"] = max(int(solve_config.max_evaluations), 500)
        if method == "hybr":
            options["eps"] = 1.0e-6
    return options


def _least_squares_kwargs_for(solve_config: SolverConfig) -> dict[str, object]:
    """Map `SolverConfig` to `scipy.optimize.least_squares(...)`."""

    kwargs: dict[str, object] = {
        "ftol": float(solve_config.max_residual),
        "xtol": float(solve_config.max_residual),
        "gtol": float(solve_config.max_residual),
    }
    if solve_config.max_evaluations > 0:
        kwargs["max_nfev"] = max(int(solve_config.max_evaluations), 500)
    return kwargs


def _count_opt_attr(opt, name: str) -> int:
    value = getattr(opt, name, 0)
    if value is None:
        return 0
    return int(value)


def _residual_array_norm(residual: np.ndarray) -> float:
    """Return the Euclidean norm; scalar residuals count as length-1 vectors."""

    residual_eval = np.asarray(residual, dtype=np.float64)
    if residual_eval.ndim == 0:
        residual_eval = residual_eval.reshape(1)
    return float(np.linalg.norm(residual_eval))


def _reference_rms_scale(residual: np.ndarray, *, floor: float) -> float:
    """Return a positive RMS scale for dimensionless mixed residual blocks."""

    residual_eval = np.asarray(residual, dtype=np.float64)
    if residual_eval.ndim == 0:
        residual_eval = residual_eval.reshape(1)
    finite = residual_eval[np.isfinite(residual_eval)]
    if finite.size == 0:
        return max(float(floor), np.finfo(np.float64).tiny)
    rms = float(np.sqrt(np.mean(finite * finite)))
    return max(rms, float(floor), np.finfo(np.float64).tiny)


def _weighted_rms_block(residual: np.ndarray, *, scale: float, weight: float) -> np.ndarray:
    """Scale one residual block so its squared norm equals weighted RMS squared."""

    residual_eval = np.asarray(residual, dtype=np.float64)
    if residual_eval.ndim == 0:
        residual_eval = residual_eval.reshape(1)
    if residual_eval.size == 0 or float(weight) <= 0.0:
        return np.zeros_like(residual_eval, dtype=np.float64)
    block_scale = max(float(scale), np.finfo(np.float64).tiny) * np.sqrt(residual_eval.size)
    return np.sqrt(float(weight)) * residual_eval / block_scale


def _reference_x_scale_vector(operator: Operator, x_reference: np.ndarray) -> np.ndarray:
    """Return a positive coefficient scale vector for local polish regularization."""

    x_ref = np.asarray(x_reference, dtype=np.float64)
    scale = _build_x_block_scale_vector(operator, x_ref)
    if scale is None:
        scale = np.maximum(np.abs(x_ref), 1.0)
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape != x_ref.shape:
        scale = np.ones_like(x_ref)
    return np.maximum(scale, np.finfo(np.float64).tiny)


def _weighted_rms_vector(residual: np.ndarray, *, scale: np.ndarray, weight: float) -> np.ndarray:
    """Scale one vector block by per-component scales and RMS block length."""

    residual_eval = np.asarray(residual, dtype=np.float64)
    scale_eval = np.asarray(scale, dtype=np.float64)
    if residual_eval.ndim == 0:
        residual_eval = residual_eval.reshape(1)
    if scale_eval.shape != residual_eval.shape:
        scale_eval = np.ones_like(residual_eval, dtype=np.float64)
    if residual_eval.size == 0 or float(weight) <= 0.0:
        return np.zeros_like(residual_eval, dtype=np.float64)
    block_scale = np.maximum(scale_eval, np.finfo(np.float64).tiny) * np.sqrt(residual_eval.size)
    return np.sqrt(float(weight)) * residual_eval / block_scale


def _uses_least_squares_api(solve_config: SolverConfig) -> bool:
    return solve_config.method in LEAST_SQUARES_METHODS


def _build_initial_state(operator: Operator, solve_config: SolverConfig) -> np.ndarray:
    """Build the packed initial state requested by ``solve_config.initial_policy``."""

    initial_policy = solve_config.initial_policy
    if initial_policy is None:
        return operator.encode_initial_state()
    if initial_policy == "zeros":
        return np.zeros(operator.x_size, dtype=np.float64)
    if initial_policy == "homothetic":
        return _build_boundary_homothetic_initial_state(
            operator, boundary_slope_factor=solve_config.initial_homothetic_lambda
        )
    if initial_policy == "warm":
        raise RuntimeError("_build_initial_state('warm') needs the current solver x0")
    raise ValueError(f"Unsupported initial_policy {initial_policy!r}")


def _build_boundary_homothetic_initial_state(
    operator: Operator, *, boundary_slope_factor: float = 1.0
) -> np.ndarray:
    """Return a cheap boundary-scaled x0 for nested, homothetic surfaces.

    ``boundary_slope_factor`` sets the target boundary slope ratio
    ``u_m'(1)=lambda*offset`` for each active c/s mode. The default ``1.0``
    matches homothetic scaling; smaller values relax the boundary more
    aggressively.
    """

    return _build_boundary_slope_initial_state(
        operator, boundary_slope_factor=boundary_slope_factor
    )


def _build_boundary_slope_initial_state(
    operator: Operator, *, boundary_slope_factor: float
) -> np.ndarray:
    """Set first c/s coefficients so ``u_m'(1)=lambda*offset``."""

    return operator.build_boundary_slope_initial_state(boundary_slope_factor=boundary_slope_factor)


def _accepted_residual_norm(solve_config: SolverConfig) -> float:
    return max(float(solve_config.max_residual) * 10.0, 1.0e-5)


def _residual_within_acceptance(residual_norm: float | None, solve_config: SolverConfig) -> bool:
    return bool(
        residual_norm is not None
        and np.isfinite(residual_norm)
        and residual_norm <= _accepted_residual_norm(solve_config)
    )


def _requires_strict_residual_acceptance(solve_config: SolverConfig, *, residual_kind: str) -> bool:
    return residual_kind == "variational"


def _hard_residual_norm_threshold() -> float:
    return 1.0e3


def _trf_robust_block_rms_threshold() -> float:
    return 2.0e40


def _should_use_robust_trf_loss(
    residual: np.ndarray | None,
    block_lengths: np.ndarray | None,
) -> bool:
    if residual is None or block_lengths is None:
        return False
    block_rms = _block_rms_values(residual, np.asarray(block_lengths, dtype=np.int64))
    if block_rms is None or block_rms.size == 0:
        return False
    return bool(np.median(block_rms) >= _trf_robust_block_rms_threshold())


def _x_scale_floor() -> float:
    return 1.0e-2


def _x_scale_core_profile_prior() -> float:
    return 1.5e-1


def _x_scale_fourier_profile_prior() -> float:
    return 5.0e-2


def _x_scale_f_profile_prior() -> float:
    return 2.5e-1


def _x_scale_kappa_profile_prior() -> float:
    return 1.0


def _x_scale_profile_prior(name: str) -> float:
    if name in {"h", "v", "psin"}:
        return _x_scale_core_profile_prior()
    if name == "k":
        return _x_scale_kappa_profile_prior()
    if name.startswith(("c", "s")):
        return _x_scale_fourier_profile_prior()
    if name == "F":
        return _x_scale_f_profile_prior()
    return _x_scale_f_profile_prior()


def _use_offset_for_x_scale(name: str) -> bool:
    return name not in {"h", "v", "psin"}


def _build_x_block_scale_vector(operator, x_guess: np.ndarray) -> np.ndarray | None:
    x_eval = np.asarray(x_guess, dtype=np.float64)
    if not hasattr(operator, "active_profile_blocks"):
        return None

    scale = np.ones_like(x_eval)
    floor = _x_scale_floor()
    for _, profile_name, coeff_indices, offset, profile_scale in operator.active_profile_blocks():
        coeff_indices = np.asarray(coeff_indices, dtype=np.int64)
        length = int(coeff_indices.size)
        if length <= 0:
            continue
        if np.any(coeff_indices < 0) or np.any(coeff_indices >= x_eval.size):
            return None
        block_guess = x_eval[coeff_indices]
        guess_rms = float(np.linalg.norm(block_guess) / np.sqrt(length))
        offset_scale = abs(float(offset)) if _use_offset_for_x_scale(profile_name) else 0.0
        profile_scale = abs(float(profile_scale))
        profile_prior = _x_scale_profile_prior(profile_name)
        if abs(profile_scale - 1.0) <= 1.0e-12:
            profile_scale = profile_prior
        block_scale = max(offset_scale, profile_scale, profile_prior, guess_rms, floor)
        scale[coeff_indices] = block_scale
    return scale
