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
from veqpy.model.problem import Problem
from veqpy.operator.operator import Operator
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

_ACCEPTED_RESIDUAL_FACTOR = 10.0
_ACCEPTED_RESIDUAL_FLOOR = 1.0e-5
_TRF_ROBUST_BLOCK_RMS_THRESHOLD = 2.0e40
_X_SCALE_FLOOR = 1.0e-2
_X_SCALE_CORE_PROFILE_PRIOR = 1.5e-1
_X_SCALE_FOURIER_PROFILE_PRIOR = 5.0e-2
_X_SCALE_F_PROFILE_PRIOR = 2.5e-1
_X_SCALE_KAPPA_PROFILE_PRIOR = 1.0
_X_SCALE_OFFSETLESS_PROFILES = frozenset({"h", "v", "psin"})
_AUTO_CURVE_STRAIN_THRESHOLD = 0.20
_AUTO_CURVE_STRAIN_SAMPLES = 32
_AUTO_CURVE_STRAIN_MAX_ORDER = 32
_AUTO_CURVE_STRAIN_THETA = np.linspace(
    0.0,
    2.0 * np.pi,
    _AUTO_CURVE_STRAIN_SAMPLES,
    endpoint=False,
    dtype=np.float64,
)
_AUTO_CURVE_STRAIN_SIN_THETA = np.sin(_AUTO_CURVE_STRAIN_THETA)
_AUTO_CURVE_STRAIN_COS_THETA = np.cos(_AUTO_CURVE_STRAIN_THETA)
_AUTO_CURVE_STRAIN_ORDERS = np.arange(
    1,
    _AUTO_CURVE_STRAIN_MAX_ORDER + 1,
    dtype=np.float64,
)
_AUTO_CURVE_STRAIN_ORDER_THETA = (
    _AUTO_CURVE_STRAIN_ORDERS[:, np.newaxis] * _AUTO_CURVE_STRAIN_THETA[np.newaxis, :]
)
_AUTO_CURVE_STRAIN_SIN_ORDER_THETA = np.sin(_AUTO_CURVE_STRAIN_ORDER_THETA)
_AUTO_CURVE_STRAIN_COS_ORDER_THETA = np.cos(_AUTO_CURVE_STRAIN_ORDER_THETA)


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
        self.x0 = self.operator.zero_state()

    def reset(self) -> None:
        """Zero the solver-owned x0 in place."""

        self.x0.fill(0.0)

    def clear(self) -> None:
        """Clear solve history without changing the current x0."""

        self.history.clear()

    def replace_problem(self, problem: Problem) -> None:
        """Replace the problem with a compatible one."""

        self.operator.replace_problem(problem)

    def replace_case(self, case: Problem) -> None:
        """Compatibility alias for ``replace_problem``."""

        self.replace_problem(case)

    def solve(
        self,
        x0: np.ndarray | None = None,
        *,
        method: str | None = None,
        max_residual: float | None = None,
        max_evaluations: int | None = None,
        initial_policy: str | None = None,
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

        call_started = perf_counter()
        solve_config = self._resolve_solve_config(
            method=method,
            max_residual=max_residual,
            max_evaluations=max_evaluations,
            initial_policy=initial_policy,
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

        solve_started = perf_counter()
        if x0 is not None:
            self.x0 = self.operator.coerce_x(x0).copy()
        elif solve_config.initial_policy == "warm":
            self.x0 = self.x0.copy()
        else:
            self.x0 = _build_initial_state(self.operator, solve_config).copy()
        if x0 is not None or solve_config.initial_policy != "warm":
            # Source stages such as PJ2/psin/uniform cache a psin query tied to
            # the previous x0.  Any externally supplied or rebuilt x0 must break
            # that cache before the first residual evaluation.
            self.operator.invalidate_source_state()

        x_guess = self.x0.copy()

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

        x_final = self.operator.coerce_x(x_opt)
        residual_final_exc = None
        if not bool(success) and not np.isfinite(residual_norm_final):
            final_residual_config = (
                self._collocation_stage_config(solve_config)
                if solve_config.enable_collocation
                else solve_config
            )
            final_residual_kind = (
                self._collocation_residual_kind(solve_config)
                if solve_config.enable_collocation
                else "variational"
            )
            residual_norm_final, residual_final_exc = self._safe_residual_norm(
                x_final,
                solve_config=final_residual_config,
                residual_kind=final_residual_kind,
            )
        if residual_final_exc is not None:
            success = False
            message = (
                f"{message} [final residual evaluation failed: "
                f"{type(residual_final_exc).__name__}: {residual_final_exc}]"
            )

        elapsed = (perf_counter() - solve_started) * 1e6
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
            total_elapsed=elapsed,
        )

        self.x0 = x_final.copy()
        self.result = replace(self.result, total_elapsed=(perf_counter() - call_started) * 1e6)
        record = SolverRecord(
            problem_snapshot=self.operator.problem.copy(),
            config_snapshot=solve_config,
            result_snapshot=self.result,
        )

        if solve_config.enable_verbose:
            Console().print(record)

        if solve_config.enable_history:
            self.history.append(record)

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
        initial_policy: str | None,
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
        if initial_policy is not None:
            overrides["initial_policy"] = str(initial_policy)
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

        # The variational stage owns global equilibrium validity.  Collocation
        # is a local polish step, so it starts from the weak-form result and is
        # not allowed to run its own fallback sequence.
        variational_config = replace(solve_config, enable_collocation=False)
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
            # Keep the two-stage accounting path even when the collocation
            # objective is disabled; callers still get a single combined result.
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
            # Fallbacks are disabled here because collocation is already a
            # second-stage repair.  Restarting with other methods would blur
            # whether the final x is still a local polish of the variational x.
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
                # For fallback success, preserve every prior attempt in the
                # message and aggregate counters so diagnostics match total work.
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
        # All attempts start from the same physical state.  This makes method
        # comparisons meaningful and avoids carrying a failed method's possibly
        # invalid source fixed-point state into the next attempt.
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
        fallback_methods = () if not solve_config.enable_fallback else solve_config.fallback_methods
        for fallback_method in fallback_methods:
            if fallback_method in seen_methods:
                continue
            seen_methods.add(fallback_method)
            fallback_config = replace(solve_config, method=fallback_method)
            attempt_plans.append(
                (
                    self._display_attempt_label(fallback_config, start_kind="warm-fallback"),
                    # Fallbacks compare methods from the same physical initial
                    # state, not from the failed method's terminal iterate.
                    x_initial.copy(),
                    fallback_config,
                )
            )
        return attempt_plans

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
                # SciPy may raise after evaluating an already-good initial
                # state (for example while estimating a Jacobian). Preserve the
                # usable x0 instead of forcing a worse fallback.
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
        if solve_config.method in LEAST_SQUARES_METHODS:
            method_label = f"least_squares/{solve_config.method}"
        else:
            method_label = f"root/{solve_config.method}"
        return f"{method_label} [{start_kind}]"

    def _display_start_kind(
        self,
        x_guess: np.ndarray,
        *,
        solve_config: SolverConfig,
        x0_was_provided: bool,
    ) -> str:
        if x0_was_provided or solve_config.initial_policy == "warm":
            return "warm-start"
        if solve_config.initial_policy == "geometric":
            return "geometric-start"
        if solve_config.initial_policy == "geometric-refined":
            return "geometric-refined-start"
        if solve_config.initial_policy == "legacy-geometric":
            return "legacy-geometric-start"
        if solve_config.initial_policy == "auto":
            return "auto-start"
        if solve_config.initial_policy == "zeros":
            return "zero-start"
        x_eval = self.operator.coerce_x(x_guess)
        return "encoded-start" if not bool(np.all(x_eval == 0.0)) else "zero-start"

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

        # If every method reports failure, still return the numerically best
        # finite attempt.  The success flag and message keep the failure visible
        # while giving callers a reproducible terminal state to inspect.
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
        # Select by final residual, not by attempt order, so a rejected but
        # informative partial solve can be returned for inspection.
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
            # Some scipy methods do not expose a useful final fun vector; compute
            # one last residual so acceptance and SolverResult use the same norm.
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
                and residual_kind != "variational"
            )
        )
        # Variational solves are accepted by residual only; collocation polish is
        # allowed to trust the least_squares success flag because its objective is
        # a local post-process around a variational solution.
        message = str(opt.message)
        if not bool(opt.success) and accepted:
            message = f"{message} [accepted by residual]"
        if bool(opt.success) and not accepted and residual_kind == "variational":
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
                return self.operator.residual_var(x_eval, check=False)
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
                # Anchor regularization uses profile-aware coefficient scales so
                # one high-amplitude family cannot dominate the local polish.
                x_scale = _reference_x_scale_vector(self.operator, x_ref)
            if collocation_scale is None:
                # Freeze the collocation RMS at the variational reference; if
                # it changed with x, the optimizer could improve the objective by
                # moving the scale rather than the equilibrium.
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
        # root() sees either the raw residual, a fixed left-preconditioned
        # residual, or both residual and x transformed for conditioning.  The
        # reported opt.fun is restored to raw residual through get_raw_residual.
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
                # SciPy hybr's default trust-region factor can overstep badly
                # once residual blocks are normalized; a smaller initial radius
                # keeps the first Jacobian model local.
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
        raw_buffer = np.empty(self.operator.x_size, dtype=np.float64)
        scaled_buffer = np.empty(self.operator.x_size, dtype=np.float64)
        last_x = np.empty_like(x_guess)
        last_x_valid = False

        def wrapped(x: np.ndarray) -> np.ndarray:
            nonlocal scale, last_x_valid
            x_eval = x
            self.operator.residual_var_into(x_eval, raw_buffer, check=False)
            np.copyto(last_x, x_eval)
            last_x_valid = True
            if scale is None:
                # Legacy mode freezes the first evaluated block RMS.  Rebuilding
                # it every call would make the residual map non-stationary.
                scale = _build_block_rms_scale(raw_buffer, block_lengths_eval)
                if scale is None:
                    scale = np.ones_like(raw_buffer)
            np.divide(raw_buffer, scale, out=scaled_buffer)
            if transform == "asinh":
                np.arcsinh(scaled_buffer, out=scaled_buffer)
            return scaled_buffer.copy()

        def get_raw_residual(x: np.ndarray) -> np.ndarray:
            x_eval = self.operator.coerce_x(x)
            if last_x_valid and np.array_equal(last_x, x_eval):
                return raw_buffer.copy()
            out = np.empty(self.operator.x_size, dtype=np.float64)
            self.operator.residual_var_into(x_eval, out, check=False)
            return out

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
        residual_size = self.operator.x_size if residual_kind == "variational" else -1
        raw_buffer = (
            np.empty(residual_size, dtype=np.float64) if residual_size >= 0 else np.empty(0)
        )
        scaled_buffer = (
            np.empty(residual_size, dtype=np.float64) if residual_size >= 0 else np.empty(0)
        )
        last_x = np.empty_like(x_guess)
        last_x_valid = False

        def build_scale(residual: np.ndarray, x_scale_guess: np.ndarray) -> np.ndarray:
            return make_residual_scale(
                mode,
                residual,
                block_lengths_eval,
                floor=floor,
                max_ratio=max_ratio,
                huber_tau=huber_tau,
                residual_fun=residual_fun,
                x_guess=x_scale_guess,
                x_scale=_build_x_block_scale_vector(self.operator, x_scale_guess),
                probe_count=int(solve_config.residual_normalization_probe_count),
                probe_step=float(solve_config.residual_normalization_probe_step),
                sensitivity_lambda=float(solve_config.residual_normalization_sensitivity_lambda),
            )

        if initial_residual is None and not _mode_is_block_rms(mode):
            try:
                # Build the balanced scale at the reference point when possible;
                # delayed construction is only a fallback for bad initial calls.
                initial_x_eval = self.operator.coerce_x(x_guess)
                if residual_kind == "variational":
                    initial_residual = np.empty(self.operator.x_size, dtype=np.float64)
                    self.operator.residual_var_into(
                        initial_x_eval,
                        initial_residual,
                        check=False,
                    )
                else:
                    initial_residual = np.asarray(residual_fun(initial_x_eval), dtype=np.float64)
                initial_x = initial_x_eval
            except Exception:
                initial_residual = None
        if initial_residual is not None:
            initial_residual_eval = np.asarray(initial_residual, dtype=np.float64)
            scale = build_scale(initial_residual_eval, self.operator.coerce_x(x_guess))
            if initial_x is not None:
                np.copyto(last_x, self.operator.coerce_x(initial_x))
                if residual_kind == "variational":
                    np.copyto(raw_buffer, initial_residual_eval)
                last_x_valid = residual_kind == "variational"

        def wrapped(x: np.ndarray) -> np.ndarray:
            nonlocal scale, last_x_valid
            if residual_kind == "variational":
                x_eval = x
                if not (last_x_valid and np.array_equal(last_x, x_eval)):
                    self.operator.residual_var_into(x_eval, raw_buffer, check=False)
                    np.copyto(last_x, x_eval)
                    last_x_valid = True
                raw_residual = raw_buffer
            else:
                x_eval = self.operator.coerce_x(x)
                raw_residual = np.asarray(residual_fun(x_eval), dtype=np.float64)
            if scale is None:
                # Like the legacy wrapper, balanced normalization is a fixed left
                # preconditioner for this solve attempt, not a dynamic objective.
                scale = build_scale(raw_residual, x_eval)
            if residual_kind == "variational":
                np.divide(raw_residual, scale, out=scaled_buffer)
                return scaled_buffer.copy()
            return raw_residual / scale

        def get_raw_residual(x: np.ndarray) -> np.ndarray:
            x_eval = self.operator.coerce_x(x)
            if residual_kind == "variational" and last_x_valid and np.array_equal(last_x, x_eval):
                return raw_buffer.copy()
            if residual_kind == "variational":
                out = np.empty(self.operator.x_size, dtype=np.float64)
                self.operator.residual_var_into(x_eval, out, check=False)
                return out
            return np.asarray(residual_fun(x_eval), dtype=np.float64)

        return wrapped, get_raw_residual

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
            # Root solvers operate in z-space for conditioning, but all operator
            # kernels must see the original packed x coordinates.
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
                    # LM already uses the transformed residual scale; x_scale=1
                    # prevents SciPy from layering a second heuristic scale on
                    # top of the solver's block-aware coefficient scale.
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
                # This branch is for pathological unnormalized starts only.
                # Once explicit residual normalization is active, robust loss
                # would change the calibrated objective and is skipped above.
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
    if (
        residual_kind in {"collocation", "blended_collocation"}
        and solve_config.method not in LEAST_SQUARES_METHODS
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


def _build_initial_state(operator: Operator, solve_config: SolverConfig) -> np.ndarray:
    """Build the packed initial state requested by ``solve_config.initial_policy``."""

    initial_policy = solve_config.initial_policy
    if initial_policy is None:
        return operator.zero_state()
    if initial_policy == "zeros":
        return np.zeros(operator.x_size, dtype=np.float64)
    if initial_policy == "geometric":
        return operator.build_boundary_slope_initial_state(include_active_psin=False)
    if initial_policy == "geometric-refined":
        return operator.build_boundary_slope_initial_state()
    if initial_policy == "legacy-geometric":
        return operator.build_legacy_boundary_slope_initial_state()
    if initial_policy == "auto":
        if _boundary_curve_strain(operator.problem.boundary) >= _AUTO_CURVE_STRAIN_THRESHOLD:
            return operator.build_boundary_slope_initial_state()
        return np.zeros(operator.x_size, dtype=np.float64)
    if initial_policy == "warm":
        raise RuntimeError("_build_initial_state('warm') needs the current solver x0")
    raise ValueError(f"Unsupported initial_policy {initial_policy!r}")


def _boundary_curve_strain(boundary: object) -> float:
    c_offsets = _boundary_offset_array(getattr(boundary, "c_offsets", None))
    s_offsets = _boundary_offset_array(getattr(boundary, "s_offsets", None))
    if c_offsets is None or s_offsets is None:
        return float("inf")
    has_c_shape = c_offsets.size > 0 and bool(np.any(c_offsets != 0.0))
    has_s_shape = s_offsets.size > 1 and bool(np.any(s_offsets[1:] != 0.0))
    if not has_c_shape and not has_s_shape:
        return 0.0

    try:
        kappa = abs(float(getattr(boundary, "ka", 1.0)))
    except (TypeError, ValueError):
        return float("inf")
    if not np.isfinite(kappa):
        return float("inf")

    theta = _AUTO_CURVE_STRAIN_THETA
    eta = np.zeros_like(_AUTO_CURVE_STRAIN_THETA)
    eta_prime = np.zeros_like(_AUTO_CURVE_STRAIN_THETA)
    if c_offsets.size:
        eta += c_offsets[0]

    c_fast_count = min(max(c_offsets.size - 1, 0), _AUTO_CURVE_STRAIN_MAX_ORDER)
    if c_fast_count:
        c_tail = c_offsets[1 : c_fast_count + 1]
        eta += c_tail @ _AUTO_CURVE_STRAIN_COS_ORDER_THETA[:c_fast_count]
        eta_prime -= (
            _AUTO_CURVE_STRAIN_ORDERS[:c_fast_count] * c_tail
        ) @ _AUTO_CURVE_STRAIN_SIN_ORDER_THETA[:c_fast_count]
    for order in range(c_fast_count + 1, c_offsets.size):
        order_theta = float(order) * theta
        eta += c_offsets[order] * np.cos(order_theta)
        eta_prime -= float(order) * c_offsets[order] * np.sin(order_theta)

    s_fast_count = min(max(s_offsets.size - 1, 0), _AUTO_CURVE_STRAIN_MAX_ORDER)
    if s_fast_count:
        s_tail = s_offsets[1 : s_fast_count + 1]
        eta += s_tail @ _AUTO_CURVE_STRAIN_SIN_ORDER_THETA[:s_fast_count]
        eta_prime += (
            _AUTO_CURVE_STRAIN_ORDERS[:s_fast_count] * s_tail
        ) @ _AUTO_CURVE_STRAIN_COS_ORDER_THETA[:s_fast_count]
    for order in range(s_fast_count + 1, s_offsets.size):
        order_theta = float(order) * theta
        eta += s_offsets[order] * np.sin(order_theta)
        eta_prime += float(order) * s_offsets[order] * np.cos(order_theta)

    speed_boundary = np.sqrt(
        (np.sin(theta + eta) * (1.0 + eta_prime)) ** 2 + (kappa * _AUTO_CURVE_STRAIN_COS_THETA) ** 2
    )
    speed_ellipse = np.sqrt(
        _AUTO_CURVE_STRAIN_SIN_THETA**2 + (kappa * _AUTO_CURVE_STRAIN_COS_THETA) ** 2
    )
    strain = (speed_boundary - speed_ellipse) / np.maximum(speed_ellipse, 1.0e-12)
    return float(np.sqrt(np.mean(strain * strain)))


def _boundary_offset_array(value: object) -> np.ndarray | None:
    if value is None:
        return np.zeros(1, dtype=np.float64)
    try:
        offsets = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if offsets.size == 0 or not bool(np.all(np.isfinite(offsets))):
        return None
    return offsets


def _residual_within_acceptance(residual_norm: float | None, solve_config: SolverConfig) -> bool:
    # Optimizer tolerances are not identical to the physical residual target; a
    # small acceptance band avoids rejecting numerically converged states solely
    # because the final norm was recomputed outside SciPy's stopping test.
    accepted_residual_norm = max(
        float(solve_config.max_residual) * _ACCEPTED_RESIDUAL_FACTOR,
        _ACCEPTED_RESIDUAL_FLOOR,
    )
    return bool(
        residual_norm is not None
        and np.isfinite(residual_norm)
        and residual_norm <= accepted_residual_norm
    )


def _should_use_robust_trf_loss(
    residual: np.ndarray | None,
    block_lengths: np.ndarray | None,
) -> bool:
    # The threshold is deliberately enormous: robust TRF is a last-resort guard
    # for explosive initial residuals, not the normal convergence policy.
    if residual is None or block_lengths is None:
        return False
    block_rms = _block_rms_values(residual, np.asarray(block_lengths, dtype=np.int64))
    if block_rms is None or block_rms.size == 0:
        return False
    return bool(np.median(block_rms) >= _TRF_ROBUST_BLOCK_RMS_THRESHOLD)


def _x_scale_profile_prior(name: str) -> float:
    # Priors reflect typical coefficient amplitudes by family.  They are only
    # conditioning hints for solver space and must not encode physical units.
    if name in {"h", "v", "psin"}:
        return _X_SCALE_CORE_PROFILE_PRIOR
    if name == "k":
        return _X_SCALE_KAPPA_PROFILE_PRIOR
    if name.startswith(("c", "s")):
        return _X_SCALE_FOURIER_PROFILE_PRIOR
    if name == "F":
        return _X_SCALE_F_PROFILE_PRIOR
    return _X_SCALE_F_PROFILE_PRIOR


def _build_x_block_scale_vector(operator, x_guess: np.ndarray) -> np.ndarray | None:
    x_eval = np.asarray(x_guess, dtype=np.float64)
    if not hasattr(operator, "active_profile_blocks"):
        return None

    scale = np.ones_like(x_eval)
    for _, profile_name, coeff_indices, offset, profile_scale in operator.active_profile_blocks():
        coeff_indices = np.asarray(coeff_indices, dtype=np.int64)
        length = int(coeff_indices.size)
        if length <= 0:
            continue
        if np.any(coeff_indices < 0) or np.any(coeff_indices >= x_eval.size):
            return None
        block_guess = x_eval[coeff_indices]
        guess_rms = float(np.linalg.norm(block_guess) / np.sqrt(length))
        offset_scale = 0.0 if profile_name in _X_SCALE_OFFSETLESS_PROFILES else abs(float(offset))
        profile_scale = abs(float(profile_scale))
        profile_prior = _x_scale_profile_prior(profile_name)
        if abs(profile_scale - 1.0) <= 1.0e-12:
            profile_scale = profile_prior
        # Combine model offset, declared profile scale, family prior, and current
        # guess RMS.  The maximum keeps z-space scaling stable for both zero
        # starts and warm starts with already-large coefficients.
        block_scale = max(offset_scale, profile_scale, profile_prior, guess_rms, _X_SCALE_FLOOR)
        scale[coeff_indices] = block_scale
    return scale
