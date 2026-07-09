"""
Module: veqpy.kernels.pareto

Role:
- Define public Pareto search result contracts and backend-neutral helpers.
- Keep cost/frontier/threshold semantics independent from backend execution.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from veqpy.kernels.types import KernelTopology, SolveResult

PARETO_BY_OPTIONS = ("counts", "time", "complexity")
PARETO_METRIC_OPTIONS = ("rms", "max")
PARETO_STRATEGY_OPTIONS = ("tail", "energy", "adaptive", "balanced")


@dataclass(frozen=True, slots=True)
class KernelParetoSignature:
    """Count-only topology signature for one Pareto candidate."""

    h_count: int
    v_count: int
    kappa_count: int
    psin_count: int
    F_count: int
    c_counts: tuple[int, ...]
    s_counts: tuple[int, ...]

    @classmethod
    def from_topology(cls, topology: KernelTopology) -> "KernelParetoSignature":
        return cls(
            h_count=int(topology.h_count),
            v_count=int(topology.v_count),
            kappa_count=int(topology.kappa_count),
            psin_count=int(topology.psin_count),
            F_count=int(topology.F_count),
            c_counts=tuple(int(value) for value in topology.c_counts),
            s_counts=tuple(int(value) for value in topology.s_counts),
        )

    def to_variant_kwargs(self) -> dict[str, object]:
        return {
            "h_count": self.h_count,
            "v_count": self.v_count,
            "kappa_count": self.kappa_count,
            "psin_count": self.psin_count,
            "F_count": self.F_count,
            "c_counts": self.c_counts,
            "s_counts": self.s_counts,
        }


@dataclass(frozen=True, slots=True)
class ParetoSample:
    """One solved topology sample in a Kernel Pareto search."""

    topology: KernelTopology
    signature: KernelParetoSignature
    counts: int
    time: float
    complexity: float
    shape_error: float
    result: SolveResult


@dataclass(frozen=True, slots=True)
class ParetoResult:
    """Full Kernel Pareto search result."""

    reference: ParetoSample
    samples: tuple[ParetoSample, ...]
    frontier: tuple[ParetoSample, ...]
    selected: dict[float, ParetoSample]
    thresholds: tuple[float, ...]
    pareto_by: str
    metric: str
    strategy: str
    max_candidates: int


def normalize_pareto_by(value: str) -> str:
    return _normalize_option(value, PARETO_BY_OPTIONS, "pareto_by")


def normalize_pareto_metric(value: str) -> str:
    return _normalize_option(value, PARETO_METRIC_OPTIONS, "metric")


def normalize_pareto_strategy(value: str) -> str:
    return _normalize_option(value, PARETO_STRATEGY_OPTIONS, "strategy")


def normalize_pareto_max_candidates(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_candidates must be an integer")
    if value < 0:
        raise ValueError("max_candidates must be non-negative")
    return int(value)


def normalize_shape_error_thresholds(
    max_shape_error: float | Sequence[float] | None,
) -> tuple[float, ...]:
    if max_shape_error is None:
        return ()
    if isinstance(max_shape_error, str):
        raise TypeError("max_shape_error must be a float, a sequence of floats, or None")
    if _is_number(max_shape_error):
        return (_finite_nonnegative_float(max_shape_error, "max_shape_error"),)
    if not isinstance(max_shape_error, Sequence):
        raise TypeError("max_shape_error must be a float, a sequence of floats, or None")
    return tuple(
        _finite_nonnegative_float(value, "max_shape_error")
        for value in max_shape_error
    )


def pareto_sample_complexity(result: SolveResult, topology: KernelTopology) -> float:
    nx = int(topology.x_size)
    nx2 = nx * nx
    return float(
        int(result.nfev) * nx
        + int(result.jvp_evaluations) * nx2
        + int(result.jacobian_component_evaluations) * nx2
        + int(result.linear_iterations) * nx2
    )


def pareto_shape_error(reference_R: np.ndarray, candidate_R: np.ndarray, *, metric: str) -> float:
    metric_name = normalize_pareto_metric(metric)
    reference = np.asarray(reference_R, dtype=np.float64)
    candidate = np.asarray(candidate_R, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"R surface shape mismatch: reference {reference.shape}, candidate {candidate.shape}"
        )
    if reference.size == 0:
        raise ValueError("R surfaces must be non-empty")
    diff = candidate - reference
    if metric_name == "rms":
        return float(np.sqrt(np.mean(diff * diff)))
    return float(np.max(np.abs(diff)))


def pareto_frontier(
    samples: Sequence[ParetoSample],
    *,
    pareto_by: str,
) -> tuple[ParetoSample, ...]:
    cost_name = normalize_pareto_by(pareto_by)
    valid = [
        sample
        for sample in samples
        if bool(sample.result.success)
        and np.isfinite(float(sample.shape_error))
        and np.isfinite(_sample_cost(sample, cost_name))
    ]
    ordered = sorted(
        valid,
        key=lambda sample: (
            _sample_cost(sample, cost_name),
            float(sample.shape_error),
            int(sample.counts),
        ),
    )

    frontier: list[ParetoSample] = []
    best_error = float("inf")
    for sample in ordered:
        error = float(sample.shape_error)
        if error < best_error:
            frontier.append(sample)
            best_error = error
    return tuple(frontier)


def select_pareto_thresholds(
    frontier: Sequence[ParetoSample],
    thresholds: Sequence[float],
    *,
    pareto_by: str,
) -> dict[float, ParetoSample]:
    cost_name = normalize_pareto_by(pareto_by)
    normalized_thresholds = normalize_shape_error_thresholds(tuple(thresholds))
    selected: dict[float, ParetoSample] = {}
    for threshold in normalized_thresholds:
        candidates = [
            sample
            for sample in frontier
            if bool(sample.result.success)
            and np.isfinite(float(sample.shape_error))
            and float(sample.shape_error) <= threshold
        ]
        if not candidates:
            continue
        selected[threshold] = min(
            candidates,
            key=lambda sample: (
                _sample_cost(sample, cost_name),
                float(sample.shape_error),
                int(sample.counts),
            ),
        )
    return selected


def _normalize_option(value: str, options: tuple[str, ...], name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().lower()
    if normalized in options:
        return normalized
    available = ", ".join(options)
    raise ValueError(f"{name} must be one of {available}")


def _sample_cost(sample: ParetoSample, name: str) -> float:
    return float(getattr(sample, name))


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float | np.floating)


def _finite_nonnegative_float(value: object, name: str) -> float:
    if not _is_number(value):
        raise TypeError(f"{name} values must be real numbers")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} values must be finite")
    if numeric < 0.0:
        raise ValueError(f"{name} values must be non-negative")
    return numeric
