"""
Module: veqpy.kernels.pareto

Role:
- Define public Pareto search result contracts and backend-neutral helpers.
- Keep cost/frontier/threshold semantics independent from backend execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from veqpy.kernels.types import KernelTopology, SolveResult
from veqpy.kernels.variant import build_kernel_variant_topology

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
    complexity: int
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


def pareto_sample_complexity(result: SolveResult, topology: KernelTopology) -> int:
    nx = int(topology.x_size)
    nx2 = nx * nx
    return (
        int(result.nfev) * nx
        + int(result.jvp_evaluations) * nx2
        + int(result.jacobian_component_evaluations) * nx2
        + int(result.linear_iterations) * nx2
    )


def coefficient_blocks_from_packed_state(
    x: np.ndarray,
    *,
    profile_names: Sequence[str],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
) -> dict[str, np.ndarray]:
    packed = np.asarray(x, dtype=np.float64)
    names = tuple(str(name) for name in profile_names)
    lengths = np.asarray(profile_L, dtype=np.int64)
    index = np.asarray(coeff_index, dtype=np.int64)
    if lengths.ndim != 1 or lengths.shape[0] != len(names):
        raise ValueError("profile_L shape does not match profile_names")
    if index.ndim != 2 or index.shape[0] != len(names):
        raise ValueError("coeff_index shape does not match profile_names")

    blocks: dict[str, np.ndarray] = {}
    for row, name in enumerate(names):
        degree = int(lengths[row])
        if degree < 0:
            continue
        positions = index[row, : degree + 1]
        if np.any(positions < 0):
            raise ValueError(f"coeff_index is missing active positions for {name!r}")
        blocks[name] = packed[positions].copy()
    return blocks


def topology_from_pareto_signature(
    capacity_topology: KernelTopology,
    signature: KernelParetoSignature,
) -> KernelTopology:
    plan = build_kernel_variant_topology(capacity_topology, **signature.to_variant_kwargs())
    if not plan.contained:
        raise ValueError("Pareto signature exceeds capacity topology")
    return plan.topology


def generate_pareto_signatures(
    capacity_topology: KernelTopology,
    *,
    strategy: str,
    coefficients_by_profile: Mapping[str, np.ndarray] | None = None,
    max_candidates: int,
) -> tuple[KernelParetoSignature, ...]:
    strategy_name = normalize_pareto_strategy(strategy)
    candidate_limit = normalize_pareto_max_candidates(max_candidates)
    if candidate_limit == 0:
        return ()

    if strategy_name == "balanced":
        candidates = _balanced_signatures(capacity_topology)
    elif strategy_name == "tail":
        candidates = _coefficient_score_signatures(
            capacity_topology,
            coefficients_by_profile,
            score_mode="tail",
        )
    elif strategy_name == "energy":
        candidates = _coefficient_score_signatures(
            capacity_topology,
            coefficients_by_profile,
            score_mode="energy",
        )
    else:
        candidates = _adaptive_seed_signatures(capacity_topology, coefficients_by_profile)

    return _unique_candidate_signatures(capacity_topology, candidates)[:candidate_limit]


def adaptive_seed_candidate_count(max_candidates: int) -> int:
    candidate_limit = normalize_pareto_max_candidates(max_candidates)
    if candidate_limit == 0:
        return 0
    return max(1, int(np.ceil(0.6 * candidate_limit)))


def generate_adaptive_refinement_signatures(
    capacity_topology: KernelTopology,
    *,
    frontier: Sequence[ParetoSample],
    seen_signatures: set[KernelParetoSignature],
    max_candidates: int,
) -> tuple[KernelParetoSignature, ...]:
    candidate_limit = normalize_pareto_max_candidates(max_candidates)
    if candidate_limit == 0:
        return ()
    names = _active_count_names(capacity_topology)
    reference_counts = _count_map_from_signature(
        KernelParetoSignature.from_topology(capacity_topology)
    )
    minimums = _minimum_counts_by_name(capacity_topology)
    candidates: list[KernelParetoSignature] = []
    seen = set(seen_signatures)

    for sample in frontier:
        counts = _count_map_from_signature(sample.signature)
        for name in names:
            current = counts.get(name, 0)
            original = reference_counts.get(name, 0)
            for delta in (-1, 1):
                trial_count = min(original, max(minimums.get(name, 0), current + delta))
                if trial_count == current:
                    continue
                trial = dict(counts)
                trial[name] = trial_count
                _ensure_one_active_count(trial, reference_counts, minimums)
                signature = _signature_from_count_map(capacity_topology, trial)
                if signature in seen:
                    continue
                seen.add(signature)
                candidates.append(signature)
                if len(candidates) >= candidate_limit:
                    return tuple(candidates)
    return tuple(candidates)


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


def _adaptive_seed_signatures(
    capacity_topology: KernelTopology,
    coefficients_by_profile: Mapping[str, np.ndarray] | None,
) -> tuple[KernelParetoSignature, ...]:
    return (
        *_balanced_signatures(capacity_topology),
        *_coefficient_score_signatures(
            capacity_topology,
            coefficients_by_profile,
            score_mode="tail",
        ),
        *_coefficient_score_signatures(
            capacity_topology,
            coefficients_by_profile,
            score_mode="energy",
        ),
    )


def _balanced_signatures(capacity_topology: KernelTopology) -> tuple[KernelParetoSignature, ...]:
    names = _active_count_names(capacity_topology)
    minimums = _minimum_counts_by_name(capacity_topology)
    reference_counts = _count_map_from_signature(
        KernelParetoSignature.from_topology(capacity_topology)
    )
    candidates: list[KernelParetoSignature] = []
    base_counts: list[dict[str, int]] = []
    for ratio in (0.25, 0.5, 0.75):
        counts = dict(reference_counts)
        for name in names:
            original = reference_counts[name]
            if original <= 0:
                continue
            counts[name] = min(
                original,
                max(minimums.get(name, 0), int(round(original * ratio))),
            )
        _ensure_one_active_count(counts, reference_counts, minimums)
        base_counts.append(counts)
        candidates.append(_signature_from_count_map(capacity_topology, counts))

    for counts in base_counts:
        for name in names:
            original = reference_counts[name]
            current = counts[name]
            for delta in (-1, 1):
                trial = dict(counts)
                trial[name] = min(original, max(minimums.get(name, 0), current + delta))
                _ensure_one_active_count(trial, reference_counts, minimums)
                candidates.append(_signature_from_count_map(capacity_topology, trial))
    return tuple(candidates)


def _coefficient_score_signatures(
    capacity_topology: KernelTopology,
    coefficients_by_profile: Mapping[str, np.ndarray] | None,
    *,
    score_mode: str,
) -> tuple[KernelParetoSignature, ...]:
    if coefficients_by_profile is None:
        raise ValueError(f"strategy={score_mode!r} requires reference coefficients")
    names = _active_count_names(capacity_topology)
    minimums = _minimum_counts_by_name(capacity_topology)
    reference_counts = _count_map_from_signature(
        KernelParetoSignature.from_topology(capacity_topology)
    )

    score_values: list[float] = []
    score_tables: dict[str, dict[int, float]] = {}
    for name in names:
        original = reference_counts[name]
        coeff = np.asarray(coefficients_by_profile.get(name, ()), dtype=np.float64)
        if coeff.size != original:
            coeff = np.zeros(original, dtype=np.float64)
        table: dict[int, float] = {}
        for keep in range(minimums.get(name, 0), original + 1):
            score = _tail_score(coeff[keep:], score_mode=score_mode)
            table[keep] = score
            if keep < original and np.isfinite(score):
                score_values.append(score)
        score_tables[name] = table

    if not score_values:
        return ()

    finite_scores = np.asarray(score_values, dtype=np.float64)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if finite_scores.size == 0:
        return ()
    thresholds = tuple(
        float(value)
        for value in np.unique(np.quantile(finite_scores, [0.0, 0.25, 0.5, 0.75, 1.0]))
    )

    candidates: list[KernelParetoSignature] = []
    for threshold in thresholds:
        counts = dict(reference_counts)
        for name in names:
            original = reference_counts[name]
            table = score_tables[name]
            keep = original
            for candidate_keep in range(minimums.get(name, 0), original + 1):
                if table[candidate_keep] <= threshold:
                    keep = candidate_keep
                    break
            counts[name] = keep
        _ensure_one_active_count(counts, reference_counts, minimums)
        candidates.append(_signature_from_count_map(capacity_topology, counts))
    return tuple(candidates)


def _tail_score(values: np.ndarray, *, score_mode: str) -> float:
    if values.size == 0:
        return 0.0
    if score_mode == "tail":
        return float(np.max(np.abs(values)))
    if score_mode == "energy":
        return float(np.sqrt(np.sum(values * values)))
    raise ValueError(f"Unknown Pareto score mode {score_mode!r}")


def _unique_candidate_signatures(
    capacity_topology: KernelTopology,
    candidates: Sequence[KernelParetoSignature],
) -> tuple[KernelParetoSignature, ...]:
    reference = KernelParetoSignature.from_topology(capacity_topology)
    unique: dict[KernelParetoSignature, KernelParetoSignature] = {}
    for candidate in candidates:
        topology = topology_from_pareto_signature(capacity_topology, candidate)
        canonical = KernelParetoSignature.from_topology(topology)
        if canonical == reference:
            continue
        unique.setdefault(canonical, canonical)
    return tuple(
        sorted(
            unique,
            key=lambda signature: (
                _signature_count_total(signature),
                signature.h_count,
                signature.v_count,
                signature.kappa_count,
                signature.psin_count,
                signature.F_count,
                signature.c_counts,
                signature.s_counts,
            ),
        )
    )


def _active_count_names(topology: KernelTopology) -> tuple[str, ...]:
    return tuple(name for name, count in topology.active_profiles if int(count) > 0)


def _minimum_counts_by_name(topology: KernelTopology) -> dict[str, int]:
    minimums: dict[str, int] = {}
    if topology.source_active_family == "psin":
        minimums["psin"] = 1
    if topology.source_active_family == "F":
        minimums["F"] = 1
    return minimums


def _count_map_from_signature(signature: KernelParetoSignature) -> dict[str, int]:
    counts = {
        "h": int(signature.h_count),
        "v": int(signature.v_count),
        "k": int(signature.kappa_count),
        "psin": int(signature.psin_count),
        "F": int(signature.F_count),
    }
    counts.update({f"c{order}": int(count) for order, count in enumerate(signature.c_counts)})
    counts.update(
        {f"s{order}": int(count) for order, count in enumerate(signature.s_counts, start=1)}
    )
    return counts


def _signature_from_count_map(
    capacity_topology: KernelTopology,
    counts: Mapping[str, int],
) -> KernelParetoSignature:
    c_highest = max(
        (int(name[1:]) for name, count in counts.items() if _is_c_profile(name) and count > 0),
        default=-1,
    )
    s_highest = max(
        (int(name[1:]) for name, count in counts.items() if _is_s_profile(name) and count > 0),
        default=0,
    )
    c_counts = tuple(int(counts.get(f"c{order}", 0)) for order in range(c_highest + 1))
    s_counts = tuple(int(counts.get(f"s{order}", 0)) for order in range(1, s_highest + 1))
    signature = KernelParetoSignature(
        h_count=int(counts.get("h", 0)),
        v_count=int(counts.get("v", 0)),
        kappa_count=int(counts.get("k", 0)),
        psin_count=int(counts.get("psin", 0)),
        F_count=int(counts.get("F", 0)),
        c_counts=c_counts,
        s_counts=s_counts,
    )
    return KernelParetoSignature.from_topology(
        topology_from_pareto_signature(capacity_topology, signature)
    )


def _ensure_one_active_count(
    counts: dict[str, int],
    reference_counts: Mapping[str, int],
    minimums: Mapping[str, int],
) -> None:
    if any(count > 0 for count in counts.values()):
        return
    for name in sorted(minimums):
        counts[name] = max(1, int(minimums[name]))
        return
    for name, count in reference_counts.items():
        if count > 0:
            counts[name] = 1
            return


def _signature_count_total(signature: KernelParetoSignature) -> int:
    return (
        signature.h_count
        + signature.v_count
        + signature.kappa_count
        + signature.psin_count
        + signature.F_count
        + sum(signature.c_counts)
        + sum(signature.s_counts)
    )


def _is_c_profile(name: str) -> bool:
    return name.startswith("c") and name[1:].isdigit()


def _is_s_profile(name: str) -> bool:
    return name.startswith("s") and name[1:].isdigit()


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
