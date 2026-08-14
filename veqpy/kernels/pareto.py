"""
Module: veqpy.kernels.pareto

Role:
- Define public Pareto evaluator result contracts and backend-neutral helpers.
- Keep cost/frontier/threshold post-processing independent from backend execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from veqpy.kernels.types import KernelTopology, SolveResult
from veqpy.kernels.variant import build_kernel_variant_topology

PARETO_TARGET_OPTIONS = ("counts", "time", "complexity")
PARETO_METRIC_OPTIONS = ("rms", "max")
PARETO_STRATEGY_OPTIONS = ("tail", "energy", "adaptive", "balanced")
_REF_PRUNE_SAMPLE_COUNT = 512
_REF_PRUNE_CORE_GRID_LIMIT = 12_000
_REF_PRUNE_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)


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
    """One solved topology sample in a Kernel Pareto evaluation."""

    topology: KernelTopology
    signature: KernelParetoSignature
    counts: int
    time: float
    complexity: int
    shape_error: float
    result: SolveResult


@dataclass(frozen=True, slots=True)
class ParetoResult:
    """Result from evaluating explicit reduced topologies against one reference."""

    reference: ParetoSample
    samples: tuple[ParetoSample, ...]
    frontier: tuple[ParetoSample, ...]
    target: str
    metric: str


def normalize_pareto_target(value: str) -> str:
    return _normalize_option(value, PARETO_TARGET_OPTIONS, "target")


def normalize_pareto_metric(value: str) -> str:
    return _normalize_option(value, PARETO_METRIC_OPTIONS, "metric")


def normalize_pareto_strategy(value: str) -> str:
    return _normalize_option(value, PARETO_STRATEGY_OPTIONS, "strategy")


def normalize_pareto_neighborod_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("neighborod_size must be an integer")
    if value < 0:
        raise ValueError("neighborod_size must be non-negative")
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


def pareto_signature_from_candidate(
    capacity_topology: KernelTopology,
    candidate: KernelTopology | KernelParetoSignature | Mapping[str, object],
) -> KernelParetoSignature:
    """Canonicalize one user-supplied reduced topology candidate."""

    if isinstance(candidate, KernelParetoSignature):
        topology = topology_from_pareto_signature(capacity_topology, candidate)
        return KernelParetoSignature.from_topology(topology)
    if isinstance(candidate, KernelTopology):
        topology = topology_from_pareto_signature(
            capacity_topology,
            KernelParetoSignature.from_topology(candidate),
        )
        return KernelParetoSignature.from_topology(topology)
    if isinstance(candidate, Mapping):
        return _signature_from_count_map(
            capacity_topology,
            _count_map_from_candidate_mapping(candidate),
        )
    raise TypeError(
        "Pareto candidates must be KernelTopology, KernelParetoSignature, or mappings"
    )


def normalize_pareto_candidates(
    capacity_topology: KernelTopology,
    candidates: Sequence[KernelTopology | KernelParetoSignature | Mapping[str, object]]
    | KernelTopology
    | KernelParetoSignature
    | Mapping[str, object],
) -> tuple[KernelParetoSignature, ...]:
    """Return unique non-reference candidate signatures in caller order."""

    if isinstance(candidates, KernelTopology | KernelParetoSignature) or isinstance(
        candidates,
        Mapping,
    ):
        raw_candidates = (candidates,)
    elif isinstance(candidates, Sequence) and not isinstance(candidates, str):
        raw_candidates = tuple(candidates)
    else:
        raise TypeError("candidates must be a candidate or a sequence of candidates")

    reference = KernelParetoSignature.from_topology(capacity_topology)
    unique: dict[KernelParetoSignature, KernelParetoSignature] = {}
    for candidate in raw_candidates:
        signature = pareto_signature_from_candidate(capacity_topology, candidate)
        if signature == reference:
            continue
        unique.setdefault(signature, signature)
    return tuple(unique.values())


def generate_pareto_signatures(
    capacity_topology: KernelTopology,
    *,
    strategy: str,
    coefficients_by_profile: Mapping[str, np.ndarray] | None = None,
) -> tuple[KernelParetoSignature, ...]:
    strategy_name = normalize_pareto_strategy(strategy)

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

    return _unique_candidate_signatures(capacity_topology, candidates)


def generate_adaptive_refinement_signatures(
    capacity_topology: KernelTopology,
    *,
    frontier: Sequence[ParetoSample],
    seen_signatures: set[KernelParetoSignature],
    neighborod_size: int,
) -> tuple[KernelParetoSignature, ...]:
    radius = normalize_pareto_neighborod_size(neighborod_size)
    if radius == 0:
        return ()
    axes = _refinement_count_axes(capacity_topology)
    reference_counts = _count_map_from_signature(
        KernelParetoSignature.from_topology(capacity_topology)
    )
    minimums = _minimum_counts_by_name(capacity_topology)
    candidates: list[KernelParetoSignature] = []
    seen = set(seen_signatures)

    for sample in frontier:
        counts = _count_map_from_signature(sample.signature)
        for axis in axes:
            for delta in range(-radius, radius + 1):
                if delta == 0:
                    continue
                trial = dict(counts)
                changed = False
                for name in axis:
                    current = counts.get(name, 0)
                    original = reference_counts.get(name, 0)
                    trial_count = min(
                        original,
                        max(minimums.get(name, 0), current + delta),
                    )
                    if trial_count != current:
                        changed = True
                    trial[name] = trial_count
                if not changed:
                    continue
                _ensure_one_active_count(trial, reference_counts, minimums)
                signature = _signature_from_count_map(capacity_topology, trial)
                if signature in seen:
                    continue
                seen.add(signature)
                candidates.append(signature)
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
    target: str,
) -> tuple[ParetoSample, ...]:
    cost_name = normalize_pareto_target(target)
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
    target: str,
) -> dict[float, ParetoSample]:
    cost_name = normalize_pareto_target(target)
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
        *_ref_prune_seed_signatures(capacity_topology, coefficients_by_profile),
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


def _ref_prune_seed_signatures(
    capacity_topology: KernelTopology,
    coefficients_by_profile: Mapping[str, np.ndarray] | None,
) -> tuple[KernelParetoSignature, ...]:
    reference_counts = _count_map_from_signature(
        KernelParetoSignature.from_topology(capacity_topology)
    )
    minimums = _minimum_counts_by_name(capacity_topology)
    names = tuple(name for name, count in reference_counts.items() if count > 0)
    candidates: list[KernelParetoSignature] = []
    structural_floor = _seed_floor_counts(reference_counts, minimums)
    candidates.append(_signature_from_count_map(capacity_topology, structural_floor))
    for name in names:
        current = int(structural_floor.get(name, minimums.get(name, 0)))
        if current >= int(reference_counts[name]):
            continue
        counts = dict(structural_floor)
        counts[name] = current + 1
        candidates.append(_signature_from_count_map(capacity_topology, counts))

    for ratio in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875):
        counts = dict(structural_floor)
        for name in names:
            original = reference_counts[name]
            floor = structural_floor.get(name, minimums.get(name, 0))
            value = int(round(original * ratio))
            if name in structural_floor and original > 0:
                value = max(1, value)
            counts[name] = min(original, max(floor, value))
        candidates.append(_signature_from_count_map(capacity_topology, counts))

    if coefficients_by_profile is not None:
        for threshold in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999):
            counts = dict(structural_floor)
            for name in names:
                coeff = np.asarray(coefficients_by_profile.get(name, ()), dtype=np.float64)
                original = reference_counts[name]
                if coeff.size != original:
                    value = int(round(original * threshold))
                else:
                    value = _energy_retention_count(coeff, threshold)
                floor = structural_floor.get(name, minimums.get(name, 0))
                counts[name] = min(original, max(floor, value))
            candidates.append(_signature_from_count_map(capacity_topology, counts))

    candidates.extend(
        _structured_ref_prune_seed_signatures(
            capacity_topology,
            reference_counts=reference_counts,
            structural_floor=structural_floor,
        )
    )

    core_names = tuple(
        name for name in ("psin", "h", "k", "v", "F") if int(reference_counts.get(name, 0)) > 0
    )
    shell_names = _fourier_shell_names(reference_counts)
    for index in range(1, _REF_PRUNE_SAMPLE_COUNT + 1):
        counts = dict(structural_floor)
        for offset, name in enumerate(core_names):
            floor = structural_floor.get(name, minimums.get(name, 0))
            counts[name] = _stratified_count(
                index,
                _REF_PRUNE_BASES[offset % len(_REF_PRUNE_BASES)],
                floor,
                reference_counts[name],
            )
        if shell_names:
            depth = _stratified_count(
                index,
                _REF_PRUNE_BASES[5],
                1,
                len(shell_names),
            )
            previous_cap = max(reference_counts[name] for name in shell_names[0])
            for shell_index, shell in enumerate(shell_names[:depth]):
                floor = max(structural_floor.get(name, 0) for name in shell)
                floor = max(1, floor)
                cap = min(previous_cap, min(reference_counts[name] for name in shell))
                if floor > cap:
                    break
                level = _stratified_count(
                    index,
                    _REF_PRUNE_BASES[(6 + shell_index) % len(_REF_PRUNE_BASES)],
                    floor,
                    cap,
                )
                for name in shell:
                    counts[name] = min(reference_counts[name], level)
                previous_cap = level
        candidates.append(_signature_from_count_map(capacity_topology, counts))
    return tuple(candidates)


def _structured_ref_prune_seed_signatures(
    capacity_topology: KernelTopology,
    *,
    reference_counts: Mapping[str, int],
    structural_floor: Mapping[str, int],
) -> tuple[KernelParetoSignature, ...]:
    core_names = tuple(
        name for name in ("psin", "h", "k", "v", "F") if int(reference_counts.get(name, 0)) > 0
    )
    core_grid = _structured_core_count_grid(reference_counts, structural_floor, core_names)
    shell_names = _fourier_shell_names(reference_counts)
    candidates: list[KernelParetoSignature] = []

    for core_counts in core_grid:
        ratio = _normalized_core_ratio(core_counts, reference_counts, structural_floor)
        spread = _normalized_core_spread(core_counts, reference_counts, structural_floor)
        pattern_maps = _structured_fourier_pattern_maps(
            reference_counts,
            shell_names=shell_names,
            ratio=ratio,
            spread=spread,
        )
        for pattern in pattern_maps:
            counts = dict(structural_floor)
            counts.update(core_counts)
            counts.update(pattern)
            _ensure_one_active_count(counts, reference_counts, structural_floor)
            candidates.append(_signature_from_count_map(capacity_topology, counts))
    return tuple(candidates)


def _structured_core_count_grid(
    reference_counts: Mapping[str, int],
    structural_floor: Mapping[str, int],
    core_names: Sequence[str],
) -> tuple[dict[str, int], ...]:
    names = tuple(core_names)
    if not names:
        return ({},)

    full_product = 1
    for name in names:
        floor = int(structural_floor.get(name, 0))
        ceiling = int(reference_counts[name])
        full_product *= max(1, ceiling - floor + 1)

    ranges: list[tuple[int, ...]] = []
    for name in names:
        floor = int(structural_floor.get(name, 0))
        ceiling = int(reference_counts[name])
        if full_product <= _REF_PRUNE_CORE_GRID_LIMIT:
            values = tuple(range(floor, ceiling + 1))
        else:
            values = _count_breakpoints(floor, ceiling)
        ranges.append(values)

    grid: list[dict[str, int]] = []
    partial: dict[str, int] = {}

    def visit(offset: int) -> None:
        if offset == len(names):
            grid.append(dict(partial))
            return
        name = names[offset]
        for value in ranges[offset]:
            partial[name] = int(value)
            visit(offset + 1)
        partial.pop(name, None)

    visit(0)
    return tuple(grid)


def _count_breakpoints(lower: int, upper: int) -> tuple[int, ...]:
    lo = int(lower)
    hi = int(upper)
    if hi <= lo:
        return (lo,)
    values = {lo, hi, max(lo, min(hi, 1))}
    for ratio in (0.125, 0.2, 0.25, 1.0 / 3.0, 0.4, 0.5, 0.6, 2.0 / 3.0, 0.75, 0.875):
        values.add(min(hi, max(lo, int(round(hi * ratio)))))
    return tuple(sorted(values))


def _normalized_core_ratio(
    core_counts: Mapping[str, int],
    reference_counts: Mapping[str, int],
    structural_floor: Mapping[str, int],
) -> float:
    ratios = [
        _normalized_count_position(
            int(value),
            int(structural_floor.get(name, 0)),
            int(reference_counts[name]),
        )
        for name, value in core_counts.items()
    ]
    if not ratios:
        return 0.0
    return float(np.mean(np.asarray(ratios, dtype=np.float64)))


def _normalized_core_spread(
    core_counts: Mapping[str, int],
    reference_counts: Mapping[str, int],
    structural_floor: Mapping[str, int],
) -> float:
    ratios = [
        _normalized_count_position(
            int(value),
            int(structural_floor.get(name, 0)),
            int(reference_counts[name]),
        )
        for name, value in core_counts.items()
    ]
    if len(ratios) <= 1:
        return 0.0
    return float(max(ratios) - min(ratios))


def _normalized_count_position(value: int, lower: int, upper: int) -> float:
    if upper <= lower:
        return 0.0
    return min(1.0, max(0.0, float(value - lower) / float(upper - lower)))


def _structured_fourier_pattern_maps(
    reference_counts: Mapping[str, int],
    *,
    shell_names: Sequence[tuple[str, ...]],
    ratio: float,
    spread: float,
) -> tuple[dict[str, int], ...]:
    if not shell_names:
        return ({},)

    clamped_ratio = min(1.0, max(0.0, float(ratio)))
    candidates = (
        _banded_fourier_pattern(
            reference_counts,
            shell_names=shell_names,
            ratio=clamped_ratio,
            spread=float(spread),
        ),
    )

    unique: dict[tuple[tuple[str, int], ...], dict[str, int]] = {}
    for candidate in candidates:
        normalized = {
            name: min(int(reference_counts[name]), max(0, int(count)))
            for name, count in candidate.items()
            if int(reference_counts.get(name, 0)) > 0 and int(count) > 0
        }
        unique.setdefault(tuple(sorted(normalized.items())), normalized)
    return tuple(unique.values()) or ({},)


def _banded_fourier_pattern(
    reference_counts: Mapping[str, int],
    *,
    shell_names: Sequence[tuple[str, ...]],
    ratio: float,
    spread: float,
) -> dict[str, int]:
    first_cap = max(int(reference_counts[name]) for name in shell_names[0])
    tail_cap = _tail_shell_capacity(reference_counts, shell_names)
    low_head = max(1, int(round(0.2 * first_cap)))
    mid_head = max(low_head, int(round(0.4 * first_cap)))
    high_head = max(mid_head, int(round(0.6 * first_cap)))
    near_head = max(high_head, int(round(0.9 * first_cap)))
    low_mid = max(1, int(round(0.4 * tail_cap)))
    high_mid = max(low_mid, int(round(0.6 * tail_cap)))

    if ratio < 0.30:
        levels = (low_head, 1)
    elif ratio < 0.40 and spread < 0.30:
        levels = (low_head, low_mid, low_mid, 1)
    elif ratio < 0.40:
        levels = (mid_head, 1, 1)
    elif ratio < 0.55:
        levels = (mid_head, low_mid, 1, 1, 1)
    elif ratio < 0.72:
        levels = (high_head, tail_cap, high_mid, low_mid, low_mid, 1)
    else:
        levels = (near_head, tail_cap, tail_cap, tail_cap, tail_cap, low_mid, 1)
    return _shell_pattern_from_levels(reference_counts, shell_names, levels)


def _tail_shell_capacity(
    reference_counts: Mapping[str, int],
    shell_names: Sequence[tuple[str, ...]],
) -> int:
    if len(shell_names) <= 1:
        return max(int(reference_counts[name]) for name in shell_names[0])
    return max(
        min(int(reference_counts[name]) for name in shell)
        for shell in shell_names[1:]
        if shell
    )


def _shell_pattern_from_levels(
    reference_counts: Mapping[str, int],
    shell_names: Sequence[tuple[str, ...]],
    levels: Sequence[int],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    previous = max(int(reference_counts[name]) for shell in shell_names for name in shell)
    for shell, raw_level in zip(shell_names, levels, strict=False):
        level = min(previous, max(1, int(round(raw_level))))
        changed = False
        for name in shell:
            count = min(int(reference_counts[name]), level)
            if count > 0:
                counts[name] = count
                changed = True
        if not changed:
            break
        previous = level
    return counts


def _seed_floor_counts(
    reference_counts: Mapping[str, int],
    minimums: Mapping[str, int],
) -> dict[str, int]:
    floors = {name: int(value) for name, value in minimums.items() if int(value) > 0}
    for name in ("h", "k", "v", "c0", "s1"):
        if int(reference_counts.get(name, 0)) > 0:
            floors.setdefault(name, 1)
    return floors


def _energy_retention_count(coefficients: np.ndarray, threshold: float) -> int:
    values = np.asarray(coefficients, dtype=np.float64)
    if values.size == 0:
        return 0
    energy = values * values
    total = float(np.sum(energy))
    if total <= 0.0 or not np.isfinite(total):
        return 1
    cumulative = np.cumsum(energy) / total
    return int(np.searchsorted(cumulative, float(threshold), side="left") + 1)


def _fourier_shell_names(reference_counts: Mapping[str, int]) -> tuple[tuple[str, ...], ...]:
    shells: list[tuple[str, ...]] = []
    order = 0
    while True:
        shell = tuple(
            name
            for name in (f"c{order}", f"s{order + 1}")
            if int(reference_counts.get(name, 0)) > 0
        )
        if not shell:
            break
        shells.append(shell)
        order += 1
    return tuple(shells)


def _refinement_count_axes(topology: KernelTopology) -> tuple[tuple[str, ...], ...]:
    reference_counts = _count_map_from_signature(KernelParetoSignature.from_topology(topology))
    scalar_axes = tuple(
        (name,)
        for name in ("psin", "h", "k", "v", "F")
        if int(reference_counts.get(name, 0)) > 0
    )
    shell_axes = _fourier_shell_names(reference_counts)
    individual_fourier_axes = tuple(
        (name,)
        for shell in shell_axes
        for name in shell
        if int(reference_counts.get(name, 0)) > 0
    )
    return (*scalar_axes, *shell_axes, *individual_fourier_axes)


def _radical_inverse(index: int, base: int) -> float:
    inverse = 0.0
    fraction = 1.0 / float(base)
    current = int(index)
    while current > 0:
        inverse += float(current % int(base)) * fraction
        current //= int(base)
        fraction /= float(base)
    return inverse


def _stratified_count(index: int, base: int, lower: int, upper: int) -> int:
    lower_int = int(lower)
    upper_int = int(upper)
    if upper_int <= lower_int:
        return lower_int
    span = upper_int - lower_int + 1
    value = lower_int + int(np.floor(_radical_inverse(index, base) * span))
    return min(upper_int, max(lower_int, value))


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


def _count_map_from_candidate_mapping(candidate: Mapping[str, object]) -> dict[str, int]:
    field_names = {
        "h_count": "h",
        "v_count": "v",
        "kappa_count": "k",
        "k_count": "k",
        "psin_count": "psin",
        "F_count": "F",
    }
    profile_names = {"h", "v", "k", "kappa", "psin", "F"}
    counts: dict[str, int] = {}
    for raw_name, raw_value in candidate.items():
        name = str(raw_name)
        if name in field_names:
            counts[field_names[name]] = _candidate_count_value(raw_value, name)
        elif name == "c_counts":
            for order, value in enumerate(_candidate_count_sequence(raw_value, name)):
                counts[f"c{order}"] = value
        elif name == "s_counts":
            for order, value in enumerate(_candidate_count_sequence(raw_value, name), start=1):
                counts[f"s{order}"] = value
        elif name in profile_names:
            counts["k" if name == "kappa" else name] = _candidate_count_value(raw_value, name)
        elif _is_c_profile(name) or _is_s_profile(name):
            counts[name] = _candidate_count_value(raw_value, name)
        else:
            raise ValueError(f"unknown Pareto candidate count field {name!r}")
    return counts


def _candidate_count_sequence(value: object, name: str) -> tuple[int, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of non-negative integers")
    return tuple(_candidate_count_value(item, name) for item in value)


def _candidate_count_value(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be a non-negative integer")
    count = int(value)
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
    return count


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
