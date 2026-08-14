#!/usr/bin/env python3
"""Numba Kernel.pareto() screening benchmark on GEQDSK Ref topologies."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.table import Table
from rich.text import Text

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks._common import (
    CASE_KEYS,
    REFERENCE_LAYOUT_NR,
    REFERENCE_LAYOUT_NT,
    REPO_ROOT,
    RouteBenchmarkSpec,
    benchmark_result_path,
    cpu_affinity,
    geqdsk_kernel_case,
    runtime_env,
    selected_cases,
    topology_profile_counts,
    write_json,
)
from benchmarks._reporting import (
    REPORT_TABLE_BOX,
    format_optional_float,
    format_optional_sci,
    print_config_tree,
    print_outputs_tree,
    progress_context,
    progress_phase,
    status_cell,
)
from benchmarks._reporting import (
    console as reporting_console,
)
from veqpy import Kernel, KernelRecipe, KernelTopology, ParetoResult, ParetoSample
from veqpy.kernels.pareto import select_pareto_thresholds

DEFAULT_OUTPUT = benchmark_result_path("numba_pareto")
DEFAULT_THRESHOLD_SCALES = (1.0e-2, 5.0e-3, 1.0e-3)
DEFAULT_MAX_EVALUATIONS = 2000
SWEEP_MODES = ("partial", "full")
DEFAULT_SWEEP_MODE = "full"
FULL_SWEEP_SIGNATURE_VERSION = "representative-pruned-v2"
FULL_SWEEP_MAX_CONFIGS_PER_CASE = 10000
REFERENCE_LAYOUT_M_MAX = 10
D_SHAPE_FULL_SINE_SAMPLES_PER_CORE = 8
GENERAL_REF_PRUNE_CORE_SAMPLE_COUNT = 4800
GENERAL_REF_PRUNE_FOURIER_SAMPLES_PER_CORE = 2
CORE_FAMILIES = ("psin", "h", "k", "v")
REF_PRUNE_HALTON_BASES = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
)
D_SHAPE_MIN_LENGTHS = {
    "psin": 1,
    "h": 1,
    "k": 1,
    "s1": 1,
}
H_MODE_MIN_LENGTHS = {
    "psin": 1,
    "h": 1,
    "k": 1,
    "v": 1,
    "c0": 1,
    "s1": 1,
}
X_POINT_MIN_LENGTHS = {
    "psin": 1,
    "h": 1,
    "k": 1,
    "v": 1,
    "c0": 1,
    "s1": 1,
}
CASE_MIN_LENGTHS = {
    "solovev": D_SHAPE_MIN_LENGTHS,
    "chease": H_MODE_MIN_LENGTHS,
    "efit": X_POINT_MIN_LENGTHS,
}
TABLE05_SELECTED_SIGNATURES: dict[str, tuple[dict[str, int], ...]] = {
    "solovev": (
        {"psin": 1, "h": 1, "k": 1, "s1": 1},
        {"psin": 1, "h": 1, "k": 2, "s1": 1},
        {"psin": 4, "h": 2, "k": 2, "s1": 2},
    ),
    "chease": (
        {
            "psin": 6,
            "h": 6,
            "k": 4,
            "v": 1,
            "c0": 4,
            "s1": 3,
            "c1": 1,
            "s2": 1,
            "c2": 1,
            "s3": 1,
        },
        {
            "psin": 3,
            "h": 8,
            "k": 5,
            "v": 5,
            "c0": 4,
            "s1": 4,
            "c1": 2,
            "s2": 2,
            "c2": 1,
            "s3": 1,
            "c3": 1,
            "s4": 1,
            "c4": 1,
            "s5": 1,
        },
        {
            "psin": 8,
            "h": 7,
            "k": 6,
            "v": 7,
            "c0": 6,
            "s1": 6,
            "c1": 5,
            "s2": 5,
            "c2": 3,
            "s3": 3,
            "c3": 2,
            "s4": 2,
            "c4": 2,
            "s5": 2,
            "c5": 1,
            "s6": 1,
        },
    ),
    "efit": (
        {
            "psin": 4,
            "h": 5,
            "k": 3,
            "v": 2,
            "c0": 2,
            "s1": 2,
            "c1": 1,
            "s2": 1,
        },
        {
            "psin": 3,
            "h": 4,
            "k": 4,
            "v": 5,
            "c0": 2,
            "s1": 2,
            "c1": 2,
            "s2": 2,
            "c2": 2,
            "s3": 2,
            "c3": 1,
            "s4": 1,
        },
        {
            "psin": 7,
            "h": 8,
            "k": 9,
            "v": 7,
            "c0": 9,
            "s1": 9,
            "c1": 5,
            "s2": 5,
            "c2": 5,
            "s3": 5,
            "c3": 5,
            "s4": 5,
            "c4": 5,
            "s5": 5,
            "c5": 2,
            "s6": 2,
            "c6": 1,
            "s7": 1,
        },
    ),
}


@dataclass(frozen=True, slots=True)
class SignatureRecord:
    strategy_name: str
    strategy_names: tuple[str, ...]
    sweep_step: str
    signature: dict[str, int]


def _script_candidate_records(
    case_key: str,
    topology: KernelTopology,
    *,
    sweep_mode: str,
) -> list[SignatureRecord]:
    min_lengths, max_lengths = _case_length_bounds(case_key, topology)
    records = _generate_case_signatures(
        case_key,
        min_lengths,
        max_lengths,
        sweep_mode=sweep_mode,
    )
    if sweep_mode == "full" and len(records) >= FULL_SWEEP_MAX_CONFIGS_PER_CASE:
        raise ValueError(
            f"{case_key} full sweep generated {len(records)} configs; "
            f"expected fewer than {FULL_SWEEP_MAX_CONFIGS_PER_CASE}"
        )
    return records


def _case_length_bounds(
    case_key: str,
    topology: KernelTopology,
) -> tuple[dict[str, int], dict[str, int]]:
    try:
        raw_min = CASE_MIN_LENGTHS[case_key]
    except KeyError as exc:
        raise KeyError(f"Missing configured min-length bounds for case {case_key!r}") from exc
    raw_max = _topology_max_lengths(topology)
    min_lengths = _normalize_length_dict(raw_min, label=f"{case_key} min-lengths")
    max_lengths = _normalize_length_dict(raw_max, label=f"{case_key} max-lengths")
    present_core = [family for family in CORE_FAMILIES if family in max_lengths]
    if not present_core:
        raise ValueError(
            f"{case_key} max-lengths must include at least one core family "
            f"from {list(CORE_FAMILIES)}"
        )
    extra_min = sorted(set(min_lengths) - set(max_lengths))
    if extra_min:
        raise ValueError(
            f"{case_key} min-lengths contain families missing from max-lengths: {extra_min}"
        )
    for family, min_count in min_lengths.items():
        max_count = max_lengths[family]
        if min_count > max_count:
            raise ValueError(
                f"{case_key} has min-length {min_count} > max-length {max_count} "
                f"for family {family!r}"
            )
    return dict(min_lengths), dict(max_lengths)


def _topology_max_lengths(topology: KernelTopology) -> dict[str, int]:
    lengths = {
        "psin": int(topology.psin_count),
        "h": int(topology.h_count),
        "k": int(topology.kappa_count),
        "v": int(topology.v_count),
        "F": int(topology.F_count),
    }
    lengths.update({f"c{idx}": int(value) for idx, value in enumerate(topology.c_counts)})
    lengths.update(
        {f"s{idx}": int(value) for idx, value in enumerate(topology.s_counts, start=1)}
    )
    return {name: count for name, count in lengths.items() if count > 0}


def _normalize_length_dict(lengths: Mapping[str, int], *, label: str) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for name, value in lengths.items():
        family = str(name)
        count = int(value)
        if count < 0:
            raise ValueError(f"{label} contains a negative length for {family!r}")
        if count == 0:
            continue
        normalized[family] = count
    return normalized


def _generate_case_signatures(
    case_key: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
    *,
    sweep_mode: str,
) -> list[SignatureRecord]:
    if sweep_mode == "full":
        if case_key == "solovev":
            return _generate_d_shape_ref_pruning_signatures(min_lengths, max_lengths)
        return _generate_general_ref_pruning_signatures(case_key, min_lengths, max_lengths)
    if sweep_mode == "partial":
        return _generate_partial_strategy_signatures(case_key, min_lengths, max_lengths)
    raise ValueError(f"Unsupported sweep mode {sweep_mode!r}; expected one of {SWEEP_MODES}")


def _generate_partial_strategy_signatures(
    case_key: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
) -> list[SignatureRecord]:
    records: list[SignatureRecord] = []
    seen: set[tuple[tuple[str, int], ...]] = set()
    for signature in TABLE05_SELECTED_SIGNATURES.get(case_key, ()):
        _add_partial_selected_signature(
            records,
            seen,
            case_key=case_key,
            min_lengths=min_lengths,
            max_lengths=max_lengths,
            signature=signature,
        )
    return records


def _add_partial_selected_signature(
    records: list[SignatureRecord],
    seen: set[tuple[tuple[str, int], ...]],
    *,
    case_key: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
    signature: dict[str, int],
) -> None:
    normalized = {name: int(length) for name, length in min_lengths.items() if int(length) > 0}
    for name, length in signature.items():
        if name not in max_lengths:
            return
        value = int(length)
        if value < int(min_lengths.get(name, 0)) or value > int(max_lengths[name]):
            return
        if value > 0:
            normalized[name] = value
    for name, floor in min_lengths.items():
        if int(normalized.get(name, 0)) < int(floor):
            return
    key = _signature_key(normalized)
    if key in seen:
        return
    seen.add(key)
    records.append(
        SignatureRecord(
            strategy_name="table05_selected",
            strategy_names=("table05_selected",),
            sweep_step=f"table05-selected_{case_key}_{len(records)}",
            signature=normalized,
        )
    )


def _generate_d_shape_ref_pruning_signatures(
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
) -> list[SignatureRecord]:
    required_core = ("psin", "h", "k")
    missing_core = [name for name in required_core if name not in max_lengths]
    if missing_core:
        raise ValueError(
            f"D-shape VEQ-ref pruning is missing required core families: {missing_core}"
        )

    records: list[SignatureRecord] = []
    seen: set[tuple[tuple[str, int], ...]] = set()

    _append_signature_record(
        records,
        seen,
        strategy_name="veq_ref_prune_full",
        sweep_step="d-shape-ref-prune-min",
        signature=dict(min_lengths),
    )

    core_ranges = [
        range(int(min_lengths.get(name, 1)), int(max_lengths[name]) + 1)
        for name in required_core
    ]
    sample_index = 1
    for psin_length, h_length, k_length in itertools.product(*core_ranges):
        core_signature = {
            "psin": int(psin_length),
            "h": int(h_length),
            "k": int(k_length),
        }
        for local_index in range(D_SHAPE_FULL_SINE_SAMPLES_PER_CORE):
            sine_signature = _d_shape_sine_ref_pruning_signature(
                min_lengths,
                max_lengths,
                sample_index=sample_index,
            )
            _append_signature_record(
                records,
                seen,
                strategy_name="veq_ref_prune_full",
                sweep_step=(
                    "d-shape-ref-prune-core-"
                    f"{psin_length}-{h_length}-{k_length}-s{local_index}"
                ),
                signature={**core_signature, **sine_signature},
            )
            sample_index += 1

    _append_signature_record(
        records,
        seen,
        strategy_name="veq_ref_prune_full",
        sweep_step="d-shape-ref-prune-ref",
        signature=dict(max_lengths),
    )
    _append_representative_neighborod_records(
        records,
        seen,
        case_key="solovev",
        min_lengths=min_lengths,
        max_lengths=max_lengths,
    )
    return records


def _d_shape_sine_ref_pruning_signature(
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
    *,
    sample_index: int,
) -> dict[str, int]:
    sine_families = [
        f"s{idx}" for idx in range(1, 1 + REFERENCE_LAYOUT_M_MAX) if f"s{idx}" in max_lengths
    ]
    if not sine_families:
        return {}

    active_depth = _stratified_count(
        sample_index,
        REF_PRUNE_HALTON_BASES[0],
        1,
        len(sine_families),
    )
    s1_name = sine_families[0]
    s1_length = _stratified_count(
        sample_index,
        REF_PRUNE_HALTON_BASES[1],
        int(min_lengths.get(s1_name, 1)),
        int(max_lengths[s1_name]),
    )
    signature = {s1_name: s1_length}
    if active_depth <= 1:
        return signature

    tail_values: list[int] = []
    tail_cap = max(
        1,
        min(s1_length, max(int(max_lengths[name]) for name in sine_families[1:active_depth])),
    )
    for offset, name in enumerate(sine_families[1:active_depth], start=2):
        upper = min(tail_cap, int(max_lengths[name]))
        tail_values.append(
            _stratified_count(sample_index, REF_PRUNE_HALTON_BASES[offset], 1, upper)
        )
    tail_values.sort(reverse=True)

    previous = s1_length
    for name, length in zip(sine_families[1:active_depth], tail_values, strict=True):
        previous = min(previous, int(length), int(max_lengths[name]))
        if previous <= 0:
            break
        signature[name] = previous
    return signature


def _generate_general_ref_pruning_signatures(
    case_key: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
) -> list[SignatureRecord]:
    records: list[SignatureRecord] = []
    seen: set[tuple[tuple[str, int], ...]] = set()

    _append_signature_record(
        records,
        seen,
        strategy_name="veq_ref_prune_full",
        sweep_step=f"{case_key}-ref-prune-min",
        signature=dict(min_lengths),
    )

    sample_index = 1
    for core_sample_index in range(1, GENERAL_REF_PRUNE_CORE_SAMPLE_COUNT + 1):
        core_signature = _ref_pruning_core_signature(
            min_lengths,
            max_lengths,
            sample_index=core_sample_index,
        )
        for local_index in range(GENERAL_REF_PRUNE_FOURIER_SAMPLES_PER_CORE):
            fourier_signature = _paired_fourier_ref_pruning_signature(
                min_lengths,
                max_lengths,
                sample_index=sample_index,
            )
            _append_signature_record(
                records,
                seen,
                strategy_name="veq_ref_prune_full",
                sweep_step=f"{case_key}-ref-prune-core-{core_sample_index}-f{local_index}",
                signature={**core_signature, **fourier_signature},
            )
            sample_index += 1

    _append_signature_record(
        records,
        seen,
        strategy_name="veq_ref_prune_full",
        sweep_step=f"{case_key}-ref-prune-ref",
        signature=dict(max_lengths),
    )
    _append_representative_neighborod_records(
        records,
        seen,
        case_key=case_key,
        min_lengths=min_lengths,
        max_lengths=max_lengths,
    )
    return records


def _ref_pruning_core_signature(
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
    *,
    sample_index: int,
) -> dict[str, int]:
    core_families = [family for family in CORE_FAMILIES if family in max_lengths]
    signature: dict[str, int] = {}
    for offset, family in enumerate(core_families):
        signature[family] = _stratified_count(
            sample_index,
            REF_PRUNE_HALTON_BASES[offset],
            int(min_lengths.get(family, 1)),
            int(max_lengths[family]),
        )
    return signature


def _paired_fourier_ref_pruning_signature(
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
    *,
    sample_index: int,
) -> dict[str, int]:
    shells = _fourier_shells(max_lengths)
    if not shells:
        return {}

    min_active_depth = max(1, _required_fourier_shell_count(min_lengths, max_lengths))
    active_depth = _stratified_count(
        sample_index,
        REF_PRUNE_HALTON_BASES[4],
        min_active_depth,
        len(shells),
    )

    signature: dict[str, int] = {}
    previous_cap = max(int(max_lengths[name]) for name in shells[0])
    for shell_idx, shell in enumerate(shells[:active_depth]):
        floor = max(1, max(int(min_lengths.get(name, 0)) for name in shell))
        cap = min(previous_cap, min(int(max_lengths[name]) for name in shell))
        if floor > cap:
            break
        length = _stratified_count(
            sample_index,
            REF_PRUNE_HALTON_BASES[5 + shell_idx],
            floor,
            cap,
        )
        for name in shell:
            count = min(int(length), int(max_lengths[name]))
            if count > 0:
                signature[name] = count
        previous_cap = int(length)
    return signature


def _append_representative_neighborod_records(
    records: list[SignatureRecord],
    seen: set[tuple[tuple[str, int], ...]],
    *,
    case_key: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
) -> None:
    anchors = [
        _normalize_representative_signature(case_key, min_lengths, max_lengths, signature)
        for signature in TABLE05_SELECTED_SIGNATURES.get(case_key, ())
    ]
    for anchor_idx, anchor in enumerate(anchors):
        _append_signature_record(
            records,
            seen,
            strategy_name="veq_ref_prune_full_selected",
            sweep_step=f"{case_key}-representative-{anchor_idx}",
            signature=anchor,
        )
        active_families = tuple(
            family
            for family in (*CORE_FAMILIES, *_available_fourier(max_lengths))
            if family in max_lengths
        )
        for family in active_families:
            current = int(anchor.get(family, min_lengths.get(family, 0)))
            if current <= 0:
                continue
            for value in _nearby_count_values(
                current=current,
                family=family,
                min_lengths=min_lengths,
                max_lengths=max_lengths,
            ):
                variant = dict(anchor)
                variant[family] = int(value)
                _append_signature_record(
                    records,
                    seen,
                    strategy_name="veq_ref_prune_full_selected_nearby",
                    sweep_step=f"{case_key}-representative-{anchor_idx}-{family}-{value}",
                    signature=variant,
                )

        core_families = tuple(
            family for family in CORE_FAMILIES if family in anchor and family in max_lengths
        )
        for delta in (-1, 1):
            variant = dict(anchor)
            changed = False
            for family in core_families:
                floor = int(min_lengths.get(family, 0))
                ceiling = int(max_lengths[family])
                value = min(max(int(anchor[family]) + delta, floor), ceiling)
                if value != int(anchor[family]):
                    variant[family] = value
                    changed = True
            if changed:
                _append_signature_record(
                    records,
                    seen,
                    strategy_name="veq_ref_prune_full_selected_nearby",
                    sweep_step=f"{case_key}-representative-{anchor_idx}-core{delta:+d}",
                    signature=variant,
                )

        for shell_idx, shell in enumerate(_fourier_shells(max_lengths)):
            if not any(family in anchor for family in shell):
                continue
            current = max(int(anchor.get(family, 0)) for family in shell)
            for value in _nearby_count_values(
                current=current,
                family=shell[0],
                min_lengths=min_lengths,
                max_lengths=max_lengths,
            ):
                variant = dict(anchor)
                for family in shell:
                    variant[family] = min(int(value), int(max_lengths[family]))
                _append_signature_record(
                    records,
                    seen,
                    strategy_name="veq_ref_prune_full_selected_nearby",
                    sweep_step=f"{case_key}-representative-{anchor_idx}-shell{shell_idx}-{value}",
                    signature=variant,
                )


def _normalize_representative_signature(
    case_key: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
    signature: dict[str, int],
) -> dict[str, int]:
    normalized = {family: int(floor) for family, floor in min_lengths.items() if int(floor) > 0}
    for family, raw_count in signature.items():
        if family not in max_lengths:
            raise ValueError(
                f"Representative signature for {case_key!r} uses unknown family {family!r}"
            )
        count = int(raw_count)
        floor = int(min_lengths.get(family, 0))
        ceiling = int(max_lengths[family])
        if count < floor or count > ceiling:
            raise ValueError(
                f"Representative signature for {case_key!r} "
                f"uses invalid {family}={count}; expected {floor}..{ceiling}"
            )
        if count > 0:
            normalized[family] = count
    for family, floor in min_lengths.items():
        if int(normalized.get(family, 0)) < int(floor):
            raise ValueError(
                f"Representative signature for {case_key!r} misses required family {family!r}"
            )
    return normalized


def _nearby_count_values(
    *,
    current: int,
    family: str,
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
) -> tuple[int, ...]:
    floor = max(1 if str(family).startswith(("c", "s")) else 0, int(min_lengths.get(family, 0)))
    ceiling = int(max_lengths[family])
    values = {
        min(max(int(current) - 1, floor), ceiling),
        min(max(int(current) + 1, floor), ceiling),
    }
    values.discard(int(current))
    return tuple(sorted(values))


def _append_signature_record(
    records: list[SignatureRecord],
    seen: set[tuple[tuple[str, int], ...]],
    *,
    strategy_name: str,
    sweep_step: str,
    signature: dict[str, int],
) -> None:
    normalized = _positive_signature(signature)
    key = _signature_key(normalized)
    if key in seen:
        return
    seen.add(key)
    records.append(
        SignatureRecord(
            strategy_name=strategy_name,
            strategy_names=(strategy_name,),
            sweep_step=sweep_step,
            signature=normalized,
        )
    )


def _positive_signature(signature: Mapping[str, int]) -> dict[str, int]:
    return {name: int(length) for name, length in signature.items() if int(length) > 0}


def _signature_key(signature: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((name, int(length)) for name, length in signature.items()))


def _fourier_shells(max_lengths: dict[str, int]) -> list[tuple[str, ...]]:
    shells: list[tuple[str, ...]] = []
    idx = 0
    while True:
        c_name = f"c{idx}"
        s_name = f"s{idx + 1}"
        shell = tuple(name for name in (c_name, s_name) if name in max_lengths)
        if not shell:
            break
        shells.append(shell)
        idx += 1
    return shells


def _available_fourier(max_lengths: dict[str, int]) -> list[str]:
    families: list[str] = []
    for shell in _fourier_shells(max_lengths):
        families.extend(shell)
    return families


def _required_fourier_shell_count(
    min_lengths: dict[str, int],
    max_lengths: dict[str, int],
) -> int:
    required_count = 0
    for idx, shell in enumerate(_fourier_shells(max_lengths), start=1):
        if any(int(min_lengths.get(name, 0)) > 0 for name in shell):
            required_count = idx
    return required_count


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
    if upper_int < lower_int:
        raise ValueError(f"invalid count range {lower_int}..{upper_int}")
    span = upper_int - lower_int + 1
    value = lower_int + int(np.floor(_radical_inverse(index, base) * span))
    return min(upper_int, max(lower_int, value))


def _candidate_queue_payload(
    records: Sequence[SignatureRecord],
    *,
    sweep_mode: str,
) -> dict[str, Any]:
    strategy_counts: dict[str, int] = {}
    for record in records:
        strategy_counts[record.strategy_name] = strategy_counts.get(record.strategy_name, 0) + 1
    return {
        "source": "benchmarks.numba_pareto.geqdsk_signature_queue",
        "signature_version": FULL_SWEEP_SIGNATURE_VERSION,
        "sweep_mode": sweep_mode,
        "record_count": int(len(records)),
        "strategy_counts": strategy_counts,
        "max_configs_per_case": FULL_SWEEP_MAX_CONFIGS_PER_CASE,
    }


def _run_case(
    args: argparse.Namespace,
    case_key: str,
    threshold_scales: tuple[float, ...],
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    route_spec = RouteBenchmarkSpec("PF", "psin", "uniform", "ip")
    kernel_case = geqdsk_kernel_case(
        case_key,
        "Ref",
        route_spec=route_spec,
        nr=args.nr,
        nt=args.nt,
        max_evaluations=args.max_evaluations,
    )
    thresholds = _thresholds_for_boundary(kernel_case.boundary.a, threshold_scales)
    row = _planned_row(case_key, kernel_case.topology, kernel_case.boundary, thresholds)
    signature_records = _script_candidate_records(
        case_key,
        kernel_case.topology,
        sweep_mode=args.sweep_mode,
    )
    row["candidate_queue"] = _candidate_queue_payload(
        signature_records,
        sweep_mode=args.sweep_mode,
    )
    kernel = None
    started = time.perf_counter_ns()
    try:
        kernel = Kernel(
            topology=kernel_case.topology,
            recipe=KernelRecipe(backend="numba", layout="degree"),
            config=kernel_case.config,
        )
        if progress_callback is not None:
            progress_callback("ref", 0, 1)
        reference = kernel.solve(
            kernel_case.boundary,
            kernel_case.source,
            config=kernel_case.config,
        )
        if progress_callback is not None:
            progress_callback("ref", 1, 1)
        result = kernel._impl.pareto(  # type: ignore[attr-defined]
            kernel_case.boundary,
            kernel_case.source,
            candidates=[record.signature for record in signature_records],
            config=kernel_case.config,
            reference=reference,
            target=args.target,
            metric=args.metric,
            _progress_callback=progress_callback,
        )
        elapsed_ms = float(time.perf_counter_ns() - started) / 1.0e6
        row["runtime"] = _runtime_payload(result, thresholds, elapsed_ms)
    except Exception as exc:
        elapsed_ms = float(time.perf_counter_ns() - started) / 1.0e6
        row["runtime"] = {
            "status": "failed",
            "failure_reason": "exception",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": elapsed_ms,
        }
    finally:
        if kernel is not None:
            kernel.close()
    return row


def _planned_row(
    case_key: str,
    topology: KernelTopology,
    boundary,
    thresholds: tuple[dict[str, float], ...],
) -> dict[str, Any]:
    return {
        "case": case_key,
        "config": "Ref",
        "route": "PF_psin_uniform_ip",
        "capacity": _topology_payload(topology),
        "boundary": _boundary_payload(boundary),
        "thresholds": list(thresholds),
        "runtime": {"status": "not_requested"},
    }


def _topology_payload(topology: KernelTopology) -> dict[str, Any]:
    return {
        "key": topology.key,
        "x_size": int(topology.x_size),
        "profile_counts": topology_profile_counts(topology),
        "grid": {
            "Nr": int(topology.Nr),
            "Nt": int(topology.Nt),
            "L_max": int(topology.L_max),
            "M_max": int(topology.M_max),
            "K_max": int(topology.K_max),
        },
        "sample_count": int(topology.sample_count),
    }


def _boundary_payload(boundary) -> dict[str, Any]:
    return {
        "source": "benchmarks._common.GEQDSK_BOUNDARY_PARAMETERS",
        "fit_backend": "numpy",
        "fit_method": boundary.fit_method,
        "fit_rms": None if boundary.fit_rms is None else float(boundary.fit_rms),
        "fit_max_curve_error": (
            None if boundary.fit_max_curve_error is None else float(boundary.fit_max_curve_error)
        ),
        "fit_c_order": boundary.fit_c_order,
        "fit_s_order": boundary.fit_s_order,
        "fit_note": (
            "Frozen GEQDSK LCFS least-square fit; no boundary fitting is performed "
            "inside this Pareto benchmark run."
        ),
    }


def _thresholds_for_boundary(
    minor_radius: float,
    scales: tuple[float, ...],
) -> tuple[dict[str, float], ...]:
    return tuple(
        {"scale": float(scale), "meters": float(scale) * float(minor_radius)} for scale in scales
    )


def _runtime_payload(
    result: ParetoResult,
    thresholds: tuple[dict[str, float], ...],
    elapsed_ms: float,
) -> dict[str, Any]:
    candidate_count = len(result.samples)
    valid_count = sum(1 for sample in result.samples if sample.result.success)
    reference_ms = float(result.reference.time)
    candidate_solve_ms = float(sum(sample.time for sample in result.samples))
    solver_elapsed_ms = reference_ms + candidate_solve_ms
    threshold_selection = select_pareto_thresholds(
        result.frontier,
        tuple(threshold["meters"] for threshold in thresholds),
        target=result.target,
    )
    selected = {
        f"{threshold['scale']:.16g}": _sample_payload(threshold_selection[threshold["meters"]])
        for threshold in thresholds
        if threshold["meters"] in threshold_selection
    }
    return {
        "status": "passed" if result.reference.result.success else "failed",
        "elapsed_ms": float(elapsed_ms),
        "reference_solve_ms": reference_ms,
        "candidate_solve_ms": candidate_solve_ms,
        "solver_elapsed_ms": solver_elapsed_ms,
        "overhead_ms": float(elapsed_ms) - solver_elapsed_ms,
        "candidate_count": int(candidate_count),
        "valid_candidate_count": int(valid_count),
        "frontier_count": int(len(result.frontier)),
        "evaluations_per_second": _evaluations_per_second(candidate_count, elapsed_ms),
        "reference": _sample_payload(result.reference),
        "samples": [_sample_payload(sample) for sample in result.samples],
        "frontier": [_sample_payload(sample) for sample in result.frontier],
        "selected": selected,
    }


def _evaluations_per_second(candidate_count: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0.0:
        return float("nan")
    return 1000.0 * float(candidate_count) / float(elapsed_ms)


def _sample_payload(sample: ParetoSample) -> dict[str, Any]:
    return {
        "signature": sample.signature.to_variant_kwargs(),
        "counts": int(sample.counts),
        "time_ms": float(sample.time),
        "complexity": int(sample.complexity),
        "shape_error": float(sample.shape_error),
        "success": bool(sample.result.success),
        "nfev": int(sample.result.nfev),
        "raw_norm": float(sample.result.raw_norm),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "total": len(rows),
        "runtime_passed": 0,
        "runtime_failed": 0,
        "runtime_not_requested": 0,
    }
    for row in rows:
        status = row.get("runtime", {}).get("status")
        if status == "passed":
            counts["runtime_passed"] += 1
        elif status == "failed":
            counts["runtime_failed"] += 1
        elif status == "not_requested":
            counts["runtime_not_requested"] += 1
    return counts


def _progress_callback(progress, task_id: int, case_key: str) -> Callable[[str, int, int], None]:
    phase_labels = {
        "ref": "[cyan]ref[/]",
        "run": "[cyan]run[/]",
    }

    def update(phase: str, completed: int, total: int) -> None:
        progress.update(
            task_id,
            total=max(int(total), 1),
            completed=max(int(completed), 0),
            current=case_key,
            phase=phase_labels.get(phase, "[cyan]run[/]"),
        )

    return update


def _finished_progress_counts(row: dict[str, Any]) -> tuple[int, int]:
    runtime = row.get("runtime", {})
    candidate_count = runtime.get("candidate_count")
    if candidate_count is None:
        return 1, 1
    try:
        count = int(candidate_count)
    except (TypeError, ValueError):
        return 1, 1
    total = max(count, 1)
    return total, total


def _print_summary(console, rows: list[dict[str, Any]]) -> None:
    summary = Table(box=REPORT_TABLE_BOX, show_lines=False, expand=False, padding=(0, 1))
    summary.add_column("case", no_wrap=True)
    summary.add_column("status", no_wrap=True)
    summary.add_column("ref x", justify="right")
    summary.add_column("evals", justify="right")
    summary.add_column("frontier", justify="right")
    summary.add_column(Text("elapsed (ms)"), justify="right")
    summary.add_column(Text("eval/s"), justify="right")

    for row in rows:
        runtime = row.get("runtime", {})
        reference = runtime.get("reference", {})
        summary.add_row(
            str(row.get("case", "n/a")),
            status_cell(runtime.get("status", "n/a")),
            str(reference.get("counts", row.get("capacity", {}).get("x_size", "n/a"))),
            str(runtime.get("candidate_count", "n/a")),
            str(runtime.get("frontier_count", "n/a")),
            format_optional_float(runtime.get("elapsed_ms"), precision=1),
            format_optional_float(runtime.get("evaluations_per_second"), precision=2),
        )
    console.print(summary)
    console.print()

    selected_table = Table(box=REPORT_TABLE_BOX, show_lines=False, expand=False, padding=(0, 1))
    selected_table.add_column("case", no_wrap=True)
    selected_table.add_column("tol", justify="right")
    selected_table.add_column(Text("tol (m)"), justify="right")
    selected_table.add_column("x", justify="right")
    selected_table.add_column("complexity", justify="right")
    selected_table.add_column(Text("time (ms)"), justify="right")
    selected_table.add_column(Text("R error (m)"), justify="right")
    for row in rows:
        selected = row.get("runtime", {}).get("selected", {})
        thresholds = row.get("thresholds", [])
        for threshold in thresholds:
            scale = float(threshold["scale"])
            sample = selected.get(f"{scale:.16g}")
            selected_table.add_row(
                str(row.get("case", "n/a")),
                f"{scale:g} a",
                format_optional_sci(threshold["meters"]),
                str(sample["counts"]) if sample else "n/a",
                str(sample["complexity"]) if sample else "n/a",
                format_optional_float(sample["time_ms"], precision=3) if sample else "n/a",
                format_optional_sci(sample["shape_error"]) if sample else "n/a",
            )
    console.print(selected_table)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASE_KEYS)
    parser.add_argument("--sweep-mode", choices=SWEEP_MODES, default=DEFAULT_SWEEP_MODE)
    parser.add_argument("--metric", choices=("rms", "max"), default="rms")
    parser.add_argument("--target", choices=("counts", "time", "complexity"), default="counts")
    parser.add_argument("--max-evaluations", type=int, default=DEFAULT_MAX_EVALUATIONS)
    parser.add_argument("--nr", type=int, default=REFERENCE_LAYOUT_NR)
    parser.add_argument("--nt", type=int, default=REFERENCE_LAYOUT_NT)
    parser.add_argument(
        "--threshold-scale",
        action="append",
        type=float,
        default=None,
        help="Shape-error tolerance as a fraction of boundary minor radius a; may be repeated.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    console = reporting_console()
    case_keys = selected_cases(args.case)
    threshold_scales = tuple(
        float(scale) for scale in (args.threshold_scale or DEFAULT_THRESHOLD_SCALES)
    )

    if not args.quiet_progress:
        print_config_tree(
            console,
            (
                f"cases: [green]{', '.join(case_keys)}[/]",
                "backend: [green]numba[/]",
                "capacity: [green]GEQDSK Ref topology[/]",
                "route: [green]PF/psin/uniform/ip[/]",
                f"grid: [green]{args.nr} x {args.nt}[/]",
                f"sweep mode: [green]{args.sweep_mode}[/]",
                f"metric: [green]{args.metric}[/]",
                f"target: [green]{args.target}[/]",
                f"thresholds: [green]{', '.join(f'{scale:g}*a' for scale in threshold_scales)}[/]",
            ),
        )
        console.print()
        console.print(Text("[progress]", style="bold cyan"))

    rows: list[dict[str, Any]] = []
    with progress_context(console, quiet=args.quiet_progress) as progress:
        task_by_case: dict[str, int] = {}
        if progress is not None:
            for case_key in case_keys:
                task_by_case[case_key] = progress.add_task(
                    "",
                    total=1,
                    current=case_key,
                    phase="[dim]pending[/]",
                )
        for case_key in case_keys:
            callback = None
            if progress is not None:
                progress.update(
                    task_by_case[case_key],
                    current=case_key,
                    phase="[cyan]ref[/]",
                )
                callback = _progress_callback(progress, task_by_case[case_key], case_key)
            row = _run_case(args, case_key, threshold_scales, progress_callback=callback)
            rows.append(row)
            if progress is not None:
                completed, total = _finished_progress_counts(row)
                progress.update(
                    task_by_case[case_key],
                    total=total,
                    current=case_key,
                    phase=progress_phase(row.get("runtime", {}).get("status")),
                    completed=completed,
                )

    payload = {
        "schema": "veqpy.numba.pareto_geqdsk.v1",
        "cpu_affinity": cpu_affinity(),
        "env": runtime_env(),
        "run_note": (
            "GEQDSK Ref-capacity Pareto screening benchmark. The benchmark "
            "builds a deterministic full/partial GEQDSK signature queue, then "
            "passes the reduced topology candidates to the Numba Kernel.pareto evaluator. "
            "threshold_scale values are multiplied by each boundary minor radius a. "
            "GEQDSK boundaries are pre-fitted frozen parameterized KernelBoundary "
            "inputs from benchmarks._common; this benchmark does not refit LCFS points."
        ),
        "args": {
            "cases": list(case_keys),
            "sweep_mode": args.sweep_mode,
            "metric": args.metric,
            "target": args.target,
            "max_evaluations": int(args.max_evaluations),
            "nr": int(args.nr),
            "nt": int(args.nt),
            "threshold_scale": list(threshold_scales),
        },
        "summary": _summary(rows),
        "rows": rows,
    }
    if not args.no_write:
        write_json(args.output, payload)
        if not args.quiet_progress:
            console.print()
            print_outputs_tree(console, {"json": args.output}, repo_root=REPO_ROOT)
            console.print()
    _print_summary(console, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
