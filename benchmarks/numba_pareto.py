#!/usr/bin/env python3
"""Smoke benchmark for Numba Kernel.pareto() topology reduction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veqpy import Kernel, KernelBoundary, KernelConfig, KernelRecipe, KernelSource, KernelTopology
from veqpy.kernels.pareto import ParetoResult, ParetoSample

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "numba_pareto.json"
MU0 = 4.0e-7 * np.pi


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    kernel = Kernel(
        topology=_topology(),
        recipe=KernelRecipe(backend="numba", layout="degree"),
        config=_config(),
    )
    try:
        result = kernel.pareto(
            _boundary(),
            _source(),
            config=_config(),
            max_shape_error=tuple(args.max_shape_error),
            pareto_by=args.pareto_by,
            strategy=args.strategy,
            metric=args.metric,
            max_candidates=args.max_candidates,
        )
    finally:
        kernel.close()

    payload = _payload(result, args)
    _print_summary(payload)
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=("tail", "energy", "adaptive", "balanced"),
        default="adaptive",
    )
    parser.add_argument("--metric", choices=("rms", "max"), default="rms")
    parser.add_argument("--pareto-by", choices=("counts", "time", "complexity"), default="counts")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument(
        "--max-shape-error",
        action="append",
        type=float,
        default=[1.0e-3, 5.0e-4],
        help="Major-radius error threshold in meters; may be repeated.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args(argv)


def _topology() -> KernelTopology:
    return KernelTopology(
        h_count=4,
        v_count=3,
        kappa_count=2,
        psin_count=3,
        F_count=0,
        c_counts=(2, 1),
        s_counts=(2,),
        Nr=8,
        Nt=8,
        route="PF",
        coordinate="psin",
        nodes="uniform",
        ip_constraint=True,
        sample_count=8,
    )


def _boundary() -> KernelBoundary:
    return KernelBoundary(
        a=0.5,
        R0=1.0,
        Z0=0.0,
        B0=3.0,
        ka=1.7,
        c_offsets=(0.02, 0.01),
        s_offsets=(float(np.arcsin(0.2)),),
    )


def _source() -> KernelSource:
    psin = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    current = 1.0 + 0.1 * psin
    heat = 1.0e6 + 0.2e6 * psin
    return KernelSource(
        heat_profile=heat / MU0,
        current_profile=current,
        Ip=3.0e6,
    )


def _config() -> KernelConfig:
    return KernelConfig(
        method="powell",
        initial="cold-zeros",
        norm="none",
        max_residual=1.0e12,
        max_evaluations=1,
    )


def _payload(result: ParetoResult, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "veqpy.numba.pareto_smoke.v1",
        "args": {
            "strategy": args.strategy,
            "metric": args.metric,
            "pareto_by": args.pareto_by,
            "max_candidates": int(args.max_candidates),
            "max_shape_error": [float(value) for value in args.max_shape_error],
        },
        "reference": _sample_payload(result.reference),
        "samples": [_sample_payload(sample) for sample in result.samples],
        "frontier": [_sample_payload(sample) for sample in result.frontier],
        "selected": {
            f"{threshold:.16g}": _sample_payload(sample)
            for threshold, sample in result.selected.items()
        },
    }


def _sample_payload(sample: ParetoSample) -> dict[str, Any]:
    return {
        "signature": sample.signature.to_variant_kwargs(),
        "counts": int(sample.counts),
        "time": float(sample.time),
        "complexity": float(sample.complexity),
        "shape_error": float(sample.shape_error),
        "success": bool(sample.result.success),
        "nfev": int(sample.result.nfev),
        "raw_norm": float(sample.result.raw_norm),
    }


def _print_summary(payload: dict[str, Any]) -> None:
    args = payload["args"]
    print(
        "Numba Pareto smoke: "
        f"strategy={args['strategy']} metric={args['metric']} pareto_by={args['pareto_by']}"
    )
    print("counts time_ms complexity shape_error success")
    for sample in payload["frontier"]:
        print(
            f"{sample['counts']:>6} "
            f"{sample['time']:>7.3f} "
            f"{sample['complexity']:>10.1f} "
            f"{sample['shape_error']:>11.3e} "
            f"{sample['success']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
