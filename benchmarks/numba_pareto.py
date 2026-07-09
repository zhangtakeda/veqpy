#!/usr/bin/env python3
"""Numba Kernel.pareto() smoke run on a tiny synthetic topology."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rich.table import Table
from rich.text import Text

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks._common import REPO_ROOT, cpu_affinity, runtime_env, write_json
from benchmarks._reporting import (
    REPORT_TABLE_BOX,
    format_optional_float,
    format_optional_sci,
    print_config_tree,
    print_outputs_tree,
    status_cell,
)
from benchmarks._reporting import (
    console as reporting_console,
)
from veqpy import Kernel, KernelBoundary, KernelConfig, KernelRecipe, KernelSource, KernelTopology
from veqpy.kernels.pareto import ParetoResult, ParetoSample

DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "numba_pareto.json"
MU0 = 4.0e-7 * np.pi


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    console = reporting_console()
    if not args.quiet_progress:
        print_config_tree(
            console,
            (
                "purpose: [green]Kernel.pareto smoke/provenance check[/]",
                "backend: [green]numba[/]",
                "case: [green]tiny synthetic PF/psin/uniform[/]",
                f"strategy: [green]{args.strategy}[/]",
                f"metric: [green]{args.metric}[/]",
                f"pareto_by: [green]{args.pareto_by}[/]",
                f"max candidates: [green]{args.max_candidates}[/]",
            ),
        )
        console.print()
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
    if not args.no_write:
        write_json(args.output, payload)
        if not args.quiet_progress:
            print_outputs_tree(console, {"json": args.output}, repo_root=REPO_ROOT)
            console.print()
    _print_summary(console, payload)
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
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
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
        "cpu_affinity": cpu_affinity(),
        "env": runtime_env(),
        "run_note": (
            "Tiny synthetic smoke run for Kernel.pareto() contract and JSON shape. "
            "This is not a GEQDSK performance matrix or a stable timing benchmark."
        ),
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
        "complexity": int(sample.complexity),
        "shape_error": float(sample.shape_error),
        "success": bool(sample.result.success),
        "nfev": int(sample.result.nfev),
        "raw_norm": float(sample.result.raw_norm),
    }


def _print_summary(console, payload: dict[str, Any]) -> None:
    args = payload["args"]
    console.print(
        Text(
            "Numba Pareto smoke: "
            f"strategy={args['strategy']} metric={args['metric']} pareto_by={args['pareto_by']}",
            style="bold cyan",
        )
    )
    table = Table(box=REPORT_TABLE_BOX, show_lines=False, expand=False, padding=(0, 1))
    table.add_column("role", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("counts", justify="right")
    table.add_column(Text("time (ms)"), justify="right")
    table.add_column("complexity", justify="right")
    table.add_column(Text("R error (m)"), justify="right")
    table.add_column("nfev", justify="right")

    reference = payload["reference"]
    for sample in payload["frontier"]:
        table.add_row(
            "ref" if sample["signature"] == reference["signature"] else "candidate",
            status_cell("passed" if sample["success"] else "failed"),
            str(sample["counts"]),
            format_optional_float(sample["time"], precision=3),
            str(sample["complexity"]),
            format_optional_sci(sample["shape_error"]),
            str(sample["nfev"]),
        )
    console.print(table)


if __name__ == "__main__":
    raise SystemExit(main())
