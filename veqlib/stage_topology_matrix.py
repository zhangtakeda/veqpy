#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_TOPOLOGIES = ("32x16x1", "32x32x1", "64x16x1")
REPRESENTATIVE_TOPOLOGIES = (
    "16x16x1",
    "32x16x1",
    "64x16x1",
    "32x8x1",
    "32x24x1",
    "32x32x1",
    "32x64x1",
    "32x16x4",
    "32x16x8",
)
FULL_MATRIX_TOPOLOGIES = tuple(
    f"{nr}x{nt}x{mmax}"
    for nr in (16, 32, 64)
    for nt in (8, 16, 24, 32, 64)
    for mmax in (1, 4, 8)
)
MATRIX_PRESETS = {
    "default": DEFAULT_TOPOLOGIES,
    "representative": REPRESENTATIVE_TOPOLOGIES,
    "full": FULL_MATRIX_TOPOLOGIES,
}


def parse_topology(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace("m", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("topology must be Nr x Nt x Mmax, for example 32x16x1")
    try:
        nr, nt, mmax = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("topology components must be integers") from exc
    if nr < 4 or nt < 4 or mmax < 1:
        raise argparse.ArgumentTypeError("topology requires Nr>=4, Nt>=4, Mmax>=1")
    return nr, nt, mmax


def topology_name(topology: tuple[int, int, int]) -> str:
    nr, nt, mmax = topology
    return f"nr{nr}_nt{nt}_m{mmax}"


def cmake_cache_args(topology: tuple[int, int, int]) -> list[str]:
    nr, nt, mmax = topology
    s_counts = ";".join("3" for _ in range(mmax))
    kmax_limit = max(2, mmax)
    return [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_COMPILER=clang++",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DENABLE_ENZYME=OFF",
        "-DVEQLIB_ENABLE_NATIVE_OPTIMIZATIONS=ON",
        "-DVEQLIB_FP_MODE=RELAXED",
        "-DVEQLIB_ENABLE_THIN_LTO=ON",
        f"-DVEQ_NR={nr}",
        f"-DVEQ_NT={nt}",
        "-DVEQ_H_PROFILE_COUNT=3",
        "-DVEQ_V_PROFILE_COUNT=0",
        "-DVEQ_KAPPA_PROFILE_COUNT=6",
        "-DVEQ_PSIN_PROFILE_COUNT=6",
        "-DVEQ_F_PROFILE_COUNT=0",
        "-DVEQ_COS_PROFILE_COUNTS=0",
        f"-DVEQ_SIN_PROFILE_COUNTS={s_counts}",
        f"-DVEQ_PROFILE_KMAX_LIMIT={kmax_limit}",
    ]


def run_json(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE)
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run VEQlib stage benchmark across CMake topology builds."
    )
    parser.add_argument(
        "--topology",
        action="append",
        type=parse_topology,
        help="Nr x Nt x Mmax, e.g. 32x16x1",
    )
    parser.add_argument(
        "--matrix-preset",
        choices=tuple(MATRIX_PRESETS),
        default="default",
        help="topology preset used when --topology is omitted",
    )
    parser.add_argument("--stage", default="all", help="stage_benchmark --stage value")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--inner", type=int, default=5000)
    parser.add_argument("--ring-size", type=int, default=16)
    parser.add_argument("--build-root", type=Path, default=Path("build/topology-matrix"))
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()

    source_dir = Path(__file__).resolve().parent
    topologies = args.topology or [
        parse_topology(item) for item in MATRIX_PRESETS[args.matrix_preset]
    ]
    results: list[dict[str, Any]] = []

    for topology in topologies:
        build_dir = source_dir / args.build_root / topology_name(topology)
        build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["cmake", "-S", str(source_dir), "-B", str(build_dir), *cmake_cache_args(topology)],
            check=True,
            stdout=sys.stderr,
        )
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "veqlib_stage_benchmark"],
            check=True,
            stdout=sys.stderr,
        )
        exe = build_dir / "veqlib_stage_benchmark"
        report = run_json(
            [
                str(exe),
                "--stage",
                args.stage,
                "--repeat",
                str(args.repeat),
                "--warmup",
                str(args.warmup),
                "--inner",
                str(args.inner),
                "--ring-size",
                str(args.ring_size),
            ],
            source_dir,
        )
        report["matrix_topology"] = {"Nr": topology[0], "Nt": topology[1], "Mmax": topology[2]}
        results.append(report)

    matrix_report = {"schema_version": 1, "topologies": results}
    text = json.dumps(matrix_report, indent=2)
    if args.output:
        output = args.output if args.output.is_absolute() else source_dir / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
