"""Compare official GEQDSK PF/psin and native PF/rho solves."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks._common import (
    CASE_KEYS,
    CONFIG_LABELS,
    REFERENCE_SOLVER_MAXFEV,
    RouteBenchmarkSpec,
    geqdsk_kernel_case,
    measure_kernel_case,
    runtime_env,
    runtime_platform_payload,
    write_json,
)
from veqpy import KernelRecipe


def _measure(
    case_key: str,
    config_label: str,
    coordinate: str,
    *,
    warmup: int,
    repeat: int,
    method: str,
    max_evaluations: int,
) -> tuple[dict[str, object], object]:
    case = geqdsk_kernel_case(
        case_key,
        config_label,
        route_spec=RouteBenchmarkSpec("PF", coordinate, "uniform", "ip"),
        method=method,
        max_residual=1.0e-6,
        max_evaluations=max_evaluations,
        initial="cold",
        norm="fast",
    )
    measure = measure_kernel_case(
        case,
        recipe=KernelRecipe(backend="numba", layout="degree"),
        warmup=warmup,
        repeat=repeat,
    )
    result = measure["result"]
    return (
        {
            "success": bool(measure["success"]),
            "x_size": int(case.topology.x_size),
            "median_ms": float(measure["median_ms"]),
            "nfev_median": float(measure["nfev_median"]),
            "raw_norm": float(result.raw_norm),
            "timings_ms": [float(value) for value in measure["timings_ms"]],
        },
        measure["kernel"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=CASE_KEYS)
    parser.add_argument("--config", action="append", choices=CONFIG_LABELS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--method", default="powell")
    parser.add_argument("--max-evaluations", type=int, default=REFERENCE_SOLVER_MAXFEV)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for case_key in args.case or CASE_KEYS:
        for config_label in args.config or CONFIG_LABELS:
            kernels = []
            row: dict[str, object] = {"case": case_key, "config": config_label}
            try:
                for coordinate in ("psin", "rho"):
                    result, kernel = _measure(
                        case_key,
                        config_label,
                        coordinate,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        method=args.method,
                        max_evaluations=args.max_evaluations,
                    )
                    kernels.append(kernel)
                    row[coordinate] = result
                psin = row["psin"]
                rho = row["rho"]
                qualified = bool(psin["success"] and rho["success"])
                row["qualified"] = qualified
                row["solve_time_ratio"] = (
                    float(rho["median_ms"]) / float(psin["median_ms"]) if qualified else None
                )
            except Exception as exc:
                row.update(
                    qualified=False,
                    solve_time_ratio=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                for kernel in kernels:
                    kernel.close()
            rows.append(row)
            ratio = row.get("solve_time_ratio")
            ratio_text = "n/a" if ratio is None else f"{float(ratio):.2f}x"
            print(
                f"{case_key:>7}/{config_label:<6} ratio={ratio_text} "
                f"qualified={row.get('qualified')}"
            )

    payload = {
        "settings": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "method": args.method,
            "max_evaluations": args.max_evaluations,
            "initial": "cold",
            "norm": "fast",
        },
        "platform": runtime_platform_payload(),
        "environment": runtime_env(),
        "rows": rows,
    }
    if args.output is not None:
        write_json(args.output, payload)


if __name__ == "__main__":
    main()
