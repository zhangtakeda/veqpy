from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/veqpy-mpl")))

import numpy as np

from veqlib.facade import KernelResult, pinned_cpu

THIS_FILE = Path(__file__).resolve()
VEQLIB_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]
CORE_DIR = VEQLIB_ROOT / "core"
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def temp_cache(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cpu_affinity() -> list[int] | None:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    return None


def runtime_env() -> dict[str, str | None]:
    keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    return {key: os.environ.get(key) for key in keys}


def quantile(values: list[float], q: float) -> float:
    values_sorted = sorted(values)
    return float(values_sorted[int((len(values_sorted) - 1) * q)])


def float_stats(values: list[float], *, samples: bool = True, prefix: str = "") -> dict[str, Any]:
    if not values:
        return {
            f"{prefix}median_ms": float("nan"),
            f"{prefix}mean_ms": float("nan"),
            f"{prefix}min_ms": float("nan"),
            f"{prefix}max_ms": float("nan"),
            f"{prefix}p05_ms": float("nan"),
            f"{prefix}p95_ms": float("nan"),
            "count": 0,
            **({"samples_ms": []} if samples else {}),
        }
    payload: dict[str, Any] = {
        f"{prefix}median_ms": float(statistics.median(values)),
        f"{prefix}mean_ms": float(statistics.mean(values)),
        f"{prefix}min_ms": float(min(values)),
        f"{prefix}max_ms": float(max(values)),
        f"{prefix}p05_ms": quantile(values, 0.05),
        f"{prefix}p95_ms": quantile(values, 0.95),
        "count": len(values),
    }
    if samples:
        payload["samples_ms"] = [float(value) for value in values]
    return payload


def int_stats(values: list[int], *, samples: bool = True) -> dict[str, Any]:
    if not values:
        return {
            "median": 0,
            "mean": 0.0,
            "min": 0,
            "max": 0,
            **({"samples": []} if samples else {}),
        }
    payload: dict[str, Any] = {
        "median": int(statistics.median(values)),
        "mean": float(statistics.mean(values)),
        "min": int(min(values)),
        "max": int(max(values)),
    }
    if samples:
        payload["samples"] = [int(value) for value in values]
    return payload


def max_abs(lhs: Any, rhs: Any) -> float:
    lhs_arr = np.asarray(lhs, dtype=np.float64)
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    if lhs_arr.shape != rhs_arr.shape:
        return float("inf")
    if lhs_arr.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs_arr - rhs_arr)))


def profile_count(profile_coeffs: dict[str, Any], name: str) -> int:
    values = profile_coeffs.get(name)
    return 0 if values is None else int(np.asarray(values, dtype=np.float64).size)


def family_counts(profile_coeffs: dict[str, Any], prefix: str, first: int) -> tuple[int, ...]:
    orders = [
        int(name[1:])
        for name, values in profile_coeffs.items()
        if values is not None
        and len(name) > 1
        and name[0] == prefix
        and name[1:].isdigit()
        and profile_count(profile_coeffs, name) > 0
    ]
    if not orders:
        return ()
    counts = [
        profile_count(profile_coeffs, f"{prefix}{order}")
        for order in range(first, max(orders) + 1)
    ]
    while counts and counts[-1] == 0:
        counts.pop()
    return tuple(counts)


def finite_or_nan(value: Any) -> float:
    parsed = float(value)
    return parsed if np.isfinite(parsed) else float("nan")


@dataclass(frozen=True, slots=True)
class NativeTiming:
    result: KernelResult
    wall_ms: list[float]
    inner_ms: list[float]
    success: list[bool]
    nfev: list[int]
    njev: list[int]
    jacobian_component_evaluations: list[int]

    def compact(self) -> dict[str, Any]:
        return {
            "success_all": all(self.success),
            "info": int(self.result.info),
            "timing": float_stats(self.wall_ms),
            "inner_timing": float_stats(self.inner_ms),
            "nfev": int_stats(self.nfev),
            "njev": int_stats(self.njev),
            "jacobian_component_evaluations": int_stats(self.jacobian_component_evaluations),
            "raw_norm": float(self.result.raw_norm),
            "scaled_norm": float(self.result.scaled_norm),
            "x": self.result.x.tolist(),
            "raw": self.result.raw.tolist(),
            "alpha": self.result.alpha.tolist(),
        }


def measure_native_solver(
    solver: Any,
    configure_runtime: Callable[[], None],
    *,
    warmup: int,
    repeat: int,
) -> NativeTiming:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    with pinned_cpu():
        configure_runtime()
        for _ in range(warmup):
            solver.solve_direct()

        wall_ms: list[float] = []
        inner_ms: list[float] = []
        success: list[bool] = []
        nfev: list[int] = []
        njev: list[int] = []
        jaccomp: list[int] = []
        final_result: Any | None = None
        for _ in range(repeat):
            start_ns = time.perf_counter_ns()
            final_result = solver.solve_direct()
            wall_ms.append(float(time.perf_counter_ns() - start_ns) / 1.0e6)
            inner_ms.append(float(final_result[0]))
            success.append(bool(final_result[1]))
            nfev.append(int(final_result[3]))
            njev.append(int(final_result[4]))
            jaccomp.append(int(final_result[6]))

    if final_result is None:
        raise RuntimeError("native timing loop did not run")
    return NativeTiming(
        result=KernelResult.from_solve_direct(final_result),
        wall_ms=wall_ms,
        inner_ms=inner_ms,
        success=success,
        nfev=nfev,
        njev=njev,
        jacobian_component_evaluations=jaccomp,
    )
