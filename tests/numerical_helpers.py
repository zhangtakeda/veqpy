from __future__ import annotations

from argparse import Namespace
from typing import Any

import numpy as np
import pytest


def benchmark_args(**overrides: object) -> Namespace:
    defaults: dict[str, object] = {
        "build": "fastmath",
        "layout": "degree",
        "cmake_build_type": None,
        "fp_mode": None,
        "enzyme_jacobian_batch_width": None,
        "method": "powell",
        "initial": "cold",
        "norm": "fast",
        "max_evaluations": 1000,
        "repeat": 1,
        "warmup": 0,
        "no_run": False,
        "boundary_fit_m": 10,
        "boundary_fit_n": 10,
        "boundary_maxtol": 1.0,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def assert_finite(value: object, *, name: str) -> float:
    numeric = float(value)
    assert np.isfinite(numeric), f"{name} is not finite: {numeric!r}"
    return numeric


def assert_status_passed(row: dict[str, Any], *, status_key: str = "status") -> None:
    status = row.get(status_key)
    if status != "passed":
        pytest.fail(f"expected passed row, got {status!r}: {row}")


def assert_runtime_passed(row: dict[str, Any]) -> None:
    runtime = row.get("runtime")
    if not isinstance(runtime, dict):
        pytest.fail(f"missing runtime payload: {row}")
    if runtime.get("status") != "passed":
        pytest.fail(f"expected passed runtime, got {runtime.get('status')!r}: {row}")


def skip_if_native_unavailable(row: dict[str, Any]) -> None:
    text = " ".join(str(row.get(key, "")) for key in ("error", "failure_reason"))
    if any(
        token in text.lower()
        for token in (
            "cmake",
            "compiler",
            "nanobind",
            "build",
            "no such file",
            "not found",
        )
    ):
        pytest.skip(f"native backend unavailable: {text}")

