from __future__ import annotations

from argparse import Namespace
from typing import Any

import numpy as np
import pytest

MU0 = 4.0e-7 * np.pi


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
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def pf_reference_profiles(psin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75
    alpha_p, alpha_f = 5.0, 3.32
    exp_ap, exp_af = np.exp(alpha_p), np.exp(alpha_f)
    den_p = 1.0 + exp_ap * (alpha_p - 1.0)
    den_f = 1.0 + exp_af * (alpha_f - 1.0)
    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    return current_input.astype(np.float64), heat_input.astype(np.float64)


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
