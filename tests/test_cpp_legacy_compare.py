from __future__ import annotations

from pathlib import Path

import pytest

from veqpy.cpp import LegacyCompareConfig, benchmark_legacy_veqpy_comparison


def test_legacy_veqpy_comparison_runs_against_debug_nanobind_module() -> None:
    module_dir = Path("veqlib/build/debug")
    if not sorted(module_dir.glob("veqlib_ext*.so")):
        pytest.skip("veqlib debug nanobind extension has not been built")

    report = benchmark_legacy_veqpy_comparison(
        config=LegacyCompareConfig(repeat=1, warmup=0, module_dir=module_dir),
    )

    assert report["schema_version"] == 1
    assert report["summary"]["case_count"] == 1
    assert report["rows"][0]["python"]["success"] is True
    assert report["rows"][0]["cxx"]["success"] is True
    assert report["summary"]["max_final_raw_abs_diff"] < 1.0e-9
