from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from veqpy.cpp import KernelRegistry, LifecycleBenchmarkConfig, benchmark_kernel_lifecycle
from veqpy.topology import Topology


def make_topology() -> Topology:
    return Topology(
        h_count=3,
        v_count=0,
        kappa_count=6,
        psin_count=6,
        F_count=0,
        c_counts=(),
        s_counts=(3,),
        Nr=32,
        Nt=16,
        route="PF",
        coordinate="psin",
        constraint="Ip",
        nodes="uniform",
        sample_count=51,
    )


def prepare_fastmath_artifact(registry: KernelRegistry, topology: Topology) -> None:
    candidates = sorted(Path("veqlib/build/release").glob("veqlib_ext*.so"))
    if not candidates:
        pytest.skip("veqlib release fastmath nanobind extension has not been built")
    artifact = registry.get_or_build(topology, dry_run=True)
    shutil.copy2(candidates[0], artifact.shared_library_path)
    metadata = json.loads(artifact.metadata_path.read_text())
    metadata["artifact"]["status"] = "built"
    metadata["artifact"]["shared_library_sha256"] = hashlib.sha256(
        artifact.shared_library_path.read_bytes()
    ).hexdigest()
    artifact.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def test_kernel_lifecycle_benchmark_reports_cache_and_thread_metrics(tmp_path: Path) -> None:
    topology = make_topology()
    registry = KernelRegistry(cache_root=tmp_path)
    prepare_fastmath_artifact(registry, topology)

    report = benchmark_kernel_lifecycle(
        topology,
        registry=registry,
        config=LifecycleBenchmarkConfig(repeat=2, warmup=0, threads=2),
    )

    assert report["schema"] == "veqpy.cpp.lifecycle_benchmark.v1"
    assert report["topology"]["build"] == "fastmath"
    assert report["result"]["first_success"] is True
    assert report["result"]["last_success"] is True
    assert report["metrics"]["warm_registry_hit_us"]["repeat_count"] == 2
    assert report["metrics"]["solver_ctor_us"]["repeat_count"] == 2
    assert report["metrics"]["same_case_set_case_us"]["repeat_count"] == 2
    assert report["metrics"]["repeated_solve_ms"]["repeat_count"] == 2
    assert (
        report["case_refresh"]["payload_schema"]
        == "KernelSolver.metadata_json() round-trip payload"
    )
    assert report["threading"]["same_so_multi_thread"]["success"] is True
    assert report["threading"]["same_solver_cross_thread_guard"]["raised"] is True
    assert report["legacy_veqpy_compare"]["status"] == "external_script_available"
