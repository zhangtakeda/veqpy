from __future__ import annotations

import fcntl
import json
from pathlib import Path

from veqlib.facade import KernelArtifact, KernelTopology, build_kernel, clean
from veqlib.facade.builder import touch_artifact_used


def test_kernel_builder_dry_run_emits_full_topology_contract(tmp_path) -> None:
    topology = KernelTopology(
        h_count=3,
        v_count=0,
        kappa_count=6,
        psin_count=0,
        F_count=0,
        c_counts=(),
        s_counts=(3,),
        Nr=32,
        Nt=16,
        route="PQ",
        coordinate="rho",
        constraint="beta",
        nodes="grid",
        layout="family",
        build="release",
        fp_mode="FMA",
        enable_native_optimizations=False,
        enable_thin_lto=False,
        analysis=True,
        enzyme_jacobian_batch_width=8,
    )

    artifact = build_kernel(topology, cache_root=tmp_path, dry_run=True)

    assert artifact.built is False
    common_archive = artifact.metadata["common_artifacts"]["nanobind_static"]["archive_path"]
    assert str(tmp_path / "release" / "common") in common_archive
    assert "_common" not in Path(common_archive).parts
    assert "nanobind-static" not in Path(common_archive).parts
    assert artifact.metadata["topology"]["source"] == {
        "route_key": ["PQ", "rho", "grid"],
        "route": "PQ",
        "route_code": 6,
        "coordinate": "rho",
        "coordinate_code": 1,
        "constraint": "beta",
        "constraint_code": 2,
        "supported_constraints": ["Ip_beta", "Ip", "beta", "null"],
        "uses_Ip": False,
        "uses_beta": True,
        "nodes": "grid",
        "nodes_code": 2,
        "sample_count": 32,
        "active_family": "none",
        "active_family_code": 0,
        "parameterization": "identity",
        "parameterization_code": 0,
    }
    assert artifact.metadata["topology"]["layout"] == {
        "packed": "family",
        "profile_first": True,
        "code": 1,
        "profile_order": [
            "h",
            "v",
            "k",
            "c0",
            "c1",
            "s1",
            "psin",
            "F",
        ],
    }

    configure = artifact.metadata["build"]["cmake_configure"]
    assert "-DVEQ_SOURCE_ROUTE_CODE=6" in configure
    assert "-DVEQ_SOURCE_COORDINATE_CODE=1" in configure
    assert "-DVEQ_SOURCE_CONSTRAINT_CODE=2" in configure
    assert "-DVEQ_SOURCE_NODES_CODE=2" in configure
    assert "-DVEQ_SOURCE_ACTIVE_FAMILY_CODE=0" in configure
    assert "-DVEQ_SOURCE_PARAMETERIZATION_CODE=0" in configure
    assert "-DVEQ_LAYOUT_PROFILE_FIRST=1" in configure
    assert "-DVEQ_ENZYME_JACOBIAN_BATCH_WIDTH=8" in configure
    assert "-DVEQLIB_FP_MODE=FMA" in configure
    assert "-DVEQLIB_ENABLE_NATIVE_OPTIMIZATIONS=OFF" in configure
    assert "-DVEQLIB_ENABLE_THIN_LTO=OFF" in configure
    assert "-DVEQLIB_ANALYSIS_BUILD=ON" in configure


def test_kernel_builder_forces_clang18_toolchain(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VEQLIB_CXX", "definitely-not-used")
    monkeypatch.setenv("CXX", "also-not-used")
    topology = KernelTopology(
        h_count=2,
        v_count=0,
        kappa_count=2,
        psin_count=3,
        F_count=0,
        c_counts=(),
        s_counts=(2,),
        Nr=8,
        Nt=8,
        route="PF",
        coordinate="psin",
        constraint="Ip",
        nodes="uniform",
        sample_count=9,
    )

    artifact = build_kernel(topology, cache_root=tmp_path, dry_run=True)

    assert artifact.metadata["build_identity"]["tools"]["cxx"] == "clang++-18"
    assert "-DCMAKE_CXX_COMPILER=clang++-18" in artifact.metadata["build"]["cmake_configure"]


def test_kernel_cache_clean_filters_by_timestamp_and_reports_dry_run(tmp_path) -> None:
    old_root = _write_artifact_metadata(
        tmp_path,
        build="fastmath",
        artifact_id="old-artifact",
        built_at="2020-01-01T00:00:00Z",
        last_used_at="2020-01-02T00:00:00Z",
    )
    new_root = _write_artifact_metadata(
        tmp_path,
        build="fastmath",
        artifact_id="new-artifact",
        built_at="2020-01-01T00:00:00Z",
        last_used_at="2099-01-01T00:00:00Z",
    )

    planned = clean(
        cache_root=tmp_path,
        build="fastmath",
        older_than="2021-01-01T00:00:00Z",
        dry_run=True,
    )

    assert planned.dry_run is True
    assert planned.removed == (old_root,)
    assert planned.skipped_recent == (new_root,)
    assert planned.skipped_locked == ()
    assert planned.errors == ()
    assert planned.bytes_removed > 0
    assert old_root.exists()
    assert new_root.exists()

    removed = clean(
        cache_root=tmp_path,
        build="fastmath",
        older_than="2021-01-01T00:00:00Z",
    )

    assert removed.dry_run is False
    assert removed.removed == (old_root,)
    assert removed.skipped_recent == (new_root,)
    assert removed.errors == ()
    assert not old_root.exists()
    assert new_root.exists()


def test_kernel_cache_clean_skips_locked_artifacts(tmp_path) -> None:
    locked_root = _write_artifact_metadata(
        tmp_path,
        build="fastmath",
        artifact_id="locked-artifact",
        built_at="2020-01-01T00:00:00Z",
        last_used_at="2020-01-02T00:00:00Z",
    )
    lock_path = tmp_path / "fastmath" / "locked-artifact.lock"

    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = clean(
                cache_root=tmp_path,
                build="fastmath",
                older_than="2021-01-01T00:00:00Z",
            )
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    assert result.removed == ()
    assert result.skipped_locked == (locked_root,)
    assert result.errors == ()
    assert locked_root.exists()


def test_touch_artifact_used_updates_metadata_timestamp(tmp_path) -> None:
    artifact_id = "loaded-artifact"
    root = _write_artifact_metadata(
        tmp_path,
        build="fastmath",
        artifact_id=artifact_id,
        built_at="2020-01-01T00:00:00Z",
        last_used_at="2020-01-02T00:00:00Z",
    )
    metadata_path = root / "metadata.json"
    artifact = KernelArtifact(
        topology=_minimal_topology(),
        artifact_id=artifact_id,
        root_dir=root,
        cmake_build_dir=root / "cmake-build",
        metadata_path=metadata_path,
        topology_path=root / "topology.json",
        build_path=root / "build.json",
        kernel_py_path=root / "kernel.py",
        shared_library_path=root / "veqlib.so",
        metadata=json.loads(metadata_path.read_text()),
        reused=False,
        built=True,
    )

    touch_artifact_used(artifact)

    reloaded = json.loads(metadata_path.read_text())
    assert reloaded["artifact"]["last_used_at"] != "2020-01-02T00:00:00Z"
    assert artifact.metadata["artifact"]["last_used_at"] == reloaded["artifact"]["last_used_at"]


def _minimal_topology() -> KernelTopology:
    return KernelTopology(
        h_count=2,
        v_count=0,
        kappa_count=2,
        psin_count=3,
        F_count=0,
        c_counts=(),
        s_counts=(2,),
        Nr=8,
        Nt=8,
        route="PF",
        coordinate="psin",
        constraint="Ip",
        nodes="uniform",
        sample_count=9,
    )


def _write_artifact_metadata(
    cache_root: Path,
    *,
    build: str,
    artifact_id: str,
    built_at: str,
    last_used_at: str,
) -> Path:
    root = cache_root / build / artifact_id
    root.mkdir(parents=True)
    (root / "payload.bin").write_bytes(b"veqlib")
    metadata = {
        "schema": "veqlib.kernel_artifact.v1",
        "artifact": {
            "artifact_id": artifact_id,
            "status": "built",
            "module_name": f"veqlib._kernel_cache.k_{artifact_id}.veqlib_ext",
            "shared_library": "veqlib.so",
            "shared_library_sha256": "not-used-by-clean",
            "built_at": built_at,
            "last_reused_at": None,
            "last_used_at": last_used_at,
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    return root
