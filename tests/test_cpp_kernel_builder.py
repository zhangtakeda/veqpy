from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import veqpy.cpp.kernel_builder as kernel_builder
from veqpy.cpp import build_kernel, default_kernel_cache_root
from veqpy.topology import Topology, TopologyError


def make_topology(**overrides: object) -> Topology:
    params: dict[str, object] = {
        "h_count": 3,
        "v_count": 0,
        "kappa_count": 6,
        "psin_count": 6,
        "F_count": 0,
        "c_counts": (),
        "s_counts": (3,),
        "Nr": 32,
        "Nt": 16,
        "route": "PF",
        "coordinate": "psin",
        "constraint": "Ip",
        "nodes": "uniform",
        "sample_count": 8,
    }
    params.update(overrides)
    return Topology(**params)  # type: ignore[arg-type]


def test_default_kernel_cache_root_is_repo_local_artifact_dir() -> None:
    expected = Path(__file__).resolve().parents[1] / "veqlib" / "artifact"
    assert default_kernel_cache_root() == expected


def test_build_kernel_dry_run_writes_artifact_plan(tmp_path: Path) -> None:
    topology = make_topology(M_max=10, K_max=10)

    artifact = build_kernel(topology, cache_root=tmp_path, dry_run=True, cxx="clang++")

    assert artifact.root_dir == tmp_path / "fastmath" / artifact.artifact_id
    assert artifact.metadata_path.exists()
    assert artifact.topology_path.exists()
    assert artifact.build_path.exists()
    assert artifact.kernel_py_path.exists()
    assert not artifact.shared_library_path.exists()
    assert artifact.built is False
    assert artifact.reused is False
    assert artifact.metadata["schema"] == "veqpy.kernel_artifact.v1"
    assert artifact.metadata["artifact"]["status"] == "planned"
    assert artifact.metadata["artifact"]["artifact_id"] == artifact.artifact_id
    assert artifact.metadata["topology"] == topology.to_canonical_dict()
    assert "veqpy_cpp_source_digest" not in artifact.metadata["build_identity"]
    assert artifact.metadata["python_client_source_digest"]["file_count"] > 0
    native_contract = artifact.metadata["build_identity"]["native_build_contract"]
    assert native_contract["defines"]["VEQ_NR"] == topology.Nr
    assert artifact.metadata["build_identity"]["veqlib_source_digest"]["file_count"] > 0

    configure = artifact.metadata["build"]["cmake_configure"]
    assert f"-DVEQ_NR={topology.Nr}" in configure
    assert f"-DVEQ_SOURCE_SAMPLE_COUNT={topology.sample_count}" in configure
    assert "-DVEQ_SIN_PROFILE_COUNTS=3" in configure
    assert "-DVEQ_BOUNDARY_M_MAX=10" in configure
    assert "-DVEQ_PROFILE_KMAX_LIMIT=10" in configure
    assert "-DENABLE_ENZYME=OFF" in configure
    assert "-DVEQLIB_FP_MODE=RELAXED" in configure
    assert f"-DVEQLIB_NB_DOMAIN=veqpy_kernel_{artifact.artifact_id}" in configure
    native_defines = native_contract["defines"]
    assert native_defines["ENABLE_ENZYME"] == "OFF"
    assert native_defines["VEQLIB_FP_MODE"] == "RELAXED"
    nanobind_static = artifact.metadata["common_artifacts"]["nanobind_static"]
    assert nanobind_static["schema"] == "veqpy.nanobind_static_artifact.v1"
    assert nanobind_static["status"] == "planned"
    assert "/_common/nanobind-static/fastmath/" in nanobind_static["archive_path"]
    assert f"-DVEQLIB_PREBUILT_NANOBIND_STATIC={nanobind_static['archive_path']}" in configure
    build = artifact.metadata["build"]["cmake_build"]
    assert "--parallel" in build
    assert int(build[build.index("--parallel") + 1]) >= 1


def test_source_digest_ignores_repo_local_kernel_artifacts(tmp_path: Path) -> None:
    source_dir = tmp_path / "veqlib"
    source_dir.mkdir()
    (source_dir / "kernel_api.h").write_text("stable source\n")
    artifact_generated = source_dir / "artifact" / "fastmath" / "abc" / "cmake-build" / "generated"
    artifact_generated.mkdir(parents=True)
    (artifact_generated / "config.h").write_text("generated artifact header\n")

    digest = kernel_builder._source_digest(source_dir)

    assert digest["file_count"] == 1


def test_build_kernel_artifact_id_distinguishes_build_mode(tmp_path: Path) -> None:
    fastmath = build_kernel(make_topology(build="fastmath"), cache_root=tmp_path, dry_run=True)
    fastmath_enzyme = build_kernel(
        make_topology(build="fastmath-enzyme"), cache_root=tmp_path, dry_run=True
    )
    release = build_kernel(make_topology(build="release"), cache_root=tmp_path, dry_run=True)
    debug = build_kernel(make_topology(build="debug"), cache_root=tmp_path, dry_run=True)

    assert len({
        fastmath.artifact_id,
        fastmath_enzyme.artifact_id,
        release.artifact_id,
        debug.artifact_id,
    }) == 4
    assert fastmath.root_dir.parts[-2] == "fastmath"
    assert fastmath_enzyme.root_dir.parts[-2] == "fastmath-enzyme"
    assert release.root_dir.parts[-2] == "release"
    assert debug.root_dir.parts[-2] == "debug"

    fastmath_configure = fastmath.metadata["build"]["cmake_configure"]
    assert "-DENABLE_ENZYME=OFF" in fastmath_configure
    assert "-DVEQLIB_FP_MODE=RELAXED" in fastmath_configure
    fastmath_enzyme_configure = fastmath_enzyme.metadata["build"]["cmake_configure"]
    assert "-DENABLE_ENZYME=ON" in fastmath_enzyme_configure
    assert "-DVEQLIB_FP_MODE=RELAXED" in fastmath_enzyme_configure
    release_configure = release.metadata["build"]["cmake_configure"]
    assert "-DENABLE_ENZYME=OFF" in release_configure
    assert "-DVEQLIB_FP_MODE=STRICT" in release_configure


def test_build_kernel_artifact_id_ignores_python_client_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    topology = make_topology()
    first = build_kernel(topology, cache_root=tmp_path, dry_run=True)

    monkeypatch.setattr(
        kernel_builder,
        "_python_source_digest",
        lambda: {
            "schema": "veqpy.cpp_python_source_digest.v1",
            "algorithm": "sha256",
            "file_count": 1,
            "files": ["veqpy/cpp/solver.py"],
            "sha256": "changed-python-client",
        },
    )
    second = build_kernel(topology, cache_root=tmp_path, dry_run=True)

    assert second.artifact_id == first.artifact_id
    assert second.root_dir == first.root_dir
    assert second.metadata["python_client_source_digest"]["sha256"] == "changed-python-client"


def test_build_kernel_reuses_verified_existing_artifact(tmp_path: Path) -> None:
    artifact = build_kernel(make_topology(), cache_root=tmp_path, dry_run=True)
    artifact.shared_library_path.write_bytes(b"fake shared library")
    metadata = json.loads(artifact.metadata_path.read_text())
    metadata["artifact"]["status"] = "built"
    metadata["artifact"]["shared_library_sha256"] = hashlib.sha256(
        artifact.shared_library_path.read_bytes()
    ).hexdigest()
    artifact.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    reused = build_kernel(make_topology(), cache_root=tmp_path, dry_run=True)

    assert reused.reused is True
    assert reused.built is False
    assert reused.metadata["artifact"]["status"] == "built"


def test_build_kernel_rejects_unsupported_mvp_topology(tmp_path: Path) -> None:
    with pytest.raises(TopologyError, match="PF/psin/uniform/Ip"):
        build_kernel(make_topology(route="PQ"), cache_root=tmp_path, dry_run=True)
