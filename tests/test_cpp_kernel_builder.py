from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from veqpy.cpp import build_kernel
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


def test_build_kernel_dry_run_writes_artifact_plan(tmp_path: Path) -> None:
    topology = make_topology(M_max=10, K_max=10)

    artifact = build_kernel(topology, cache_root=tmp_path, dry_run=True, cxx="clang++")

    assert artifact.root_dir == tmp_path / "v1.0.0" / "fastmath" / artifact.artifact_id
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
    assert artifact.metadata["build_identity"]["veqlib_source_digest"]["file_count"] > 0

    configure = artifact.metadata["build"]["cmake_configure"]
    assert f"-DVEQ_NR={topology.Nr}" in configure
    assert "-DVEQ_SIN_PROFILE_COUNTS=3" in configure
    assert "-DVEQ_BOUNDARY_M_MAX=10" in configure
    assert "-DVEQ_PROFILE_KMAX_LIMIT=10" in configure


def test_build_kernel_artifact_id_distinguishes_build_mode(tmp_path: Path) -> None:
    fastmath = build_kernel(make_topology(build="fastmath"), cache_root=tmp_path, dry_run=True)
    debug = build_kernel(make_topology(build="debug"), cache_root=tmp_path, dry_run=True)

    assert fastmath.artifact_id != debug.artifact_id
    assert fastmath.root_dir.parts[-3:-1] == ("v1.0.0", "fastmath")
    assert debug.root_dir.parts[-3:-1] == ("v1.0.0", "debug")


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
