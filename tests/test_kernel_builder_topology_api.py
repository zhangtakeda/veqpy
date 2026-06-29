from __future__ import annotations

from veqlib.facade import KernelTopology, build_kernel


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
