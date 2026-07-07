"""
Module: veqpy.kernels.abi.identity

Role:
- Build canonical Kernel payloads used for artifact identity and metadata.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

_KERNEL_TOPOLOGY_KEY_LENGTH = 32


def recipe_identity_payload(recipe: Any) -> dict[str, object]:
    return {
        "backend": recipe.backend,
        "preset": recipe.build,
        "layout": {
            "packed": recipe.layout,
            "profile_first": recipe.layout_profile_first,
            "code": recipe.layout_code,
        },
        "cmake_build_type": recipe.cmake_build_type,
        "fp_mode": recipe.fp_mode,
        "enable_enzyme": recipe.enable_enzyme,
        "enable_native_optimizations": recipe.enable_native_optimizations,
        "enable_thin_lto": recipe.enable_thin_lto,
        "analysis": recipe.analysis,
        "enzyme_jacobian_batch_width": recipe.enzyme_jacobian_batch_width,
    }


def topology_identity_payload(topology: Any) -> dict[str, Any]:
    return {
        "profiles": {
            "h_count": topology.h_count,
            "v_count": topology.v_count,
            "kappa_count": topology.kappa_count,
            "psin_count": topology.psin_count,
            "F_count": topology.F_count,
            "c_counts": list(topology.c_counts),
            "s_counts": list(topology.s_counts),
            "L_max": topology.L_max,
            "M_max": topology.M_max,
            "K_max": topology.K_max,
        },
        "grid": {
            "Nr": topology.Nr,
            "Nt": topology.Nt,
            "quadrature": topology.quadrature,
            "calculus": topology.calculus,
        },
        "source": source_policy_payload(topology),
    }


def source_policy_payload(topology: Any) -> dict[str, Any]:
    return {
        "route_key": list(topology.source_route_key),
        "route": topology.route,
        "route_code": topology.source_route_code,
        "coordinate": topology.coordinate,
        "coordinate_code": topology.source_coordinate_code,
        "constraint": topology.constraint_label,
        "constraint_code": topology.source_constraint_code,
        "supported_constraints": list(topology.source_supported_constraints),
        "uses_Ip": topology.source_uses_ip_constraint,
        "uses_beta": topology.source_uses_beta_constraint,
        "nodes": topology.nodes,
        "nodes_code": topology.source_nodes_code,
        "sample_count": topology.sample_count,
        "active_family": topology.source_active_family,
        "active_family_code": topology.source_active_family_code,
        "parameterization": topology.source_parameterization,
        "parameterization_code": topology.source_parameterization_code,
    }


def topology_json_bytes(topology: Any) -> bytes:
    return json.dumps(
        topology_identity_payload(topology),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def compute_topology_key(topology: Any) -> str:
    digest = hashlib.sha256(topology_json_bytes(topology)).digest()
    encoded = base64.b32encode(digest).decode("ascii").lower().rstrip("=")
    return encoded[:_KERNEL_TOPOLOGY_KEY_LENGTH]
