"""
Module: veqpy.kernels.cxx_kernel.validation

Role:
- Validate topology features supported by the Cxx backend.
"""

from __future__ import annotations

from veqpy.kernels.errors import TopologyError
from veqpy.kernels.types import KernelTopology, _infer_l_max


def validate_supported_for_cxx_backend(topology: KernelTopology) -> None:
    mismatches: list[str] = []
    if topology.quadrature != "legendre":
        mismatches.append(f"quadrature={topology.quadrature!r}")
    if topology.calculus != "spectral":
        mismatches.append(f"calculus={topology.calculus!r}")
    if topology.constraint_label not in topology.source_supported_constraints:
        mismatches.append(
            f"route_key={topology.source_route_key!r}, "
            f"constraint={topology.constraint_label!r}"
        )
    if topology.source_active_family != "F" and topology.F_count > 0:
        mismatches.append("F_count > 0 outside PJ2/PJ3")
    if topology.source_active_family == "F" and topology.F_count <= 0:
        mismatches.append("PJ2/PJ3 requires F_count > 0")
    if topology.source_active_family != "psin" and topology.psin_count > 0:
        mismatches.append("source-owned topology does not accept psin_count > 0")
    inferred_l_max = _infer_l_max(
        (
            topology.h_count,
            topology.v_count,
            topology.kappa_count,
            topology.psin_count,
            topology.F_count,
            *topology.c_counts,
            *topology.s_counts,
        )
    )
    if topology.L_max > inferred_l_max:
        mismatches.append("capacity-style L_max > inferred is not supported by Cxx")
    if mismatches:
        raise TopologyError("unsupported Cxx topology: " + "; ".join(mismatches))
