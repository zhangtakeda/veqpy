"""Validation helpers for VEQlib facade values at native execution boundaries."""

from __future__ import annotations

from veqpy.types import KernelTopology, TopologyError


def validate_supported_for_veqlib_native(topology: KernelTopology) -> None:
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
        mismatches.append("F_count > 0 outside PJ2")
    if topology.source_active_family == "F" and topology.F_count <= 0:
        mismatches.append("PJ2 requires F_count > 0")
    if topology.source_active_family != "psin" and topology.psin_count > 0:
        mismatches.append("source-owned topology does not accept psin_count > 0")
    if mismatches:
        raise TopologyError("unsupported VEQlib native topology: " + "; ".join(mismatches))
