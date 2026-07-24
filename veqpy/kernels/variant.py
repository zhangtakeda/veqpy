"""
Module: veqpy.kernels.variant

Role:
- Build count-only Kernel topology variants without mutating existing topology objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from veqpy.kernels.errors import TopologyError
from veqpy.kernels.types import KernelTopology, _infer_l_max, _infer_m_max


@dataclass(frozen=True, slots=True)
class KernelVariantPlan:
    """Concrete count-only variant topology plus capacity containment result."""

    topology: KernelTopology
    contained: bool


def build_kernel_variant_topology(
    current: KernelTopology,
    *,
    h_count: int | None = None,
    v_count: int | None = None,
    kappa_count: int | None = None,
    psin_count: int | None = None,
    F_count: int | None = None,
    c_counts: tuple[int, ...] | None = None,
    s_counts: tuple[int, ...] | None = None,
) -> KernelVariantPlan:
    """Return a new topology where only active count fields may change."""

    next_counts = {
        "h_count": current.h_count if h_count is None else h_count,
        "v_count": current.v_count if v_count is None else v_count,
        "kappa_count": current.kappa_count if kappa_count is None else kappa_count,
        "psin_count": current.psin_count if psin_count is None else psin_count,
        "F_count": current.F_count if F_count is None else F_count,
        "c_counts": current.c_counts if c_counts is None else c_counts,
        "s_counts": current.s_counts if s_counts is None else s_counts,
    }
    topology = KernelTopology(
        **next_counts,
        Nr=current.Nr,
        Nt=current.Nt,
        route=current.route,
        coordinate=current.coordinate,
        nodes=current.nodes,
        constraint=current.constraint,
        sample_count=current.sample_count,
        quadrature=current.quadrature,
        calculus=current.calculus,
        L_max=current.L_max,
        M_max=current.M_max,
        K_max=current.K_max,
    )
    return KernelVariantPlan(
        topology=topology,
        contained=topology_counts_contained_by_capacity(topology, current),
    )


def topology_counts_contained_by_capacity(
    topology: KernelTopology,
    capacity: KernelTopology,
) -> bool:
    """Return whether ``topology`` active counts fit within capacity limits."""

    required_l = _infer_l_max(
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
    required_m = _infer_m_max(topology.c_counts, topology.s_counts)
    required_k = max(2, required_m)
    return (
        required_l <= capacity.L_max
        and required_m <= capacity.M_max
        and required_k <= capacity.K_max
    )


def require_topology_counts_contained_by_capacity(
    topology: KernelTopology,
    capacity: KernelTopology,
) -> None:
    """Raise when ``topology`` active counts do not fit within capacity limits."""

    if topology_counts_contained_by_capacity(topology, capacity):
        return
    raise TopologyError(
        "variant counts exceed kernel capacity limits: "
        f"required active counts do not fit L_max={capacity.L_max}, "
        f"M_max={capacity.M_max}, K_max={capacity.K_max}"
    )
