"""KernelSource semantic lowering for the VEQlib facade.

``KernelSource`` is the public raw case input. Native runtimes still consume the
same scaled arrays as the existing VEQlib core, so this module binds Kernel field
names and topology lengths to the package-level Python-side conversion table.
"""

from __future__ import annotations

from veqlib.source_semantics import MaterializedSourceInputs, materialize_source_inputs

from .types import KernelSource, KernelTopology

KERNEL_SOURCE_ADVICE = (
    "Pass raw case values to KernelSource; facade materialization applies mu0 scaling once."
)
MaterializedKernelSource = MaterializedSourceInputs


def materialize_kernel_source(
    topology: KernelTopology,
    source: KernelSource,
    *,
    case_name: str | None = None,
) -> MaterializedKernelSource:
    """Validate one raw source case and lower it to backend-internal units."""

    if not isinstance(topology, KernelTopology):
        raise TypeError(f"topology must be KernelTopology, got {type(topology).__name__}")
    if not isinstance(source, KernelSource):
        raise TypeError(f"source must be KernelSource, got {type(source).__name__}")
    _validate_source_length(topology, source)
    return materialize_source_inputs(
        route=topology.route,
        heat=source.heat_profile,
        current=source.current_profile,
        Ip=source.Ip,
        beta=source.beta,
        heat_name="heat_profile",
        current_name="current_profile",
        advice=KERNEL_SOURCE_ADVICE,
        case_name=source.case_name if case_name is None else case_name,
    )


def _validate_source_length(topology: KernelTopology, source: KernelSource) -> None:
    expected_samples = topology.sample_count
    heat_length = source.heat_profile.size
    current_length = source.current_profile.size
    if heat_length != expected_samples or current_length != expected_samples:
        raise ValueError(
            "case does not match kernel topology: heat_profile and current_profile "
            f"must have length {expected_samples} for "
            f"route={topology.route}/{topology.coordinate}/{topology.nodes}, "
            f"got {heat_length} and {current_length}"
        )
