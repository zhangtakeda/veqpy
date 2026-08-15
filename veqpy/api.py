"""Function-style entry points over the four-buffer Kernel API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .kernels import Kernel, KernelConfig, KernelInput, KernelOutput, KernelTopology

__all__ = ["build", "solve"]


def build(
    *,
    topology: KernelTopology,
    input: KernelInput | None = None,
    output: KernelOutput | None = None,
    config: KernelConfig | None = None,
    backend: str = "numba",
    build_policy: object | None = None,
    registry: object | None = None,
    cache_root: Path | None = None,
    source_dir: Path | None = None,
    pin_cpu: bool | int | None = None,
) -> Kernel:
    """Construct and prepare a reusable Kernel handle."""

    kernel = Kernel(
        topology=topology,
        input=input,
        output=output,
        config=config,
        backend=backend,
        build_policy=build_policy,
        registry=registry,
        cache_root=cache_root,
        source_dir=source_dir,
        pin_cpu=pin_cpu,
    )
    kernel.prepare()
    return kernel


def solve(
    *,
    topology: KernelTopology,
    input: KernelInput,
    output: KernelOutput | None = None,
    config: KernelConfig | None = None,
    backend: str = "numba",
    **_: Any,
) -> KernelOutput:
    """Prepare a short-lived Kernel, solve, and release its backend resources."""

    kernel = Kernel(
        topology=topology,
        input=input,
        output=output,
        config=config,
        backend=backend,
    )
    try:
        kernel.prepare()
        return kernel.solve()
    finally:
        kernel.close()
