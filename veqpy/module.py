"""FusionPRIME ``VEQ`` Module, Record, and lifecycle integration."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from fusionprime_base import Equilibrium, Plasma, module, record

from .adapter import VEQAdapter
from .kernels import Kernel, KernelConfig, KernelTopology


@record(outputs=("equilibrium",))
@dataclass(frozen=True, slots=True)
class VEQRecord:
    """Immutable execution record for one VEQ solve and materialization."""

    solved: bool
    accepted: bool
    materialized: bool
    backend: str
    detail: str
    preprocess_ms: float
    solve_ms: float
    postprocess_ms: float
    elapsed_ms: float
    equilibrium: Equilibrium | None
    residual_norm: float
    scaled_residual_norm: float
    evaluations: int
    jacobian_evaluations: int
    source_count: int
    route: str
    coordinate: str


@module
class VEQ:
    """Fixed-boundary VEQ solver with a frozen Plasma-only public run port."""

    def __init__(
        self,
        *,
        topology: KernelTopology,
        backend: str = "numba",
        config: KernelConfig | None = None,
    ) -> None:
        self.topology = topology
        self.backend = str(backend).strip().lower()
        self.config = KernelConfig() if config is None else config
        self.kernel = Kernel(
            topology=topology,
            config=self.config,
            backend=self.backend,
        )
        self.adapter = VEQAdapter(topology, self.kernel.input)
        # The base lifecycle guard forbids preparing a Module recursively from
        # inside run().  Kernel preparation is topology-only, so perform it at
        # construction and keep the public prepare() method idempotent.
        self.prepare()

    def prepare(self) -> None:
        """Compile the configured backend and allocate its persistent workspace."""

        self.kernel.prepare()

    def run(self, *, plasma: Plasma, materialize: bool = True) -> VEQRecord:
        """Solve the Plasma equilibrium and optionally materialize a new State."""

        started = perf_counter()
        preprocess_started = perf_counter()
        source_count = self.adapter.fill(plasma)
        preprocess_ms = (perf_counter() - preprocess_started) * 1000.0
        solve_started = perf_counter()
        output = self.kernel.solve()
        solve_ms = (perf_counter() - solve_started) * 1000.0
        postprocess_started = perf_counter()
        solved = bool(output.success)
        accepted = bool(solved and bool(np.isfinite(output.raw_norm)))
        equilibrium = None
        if accepted and materialize:
            equilibrium = self.kernel.build_equilibrium()
        materialized = equilibrium is not None
        postprocess_ms = (perf_counter() - postprocess_started) * 1000.0
        elapsed_ms = max(
            (perf_counter() - started) * 1000.0,
            preprocess_ms + solve_ms + postprocess_ms,
        )
        detail = "converged" if accepted else "solver did not meet acceptance"
        return VEQRecord(
            solved=solved,
            accepted=accepted,
            materialized=materialized,
            backend=self.backend,
            detail=detail,
            preprocess_ms=float(preprocess_ms),
            solve_ms=float(solve_ms),
            postprocess_ms=float(postprocess_ms),
            elapsed_ms=float(elapsed_ms),
            equilibrium=equilibrium,
            residual_norm=float(output.raw_norm),
            scaled_residual_norm=float(output.scaled_norm),
            evaluations=int(output.nfev),
            jacobian_evaluations=int(output.njev),
            source_count=int(source_count),
            route=self.topology.route,
            coordinate=self.topology.coordinate,
        )

    def new_runtime(self) -> "VEQ":
        """Create an isolated scratch Module with shared immutable configuration."""

        return VEQ(topology=self.topology, backend=self.backend, config=self.config)

    def clear(self) -> None:
        """Clear warm state, output snapshots, and adapter case data."""

        self.kernel.clear()

    def close(self) -> None:
        """Release the backend runtime."""

        self.kernel.close()


__all__ = ["VEQ", "VEQRecord"]
