"""FusionPRIME ``VEQ`` Module, Record, and high-level lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from fusionprime_base import Equilibrium, module, record
from rich.console import Console

from .adapter import VEQAdapter
from .config import (
    merge_solver,
    normalize_artifact_dir,
    normalize_backend,
    normalize_cpu_affinity,
    normalize_rebuild,
    normalize_topology,
    solver_config,
)
from .io import write_report
from .kernels.contracts import KernelConfig, KernelTopology
from .kernels.kernel import Kernel

_UNSET = object()


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
    source_capacity: int
    capacity_epoch: int
    route: str
    coordinate: str
    report_path: str | None = None


@module(inputs=("boundary", "source", "targets"))
class VEQ:
    """Fixed-boundary VEQ solver with three standalone physical input ports.

    The constructor accepts only plain topology/solver mappings. Boundary,
    source, and target dictionaries are copied directly into private
    four-buffer ABI objects.
    """

    def __init__(
        self,
        *,
        topology: Mapping[str, Any],
        solver: Mapping[str, Any] | None = None,
        backend: str = "numba",
        artifact_dir: str | Path | None = None,
        cpu_affinity: bool | int | None = None,
        rebuild: bool = False,
        materialize: bool = True,
        verbose: bool = True,
        report: bool = False,
        report_dir: str | Path | None = None,
    ) -> None:
        if not isinstance(topology, Mapping):
            raise TypeError(f"topology must be a dict-like mapping, got {type(topology).__name__}")
        self._topology_mapping = dict(topology)
        self._solver_mapping = merge_solver(solver)
        self._topology: KernelTopology = normalize_topology(self._topology_mapping)
        self._backend = normalize_backend(backend)
        self._artifact_dir = normalize_artifact_dir(artifact_dir)
        self._cpu_affinity = normalize_cpu_affinity(cpu_affinity)
        self._rebuild = normalize_rebuild(rebuild)
        self._materialize_default = _require_bool(materialize, "materialize")
        self._verbose_default = _require_bool(verbose, "verbose")
        self._report_default = _require_bool(report, "report")
        self._report_dir_default = None if report_dir is None else Path(report_dir).expanduser()
        self._config: KernelConfig = solver_config(self._solver_mapping)
        self._kernel = Kernel(
            topology=self._topology,
            config=self._config,
            backend=self._backend,
            cache_root=self._artifact_dir,
            pin_cpu=self._cpu_affinity,
        )
        self._adapter = VEQAdapter(self._topology, self._kernel.input)
        self._active_options: dict[str, object] | None = None
        self.prepare()

    @property
    def topology(self) -> dict[str, Any]:
        """Return a copy of the user topology mapping."""

        return dict(self._topology_mapping)

    @property
    def backend(self) -> str:
        """Return the normalized build backend."""

        return self._backend

    @property
    def source_capacity(self) -> int:
        """Return the current resident source buffer capacity."""

        return self._kernel.input.source_capacity

    @property
    def capacity_epoch(self) -> int:
        """Return the source buffer capacity epoch."""

        return self._kernel.input.capacity_epoch

    def prepare(self) -> None:
        """Compile the configured backend and allocate persistent workspace."""

        self._kernel.prepare(force=self._rebuild)

    def run(
        self,
        *,
        boundary: dict,
        source: dict,
        targets: dict,
        materialize: bool = True,
    ) -> VEQRecord:
        """Solve one explicit standalone problem using behavior defaults."""

        options = self._active_options
        if options is None:
            options = {
                "solver": None,
                "materialize": (
                    materialize
                    if getattr(self, "_materialize_argument_explicit", False)
                    else self._materialize_default
                ),
                "verbose": self._verbose_default,
                "report": self._report_default,
                "report_dir": self._report_dir_default,
            }
        else:
            options = dict(options)
            options["materialize"] = materialize
        return self._run_with_options(boundary, source, targets, options)

    def solve(
        self,
        *,
        boundary: dict,
        source: dict,
        targets: dict,
        solver: Mapping[str, Any] | None = None,
        materialize: bool | object = _UNSET,
        verbose: bool | object = _UNSET,
        report: bool | object = _UNSET,
        report_dir: str | Path | None | object = _UNSET,
    ) -> VEQRecord:
        """Run once with optional solver and behavior overrides."""

        options = {
            "solver": solver,
            "materialize": self._materialize_default
            if materialize is _UNSET
            else _require_bool(materialize, "materialize"),
            "verbose": self._verbose_default if verbose is _UNSET else _require_bool(verbose, "verbose"),
            "report": self._report_default if report is _UNSET else _require_bool(report, "report"),
            "report_dir": self._report_dir_default
            if report_dir is _UNSET
            else (None if report_dir is None else Path(report_dir).expanduser()),
        }
        self._active_options = options
        try:
            return self.run(
                boundary=boundary,
                source=source,
                targets=targets,
                materialize=bool(options["materialize"]),
            )
        finally:
            self._active_options = None

    def new_runtime(self) -> "VEQ":
        """Create an isolated scratch Module for forward finite differences."""

        return VEQ(
            topology=self._topology_mapping,
            solver=self._solver_mapping,
            backend=self._backend,
            artifact_dir=self._artifact_dir,
            cpu_affinity=self._cpu_affinity,
            rebuild=False,
            materialize=True,
            verbose=False,
            report=False,
            report_dir=None,
        )

    def clear(self) -> None:
        """Clear warm state, output snapshots, and adapter case data."""

        self._kernel.clear()

    def close(self) -> None:
        """Release the backend runtime."""

        self._kernel.close()

    def _run_with_options(
        self,
        boundary: dict,
        source: dict,
        targets: dict,
        options: Mapping[str, object],
    ) -> VEQRecord:
        started = perf_counter()
        preprocess_started = perf_counter()
        source_count = self._adapter.fill(boundary, source, targets)
        preprocess_ms = (perf_counter() - preprocess_started) * 1000.0

        run_config = solver_config(merge_solver(self._solver_mapping, options.get("solver")))
        solve_started = perf_counter()
        output = self._kernel.solve(config=run_config)
        solve_ms = (perf_counter() - solve_started) * 1000.0

        postprocess_started = perf_counter()
        solved = bool(output.success)
        accepted = bool(solved and np.isfinite(output.raw_norm))
        equilibrium = None
        materialized = bool(options["materialize"])
        if accepted and materialized:
            equilibrium = self._kernel.build_equilibrium()
        materialized = equilibrium is not None
        postprocess_ms = (perf_counter() - postprocess_started) * 1000.0
        elapsed_ms = max(
            (perf_counter() - started) * 1000.0,
            preprocess_ms + solve_ms + postprocess_ms,
        )
        report_path: Path | None = None
        if bool(options["verbose"]):
            _print_kernel_diagnostics(self._backend, output, source_count, self.source_capacity)
        if bool(options["report"]):
            report_path = write_report(
                topology=self._topology,
                config=run_config,
                input_buffer=self._kernel.input,
                output=output,
                report_dir=options["report_dir"],
                backend=self._backend,
            )
        detail = "converged" if accepted else "solver did not meet acceptance"
        return VEQRecord(
            solved=solved,
            accepted=accepted,
            materialized=materialized,
            backend=self._backend,
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
            source_capacity=self.source_capacity,
            capacity_epoch=self.capacity_epoch,
            route=self._topology.route,
            coordinate=self._topology.coordinate,
            report_path=None if report_path is None else str(report_path),
        )


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _print_kernel_diagnostics(backend: str, output: object, source_count: int, capacity: int) -> None:
    """Print the small stable KO diagnostic surface through Rich."""

    Console().print(
        f"[bold]VEQ KernelOutput[/bold] backend={backend} "
        f"success={bool(output.success)} raw_norm={float(output.raw_norm):.3e} "
        f"scaled_norm={float(output.scaled_norm):.3e} evaluations={int(output.nfev)} "
        f"source_count={source_count} capacity={capacity}"
    )


def _track_explicit_materialize_argument(cls: type[VEQ]) -> type[VEQ]:
    """Preserve the base run signature while distinguishing omitted True."""

    checked_run = cls.run

    @wraps(checked_run)
    def tracked_run(instance: VEQ, *args: object, **kwargs: object) -> VEQRecord:
        instance._materialize_argument_explicit = "materialize" in kwargs
        try:
            return checked_run(instance, *args, **kwargs)
        finally:
            instance._materialize_argument_explicit = False

    cls.run = tracked_run  # type: ignore[method-assign]
    return cls


VEQ = _track_explicit_materialize_argument(VEQ)


__all__ = ["VEQ", "VEQRecord"]
