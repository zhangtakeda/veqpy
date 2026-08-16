"""Backend-neutral four-buffer Kernel handle."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .contracts import KernelConfig, KernelInput, KernelOutput, KernelTopology
from .dispatch import _make_kernel_impl
from .lowering import boundary_case, private_config, source_case
from .lowering import build_policy as build_private_policy

if TYPE_CHECKING:
    from fusionprime_base import Equilibrium


class Kernel:
    """Prepared numerical runtime bound to exactly four public data objects.

    ``KernelInput`` and ``KernelOutput`` are retained by identity for the
    lifetime of this handle.  Backend lowering may create private read-only
    views, but no old boundary/source/result object crosses this public API.
    """

    def __init__(
        self,
        *,
        topology: KernelTopology,
        input: KernelInput | None = None,
        config: KernelConfig | None = None,
        output: KernelOutput | None = None,
        backend: str = "numba",
        build_policy: object | None = None,
        registry: object | None = None,
        cache_root: Path | None = None,
        source_dir: Path | None = None,
        pin_cpu: bool | int | None = None,
    ) -> None:
        if not isinstance(topology, KernelTopology):
            raise TypeError(f"topology must be KernelTopology, got {type(topology).__name__}")
        selected_backend = str(backend).strip().lower()
        if selected_backend not in {
            "numba",
            "cxx",
            "cxx-strict",
            "cxx-relaxed",
            "cxx-enzyme",
        }:
            raise ValueError(
                "backend must be 'numba', 'cxx', 'cxx-strict', 'cxx-relaxed', or 'cxx-enzyme'"
            )
        self.topology = topology
        self.input = KernelInput.allocate(topology) if input is None else input
        self.output = KernelOutput.allocate(topology) if output is None else output
        _validate_input(self.input, topology)
        _validate_output(self.output, topology)
        self.backend = "cxx-relaxed" if selected_backend == "cxx" else selected_backend
        self.config = KernelConfig() if config is None else _as_config(config)
        self._private_config = private_config(self.config)
        self._policy = (
            build_private_policy(backend=selected_backend)
            if build_policy is None
            else build_policy
        )
        self._impl = _make_kernel_impl(
            topology=topology,
            recipe=self._policy,
            config=self._private_config,
            registry=registry,
            cache_root=cache_root,
            source_dir=source_dir,
            pin_cpu=pin_cpu,
        )
        self._prepared = False
        self._closed = False
        self._has_solve = False
        self._last_equilibrium: Equilibrium | None = None

    @property
    def x_size(self) -> int:
        """Packed unknown vector size fixed by ``KernelTopology``."""

        return int(self.topology.x_size)

    @property
    def prepared(self) -> bool:
        """Whether backend compilation and workspace preparation completed."""

        return self._prepared

    def prepare(self, *, force: bool = False) -> None:
        """Compile and allocate backend state exactly once per topology."""

        self._ensure_open()
        if force:
            self._prepared = False
        if not self._prepared:
            self._impl.prepare(force=force)
            self._prepared = True

    def solve(self, *, config: KernelConfig | None = None) -> KernelOutput:
        """Run the numerical solve and return the current KO snapshot.

        Equilibrium materialization is deliberately outside this path.  A
        caller that needs a frozen State must explicitly call
        :meth:`build_equilibrium` after this method returns.
        """

        self._ensure_open()
        # A new solve owns the only valid materialization point.  Invalidate
        # both the cached State and the KO root snapshot before any backend
        # work so an exception cannot expose the previous solve's State.
        self._last_equilibrium = None
        self._has_solve = False
        _reset_equilibrium_roots(self.output)
        _validate_input(self.input, self.topology)
        if not self._prepared:
            self.prepare()
        self.input.clear_unused_source_tail()
        boundary = boundary_case(self.topology, self.input)
        source = source_case(self.topology, self.input)
        x0 = self.input.x0 if self.input.has_x0 and self.input.x0 is not None else None
        active_config = self.config if config is None else _as_config(config)
        snapshot = self._impl.solve(
            boundary,
            source,
            config=private_config(active_config),
            case_name=None,
            x0=x0,
        )
        _copy_snapshot(self.output, snapshot)
        self._has_solve = True
        return self.output

    def residual_into(self, out: np.ndarray, x: Any | None = None) -> None:
        """Write the residual for a packed state into caller-owned ``out``."""

        self._ensure_open()
        _validate_input(self.input, self.topology)
        if not self._prepared:
            self.prepare()
        packed = self.output.x if x is None else _packed_x(x, self.x_size)
        self._impl.residual_into(
            _packed_out(out, (self.x_size,), "out"),
            packed,
            boundary_case(self.topology, self.input),
            source_case(self.topology, self.input),
        )

    def residual(self, x: Any | None = None) -> np.ndarray:
        """Convenience residual allocation outside the hot path."""

        out = np.empty(self.x_size, dtype=np.float64)
        self.residual_into(out, x)
        return out

    def residual_jvp_into(self, out: np.ndarray, x: Any, v: Any) -> None:
        """Write the residual Jacobian-vector product into ``out``."""

        self._ensure_open()
        _validate_input(self.input, self.topology)
        if not self._prepared:
            self.prepare()
        self._impl.jvp_into(
            _packed_out(out, (self.x_size,), "out"),
            _packed_x(x, self.x_size),
            _packed_x(v, self.x_size),
            boundary_case(self.topology, self.input),
            source_case(self.topology, self.input),
        )

    def jacobian_into(self, out: np.ndarray, x: Any | None = None) -> None:
        """Write the residual Jacobian into a caller-owned matrix."""

        self._ensure_open()
        _validate_input(self.input, self.topology)
        if not self._prepared:
            self.prepare()
        packed = self.output.x if x is None else _packed_x(x, self.x_size)
        self._impl.jacobian_into(
            _packed_out(out, (self.x_size, self.x_size), "out"),
            packed,
            boundary_case(self.topology, self.input),
            source_case(self.topology, self.input),
        )

    def build_equilibrium(self) -> "Equilibrium":
        """Return a newly owned frozen base Equilibrium snapshot."""

        self._ensure_open()
        if not self._has_solve:
            raise RuntimeError("build_equilibrium requires a previous solve")
        if self._last_equilibrium is None:
            if not self._prepared:
                self.prepare()
            self._last_equilibrium = self._impl.build_equilibrium()
            _copy_equilibrium_roots(self.output, self._last_equilibrium)
        return self._last_equilibrium

    def clear(self) -> None:
        """Clear transient backend state while retaining ABI buffer identity."""

        if self._closed:
            return
        self._impl.clear()
        self.output.reset()
        _reset_equilibrium_roots(self.output)
        self._has_solve = False
        self._last_equilibrium = None

    def close(self) -> None:
        """Release backend resources; repeated close is harmless."""

        if self._closed:
            return
        self._impl.close()
        self._closed = True

    def pinned(self) -> AbstractContextManager[None, bool | None]:
        """Return the optional backend CPU-affinity scope."""

        self._ensure_open()
        return self._impl.pinned()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Kernel is closed")


def _as_config(value: KernelConfig) -> KernelConfig:
    if not isinstance(value, KernelConfig):
        raise TypeError(f"config must be KernelConfig, got {type(value).__name__}")
    return value


def _validate_input(value: KernelInput, topology: KernelTopology) -> None:
    if not isinstance(value, KernelInput):
        raise TypeError(f"input must be KernelInput, got {type(value).__name__}")
    if type(value.source_count) is not int or value.source_count < 0:
        raise ValueError("KernelInput source_count must be a non-negative int")
    if value.source_count > 1024:
        raise ValueError("VEQ source count is limited to 1024 nodes")
    if value.source_count > value.source_capacity:
        raise ValueError("KernelInput source_count exceeds its resident capacity")
    if value.driver.size != value.pressure.size or value.source_nodes.size != value.pressure.size:
        raise ValueError("KernelInput source arrays must share one capacity")
    if value.x0 is not None and value.x0.shape != (topology.x_size,):
        raise ValueError(f"KernelInput.x0 must have shape {(topology.x_size,)}")


def _validate_output(value: KernelOutput, topology: KernelTopology) -> None:
    if not isinstance(value, KernelOutput):
        raise TypeError(f"output must be KernelOutput, got {type(value).__name__}")
    for name in ("x", "raw", "scaled"):
        array = getattr(value, name)
        if array.shape != (topology.x_size,) or array.dtype != np.float64 or not array.flags.c_contiguous:
            raise ValueError(f"KernelOutput.{name} must be C-contiguous float64 shape {(topology.x_size,)}")
    for name in ("psin", "psin_r", "psin_rr", "FF_psi", "P_psi"):
        array = getattr(value, name)
        if array.shape != (topology.Nr,) or array.dtype != np.float64 or not array.flags.c_contiguous:
            raise ValueError(f"KernelOutput.{name} must be C-contiguous float64 shape {(topology.Nr,)}")


def _packed_x(value: Any, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"packed state must have shape {(size,)}, got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float64)


def _packed_out(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.shape != shape:
        raise ValueError(f"{name} must be a C-contiguous float64 array with shape {shape}")
    if not value.flags.c_contiguous or not value.flags.writeable:
        raise ValueError(f"{name} must be writable and C-contiguous")
    return value


def _copy_snapshot(output: KernelOutput, snapshot: object) -> None:
    """Copy private solver diagnostics without exposing the snapshot object."""

    output.success = bool(snapshot.success)
    output.info = int(snapshot.info)
    output.nfev = int(snapshot.nfev)
    output.njev = int(snapshot.njev)
    output.callbacks = int(snapshot.callbacks)
    output.jacobian_component_evaluations = int(snapshot.jacobian_component_evaluations)
    output.jvp_evaluations = int(snapshot.jvp_evaluations)
    output.linear_iterations = int(snapshot.linear_iterations)
    output.raw_norm = float(snapshot.raw_norm)
    output.scaled_norm = float(snapshot.scaled_norm)
    output.elapsed_ms = float(snapshot.elapsed_ms)
    output.preprocess_ms = float(snapshot.preprocess_ms)
    output.solve_ms = float(snapshot.solver_ms)
    output.postprocess_ms = float(snapshot.postprocess_ms)
    np.copyto(output.x, snapshot.x)
    np.copyto(output.raw, snapshot.raw)
    np.copyto(output.scaled, snapshot.scaled)
    np.copyto(output.alpha, snapshot.alpha)


def _copy_equilibrium_roots(output: KernelOutput, equilibrium: object) -> None:
    """Copy materialization roots into output-owned arrays."""

    np.copyto(output.psin, np.asarray(equilibrium.psin, dtype=np.float64))
    np.copyto(output.psin_r, np.asarray(equilibrium.psin_r, dtype=np.float64))
    np.copyto(output.psin_rr, np.asarray(equilibrium.psin_rr, dtype=np.float64))
    np.copyto(output.FF_psi, np.asarray(equilibrium.FF_psi, dtype=np.float64))
    np.copyto(output.P_psi, np.asarray(equilibrium.P_psi, dtype=np.float64))


def _reset_equilibrium_roots(output: KernelOutput) -> None:
    """Mark KO materialization roots unavailable until an explicit build."""

    for name in ("psin", "psin_r", "psin_rr", "FF_psi", "P_psi"):
        getattr(output, name).fill(np.nan)


__all__ = ["Kernel"]
