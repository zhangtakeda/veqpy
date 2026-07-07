"""
Module: veqpy.kernels.cxx_kernel.affinity

Role:
- Provide scoped CPU affinity helpers for short Cxx backend calls.

Notes:
- Pinning is a runtime policy, not part of artifact identity. Nested calls are
  tracked per Python thread so benchmark loops can pin once around many native
  solves without fighting lower-level pin scopes.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

_DISABLE_TOKENS = {"0", "false", "no", "off", "none", "disable", "disabled"}
_ENABLE_TOKENS = {"1", "true", "yes", "on", "auto", "pin", "enable", "enabled"}
_PIN_STATE = threading.local()


def current_cpu_affinity() -> tuple[int, ...] | None:
    """Return the current Linux CPU affinity set, or ``None`` when unavailable."""

    if not _has_affinity_api():
        return None
    try:
        return tuple(sorted(int(cpu) for cpu in os.sched_getaffinity(0)))
    except OSError:
        return None


def cpu_pin_scope_active() -> bool:
    """Return whether this thread is already inside a Cxx pinning scope."""

    return int(getattr(_PIN_STATE, "depth", 0)) > 0


@contextmanager
def pinned_cpu(policy: bool | int | None = None) -> Iterator[None]:
    """Temporarily pin the current thread/process to one CPU for Cxx calls.

    ``policy=None`` reads the environment and defaults to enabled auto pinning.
    Auto pinning chooses the smallest CPU from the current affinity set so an
    outer ``taskset``/cgroup/SLURM allocation remains authoritative.  The
    previous affinity is restored after the outermost pin scope exits. Nested
    calls are Python-only no-ops so a batch can pin once around many solves.
    """

    resolved = _default_policy_from_env() if policy is None else policy
    if resolved is False:
        yield
        return

    if cpu_pin_scope_active():
        depth = int(getattr(_PIN_STATE, "depth", 0))
        _PIN_STATE.depth = depth + 1
        try:
            yield
        finally:
            _PIN_STATE.depth = depth
        return

    if not _has_affinity_api():
        yield
        return

    previous = set(os.sched_getaffinity(0))
    target = _resolve_target(resolved, previous)
    if target is None:
        yield
        return

    did_pin = False
    if target != previous:
        try:
            os.sched_setaffinity(0, target)
            did_pin = True
        except OSError:
            yield
            return

    try:
        _PIN_STATE.depth = 1
        yield
    finally:
        _PIN_STATE.depth = 0
        if did_pin:
            try:
                os.sched_setaffinity(0, previous)
            except OSError:
                pass


def _has_affinity_api() -> bool:
    return callable(getattr(os, "sched_getaffinity", None)) and callable(
        getattr(os, "sched_setaffinity", None)
    )


def _resolve_target(resolved: bool | int, allowed: set[int]) -> set[int] | None:
    if not allowed:
        return None
    if isinstance(resolved, bool):
        cpu = min(allowed)
    elif isinstance(resolved, int):
        cpu = resolved
    else:
        raise TypeError(f"pin_cpu must be bool, int, or None, got {type(resolved).__name__}")
    if cpu < 0:
        raise ValueError("pin_cpu CPU id must be non-negative")
    if cpu not in allowed:
        cpu = min(allowed)
    return {int(cpu)}


def _default_policy_from_env() -> bool | int:
    token = os.environ.get("VEQLIB_PIN_CPU", "auto").strip().lower()
    if token in _DISABLE_TOKENS:
        return False
    if token in _ENABLE_TOKENS or token == "":
        enabled = True
    else:
        try:
            return int(token)
        except ValueError as exc:
            raise ValueError(
                "VEQLIB_PIN_CPU must be auto/on/1, off/0, or an integer CPU id; "
                "use VEQLIB_PIN_CPU_ID to select CPU 0 unambiguously"
            ) from exc

    if not enabled:
        return False
    requested = os.environ.get("VEQLIB_PIN_CPU_ID")
    if requested is None or requested.strip() == "":
        return True
    try:
        return int(requested)
    except ValueError as exc:
        raise ValueError("VEQLIB_PIN_CPU_ID must be an integer CPU id") from exc
