from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeAlias

CpuPinning: TypeAlias = bool | int | None

_DISABLE_TOKENS = {"0", "false", "no", "off", "none", "disable", "disabled"}
_ENABLE_TOKENS = {"1", "true", "yes", "on", "auto", "pin", "enable", "enabled"}


def current_cpu_affinity() -> tuple[int, ...] | None:
    """Return the current Linux CPU affinity set, or ``None`` when unavailable."""

    if not _has_affinity_api():
        return None
    try:
        return tuple(sorted(int(cpu) for cpu in os.sched_getaffinity(0)))
    except OSError:
        return None


@contextmanager
def pinned_cpu(policy: CpuPinning = None) -> Iterator[None]:
    """Temporarily pin the current thread/process to one CPU for a VEQlib call.

    ``policy=None`` reads the environment and defaults to enabled auto pinning.
    Auto pinning chooses the smallest CPU from the current affinity set so an
    outer ``taskset``/cgroup/SLURM allocation remains authoritative.  The
    previous affinity is restored after the call.
    """

    if not _has_affinity_api():
        yield
        return

    previous = set(os.sched_getaffinity(0))
    target = _resolve_target(policy, previous)
    if target is None or target == previous:
        yield
        return

    did_pin = False
    try:
        os.sched_setaffinity(0, target)
        did_pin = True
    except OSError:
        yield
        return

    try:
        yield
    finally:
        if did_pin:
            try:
                os.sched_setaffinity(0, previous)
            except OSError:
                pass


def _has_affinity_api() -> bool:
    return callable(getattr(os, "sched_getaffinity", None)) and callable(
        getattr(os, "sched_setaffinity", None)
    )


def _resolve_target(policy: CpuPinning, allowed: set[int]) -> set[int] | None:
    if not allowed:
        return None
    resolved = _default_policy_from_env() if policy is None else policy
    if isinstance(resolved, bool):
        if not resolved:
            return None
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


def _default_policy_from_env() -> CpuPinning:
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
