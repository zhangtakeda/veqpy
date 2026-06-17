"""
Private JAX backend configuration helpers.

This module does not import JAX at module import time.  JAX is imported only by
``require_jax`` after JAX-specific environment options have been applied.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType

from veqpy.engine.backend import JaxBackendOptions, MissingOptionalBackendError

_CONFIG_LOCKED = False
_LAST_OPTIONS: JaxBackendOptions | None = None


def apply_preimport_options(options: JaxBackendOptions | None) -> None:
    """Apply JAX options that must be set before importing JAX."""

    global _CONFIG_LOCKED, _LAST_OPTIONS
    if options is None:
        options = JaxBackendOptions()
    if "jax" in sys.modules:
        _CONFIG_LOCKED = True
        if _LAST_OPTIONS is not None and options != _LAST_OPTIONS:
            raise RuntimeError("JAX backend options cannot be changed after JAX is imported.")
        _LAST_OPTIONS = options
        return

    if options.platform:
        os.environ["JAX_PLATFORM_NAME"] = str(options.platform)
    if options.preallocate is not None:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(bool(options.preallocate)).lower()
    if options.mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(float(options.mem_fraction))
    if options.allocator:
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = str(options.allocator)
    if options.enable_x64:
        os.environ.setdefault("JAX_ENABLE_X64", "true")
    else:
        os.environ.setdefault("JAX_ENABLE_X64", "false")
    _LAST_OPTIONS = options


def require_jax(options: JaxBackendOptions | None = None) -> ModuleType:
    """Lazy import and return JAX, or raise a clear optional dependency error."""

    apply_preimport_options(options)
    try:
        jax = importlib.import_module("jax")
    except ModuleNotFoundError as exc:
        if exc.name != "jax":
            raise
        raise MissingOptionalBackendError(
            "backend='jax' requires the optional JAX dependency. "
            "Install VEQPy with the 'jax' extra or install JAX following upstream JAX docs."
        ) from exc
    if options is not None:
        try:
            jax.config.update("jax_enable_x64", bool(options.enable_x64))
        except Exception as exc:  # pragma: no cover - depends on installed JAX behavior
            raise RuntimeError("Failed to apply JAX x64 configuration.") from exc
    return jax
