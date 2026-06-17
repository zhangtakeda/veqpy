"""
Compatibility shim for the historical Numba backend ABI module.

The concrete Numba ABI now lives in ``veqpy.engine.numba_abi``.  This module is
kept so existing tests, benchmarks, and downstream imports continue to work
during the backend split.
"""

from __future__ import annotations

from veqpy.engine.numba_abi import *  # noqa: F403
