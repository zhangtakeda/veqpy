"""Public VEQPy Kernel error types."""

from __future__ import annotations


class TopologyError(ValueError):
    """Raised when a kernel topology cannot be canonicalized."""
