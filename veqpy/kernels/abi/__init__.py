"""
Package: veqpy.kernels.abi

Role:
- Store backend-neutral integer codes, option normalization, identity payloads,
  and source lowering semantics.
- Keep shared Kernel ABI rules out of concrete backend implementations.

Public API:
- Concrete modules provide enums, options, identity, and source_semantics helpers.

Dependencies:
- veqpy.kernels public dataclasses for source-lowering inputs where needed.

Downstream:
- veqpy.kernels.types uses ABI metadata when canonicalizing Kernel dataclasses.
- Cxx and Numba backend modules use ABI helpers for matching route semantics.

Design notes:
- This package contains shared contracts, not backend runtime state.
- The package root documents ownership; concrete symbols live in concrete modules.
"""

from __future__ import annotations

__all__: list[str] = []
