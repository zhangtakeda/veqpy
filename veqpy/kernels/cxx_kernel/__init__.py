"""
Package: veqpy.kernels.cxx_kernel

Role:
- Implement the private Cxx backend selected by KernelRecipe.backend.
- Own native artifact preparation, nanobind ABI lowering, solver lifecycle, and
  generated C++ core sources.

Public API:
- No package-root public symbols.

Dependencies:
- veqpy.kernels for Kernel dataclasses and ABI helpers.
- A local C++ toolchain and nanobind stack when native artifacts are built.

Downstream:
- veqpy.kernels.dispatch instantiates this backend for backend="cxx".
- Benchmarks and backend tests may import concrete modules directly.

Design notes:
- Python users should go through veqpy.Kernel or veqpy.api.
- C++ symbol names are internal to generated native artifacts.
"""

from __future__ import annotations

__all__: list[str] = []
