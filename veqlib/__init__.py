"""VEQlib package root.

Runtime/build helpers live under :mod:`veqlib.facade`, benchmark scripts live in
the top-level :mod:`benchmarks` package, and ``veqlib/core`` contains the
C++/CMake implementation.
"""

from __future__ import annotations

from .source_semantics import MaterializedSourceInputs, materialize_source_inputs

__all__ = [
    "MaterializedSourceInputs",
    "materialize_source_inputs",
]
