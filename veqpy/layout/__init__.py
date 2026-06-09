"""
Module: layout.__init__

Role:
- Mark ``veqpy.layout`` as the executable layout package.

Public API:
- No broad package-root re-exports.

Notes:
- Import executable layout types from ``veqpy.layout.runtime``.
- Import binders from their concrete ``veqpy.layout.*_binding`` modules.
- Package roots are the only modules that declare ``__all__``.
"""

from __future__ import annotations

__all__: list[str] = []
