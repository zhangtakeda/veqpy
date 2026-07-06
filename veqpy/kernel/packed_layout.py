"""Temporary bridge to the VEQlib-owned Numba packed-layout module."""

from __future__ import annotations

import sys
from importlib import import_module

sys.modules[__name__] = import_module("veqlib.numba_core.packed_layout")
