"""
Module: veqpy.kernels.abi.enums

Role:
- Define shared integer ABI codes used by Kernel backends.
"""

from __future__ import annotations

SOURCE_ROUTE_CODES = {
    "PF": 1,
    "PP": 2,
    "PI": 3,
    "PJ1": 4,
    "PJ2": 5,
    "PQ": 6,
}
SOURCE_COORDINATE_CODES = {"rho": 1, "psin": 2}
SOURCE_CONSTRAINT_CODES_BY_FLAGS = {
    (False, False): 0,
    (True, False): 1,
    (False, True): 2,
    (True, True): 3,
}
SOURCE_CONSTRAINT_FLAGS_BY_NAME = {
    "ip": (True, False),
    "beta": (False, True),
    "both": (True, True),
    "none": (False, False),
}
SOURCE_CONSTRAINT_LABELS_BY_FLAGS = {
    (False, False): "null",
    (True, False): "Ip",
    (False, True): "beta",
    (True, True): "Ip_beta",
}
SOURCE_CONSTRAINT_FLAG_ORDER = ((True, True), (True, False), (False, True), (False, False))
SOURCE_NODES_CODES = {"uniform": 1, "grid": 2}
SOURCE_ACTIVE_FAMILY_CODES = {"none": 0, "psin": 1, "F": 2}
SOURCE_PARAMETERIZATION_CODES = {"identity": 0, "sqrt_psin": 1}
SOURCE_CONSTRAINT_FLAGS_BY_ROUTE = {
    "PF": frozenset({(False, False), (True, False), (False, True)}),
    "PP": frozenset(SOURCE_CONSTRAINT_CODES_BY_FLAGS),
    "PI": frozenset(SOURCE_CONSTRAINT_CODES_BY_FLAGS),
    "PJ1": frozenset(SOURCE_CONSTRAINT_CODES_BY_FLAGS),
    "PJ2": frozenset(SOURCE_CONSTRAINT_CODES_BY_FLAGS),
    "PQ": frozenset(SOURCE_CONSTRAINT_CODES_BY_FLAGS),
}
LAYOUT_CODES = {"degree": 0, "family": 1}
SUPPORTED_BACKENDS = frozenset({"cxx", "numba"})
