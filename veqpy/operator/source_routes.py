"""
Module: operator.source_routes

Role:
- Hold backend-neutral source route semantics.
- Normalize route keys and expose route metadata without concrete kernel callables.

Notes:
- This module intentionally does not import Numba or JAX.
"""

from __future__ import annotations

from dataclasses import dataclass

RouteKey = tuple[str, str, str]

RHO_COORDINATE_NAME = "rho"
PSIN_COORDINATE_NAME = "psin"
UNIFORM_NODES = "uniform"
GRID_NODES = "grid"

COORDINATE_NAMES: tuple[str, ...] = (RHO_COORDINATE_NAME, PSIN_COORDINATE_NAME)
NODE_NAMES: tuple[str, ...] = (UNIFORM_NODES, GRID_NODES)

SOURCE_PARAMETERIZATION_IDENTITY = "identity"
SOURCE_PARAMETERIZATION_SQRT_PSIN = "sqrt_psin"

SOURCE_ROUTE_KEYS: tuple[RouteKey, ...] = (
    ("PF", "rho", "uniform"),
    ("PF", "rho", "grid"),
    ("PF", "psin", "uniform"),
    ("PF", "psin", "grid"),
    ("PP", "rho", "uniform"),
    ("PP", "rho", "grid"),
    ("PP", "psin", "uniform"),
    ("PP", "psin", "grid"),
    ("PI", "rho", "uniform"),
    ("PI", "rho", "grid"),
    ("PI", "psin", "uniform"),
    ("PI", "psin", "grid"),
    ("PJ1", "rho", "uniform"),
    ("PJ1", "rho", "grid"),
    ("PJ1", "psin", "uniform"),
    ("PJ1", "psin", "grid"),
    ("PJ2", "rho", "uniform"),
    ("PJ2", "rho", "grid"),
    ("PJ2", "psin", "uniform"),
    ("PJ2", "psin", "grid"),
    ("PQ", "rho", "uniform"),
    ("PQ", "rho", "grid"),
    ("PQ", "psin", "uniform"),
    ("PQ", "psin", "grid"),
)
SOURCE_ROUTE_KEY_SET: frozenset[RouteKey] = frozenset(SOURCE_ROUTE_KEYS)


@dataclass(frozen=True, slots=True)
class SourceRouteMetadata:
    """Backend-neutral source route metadata."""

    route: str
    coordinate: str
    nodes: str
    parameterization: str

    @property
    def route_key(self) -> RouteKey:
        return (self.route, self.coordinate, self.nodes)


def normalize_route_key(value: RouteKey) -> RouteKey:
    """Normalize a source route key to canonical spelling."""

    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("Source route key must be a three-string tuple: (route, coordinate, nodes)")
    route, coordinate, nodes = value
    if not isinstance(route, str) or not isinstance(coordinate, str) or not isinstance(nodes, str):
        raise TypeError(
            "Source route key must contain strings only: "
            f"got {type(route).__name__}, {type(coordinate).__name__}, {type(nodes).__name__}"
        )
    route_name = route.upper()
    coordinate_name = coordinate.lower()
    nodes_name = nodes.lower()
    if coordinate_name not in COORDINATE_NAMES:
        raise ValueError(f"Unsupported coordinate {coordinate!r}")
    if nodes_name not in NODE_NAMES:
        raise ValueError(f"Unsupported nodes {nodes!r}")
    return (route_name, coordinate_name, nodes_name)


def validate_route_metadata(
    route: str,
    coordinate: str,
    nodes: str = UNIFORM_NODES,
) -> SourceRouteMetadata:
    """Validate and return backend-neutral metadata for a source route."""

    route_key = normalize_route_key((route, coordinate, nodes))
    if route_key not in SOURCE_ROUTE_KEY_SET:
        supported = ", ".join("/".join(item) for item in sorted(SOURCE_ROUTE_KEY_SET))
        raise KeyError(
            f"Unknown source route {route!r}/{coordinate!r}/{nodes!r}; supported: {supported}"
        )
    return SourceRouteMetadata(
        route=route_key[0],
        coordinate=route_key[1],
        nodes=route_key[2],
        parameterization=source_parameterization_for_route_key(route_key),
    )


def source_parameterization_for_route_key(route_key: RouteKey) -> str:
    """Return the backend-neutral source-input parameterization for a route."""

    normalized_key = normalize_route_key(route_key)
    if normalized_key not in SOURCE_ROUTE_KEY_SET:
        supported = ", ".join("/".join(item) for item in sorted(SOURCE_ROUTE_KEY_SET))
        raise KeyError(f"Unknown source route {normalized_key!r}; supported: {supported}")
    if normalized_key == ("PP", "psin", "uniform"):
        return SOURCE_PARAMETERIZATION_SQRT_PSIN
    return SOURCE_PARAMETERIZATION_IDENTITY
