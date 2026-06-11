from __future__ import annotations

from veqpy.engine import numba_source


def test_default_source_routes_are_registered() -> None:
    assert frozenset(numba_source.ROUTE_REGISTRY) == numba_source.SOURCE_ROUTE_KEY_SET
    assert frozenset(numba_source.SOURCE_ROUTE_KERNELS.registry) == (
        numba_source.SOURCE_ROUTE_KEY_SET
    )

    for route_key in numba_source.SOURCE_ROUTE_KEYS:
        route_spec = numba_source.validate_route(*route_key)
        assert route_spec.route == route_key[0]
        assert route_spec.coordinate == route_key[1]
        assert route_spec.nodes == route_key[2]
        assert route_spec.implementation is numba_source.SOURCE_ROUTE_KERNELS.registry[route_key]


def test_default_source_route_kernel_sharing_is_stable() -> None:
    implementation_count = len(
        {id(route_spec.implementation) for route_spec in numba_source.ROUTE_REGISTRY.values()}
    )

    assert implementation_count == 18
