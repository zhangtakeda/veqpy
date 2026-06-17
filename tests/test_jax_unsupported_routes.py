from __future__ import annotations

import importlib.util
import sys

import pytest
from helpers import tiny_grid, tiny_pf_problem
from jax_helpers import tiny_pf_rho_grid_problem

from veqpy.engine.backend import MissingOptionalBackendError, UnsupportedBackendFeature
from veqpy.layout.jax_binding import supported_jax_routes
from veqpy.operator import Operator


def test_jax_supported_route_list_is_explicit() -> None:
    assert supported_jax_routes() == ["PF/rho/grid"]


def test_jax_unsupported_route_fails_before_optional_import() -> None:
    sys.modules.pop("jax", None)

    with pytest.raises(UnsupportedBackendFeature) as exc_info:
        Operator(tiny_grid(), tiny_pf_problem(), backend="jax")

    message = str(exc_info.value)
    assert "backend='jax' does not support route='PF/psin/uniform'" in message
    assert "PF/rho/grid" in message
    assert "jax" not in sys.modules


def test_jax_supported_route_requires_optional_dependency_when_missing() -> None:
    if importlib.util.find_spec("jax") is not None:
        pytest.skip("JAX is installed in this environment")

    grid = tiny_grid()
    with pytest.raises(MissingOptionalBackendError):
        Operator(grid, tiny_pf_rho_grid_problem(grid), backend="jax")
