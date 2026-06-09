from __future__ import annotations

import numpy as np
import pytest
from helpers import tiny_operator
from numpy.testing import assert_allclose

from veqpy.engine.numba_source import (
    DEFAULT_LOCAL_BARYCENTRIC_STENCIL as ENGINE_LOCAL_BARYCENTRIC_STENCIL,
)
from veqpy.math import (
    DEFAULT_LOCAL_BARYCENTRIC_STENCIL,
    SOURCE_INTERP_DEFAULT,
    normalize_source_interpolation_kind,
)
from veqpy.operator import Operator


def test_operator_residual_interfaces_and_in_place_outputs() -> None:
    operator = tiny_operator()
    x0 = operator.encode_initial_state()

    assert x0.shape == (operator.x_size,)
    assert operator.active_profile_ids.ndim == 1
    assert "h" in operator.profile_names

    residual = operator.residual_var(x0)
    assert residual.shape == (operator.x_size,)
    assert np.all(np.isfinite(residual))

    out = np.empty_like(residual)
    operator.residual_var_into(x0, out)
    assert_allclose(out, residual)

    collocation = operator.residual_collocation(x0)
    assert collocation.shape == (
        operator.plan.grid_workspace.Nr * operator.plan.grid_workspace.Nt,
    )
    assert np.all(np.isfinite(collocation))

    residual_stage = operator.stage_d_residual()
    assert residual_stage.shape == (operator.x_size,)


def test_source_interpolation_default_is_shared() -> None:
    assert normalize_source_interpolation_kind(None) == SOURCE_INTERP_DEFAULT
    source_interpolation_field = Operator.__dataclass_fields__["source_interpolation_kind"]
    assert source_interpolation_field.default == SOURCE_INTERP_DEFAULT
    assert ENGINE_LOCAL_BARYCENTRIC_STENCIL == DEFAULT_LOCAL_BARYCENTRIC_STENCIL


def test_operator_validation_and_snapshot_helpers() -> None:
    operator = tiny_operator()
    x0 = operator.encode_initial_state()

    with pytest.raises(ValueError, match="Expected x to have shape"):
        operator.coerce_x(np.zeros(operator.x_size + 1))
    with pytest.raises(TypeError, match="dtype float64"):
        operator.residual_var_into(x0, np.empty(operator.x_size, dtype=np.float32))

    coeffs = operator.build_coeffs(x0, include_none=False)
    assert set(coeffs) == {"h", "k", "s1", "psin"}

    equilibrium = operator.build_equilibrium(x0)
    assert equilibrium.grid.Nr == operator.plan.grid_workspace.Nr
    assert np.isfinite(equilibrium.Ip)
