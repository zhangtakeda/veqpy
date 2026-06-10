from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from veqpy.engine.numba_operator import convert_f_squared_fields_to_f
from veqpy.model import Grid
from veqpy.model.profile import Profile
from veqpy.operator.packed_layout import (
    PROFILE_STATIC_KWARGS,
    build_active_profile_metadata,
    build_fourier_profile_names,
    build_profile_index,
    build_profile_layout,
    build_profile_names,
    build_shape_profile_names,
    coeff_array_from_list,
    decode_packed_blocks,
    encode_packed_state,
    get_prefix_profile_names,
    packed_size,
    validate_packed_state,
)
from veqpy.workspace.grid_workspace import GridWorkspace
from veqpy.workspace.profile_workspace import ProfileWorkspace


def test_profile_name_order_is_stable() -> None:
    assert get_prefix_profile_names() == ("psin", "F")
    assert build_fourier_profile_names(2) == ("c0", "c1", "c2", "s1", "s2")
    assert build_shape_profile_names(2) == ("h", "v", "k", "c0", "c1", "c2", "s1", "s2")
    assert build_profile_names(2) == (
        "h",
        "v",
        "k",
        "c0",
        "c1",
        "c2",
        "s1",
        "s2",
        "psin",
        "F",
    )


def test_f_profile_keeps_edge_value_but_allows_edge_slope() -> None:
    grid_workspace = GridWorkspace.from_grid(
        Grid(Nr=5, Nt=4, M_max=1, L_max=1, quadrature_scheme="uniform")
    )
    workspace = ProfileWorkspace(
        nr=grid_workspace.Nr,
        m_max=grid_workspace.M_max,
        profile_names=("F",),
        profile_index={"F": 0},
        active_profile_ids=np.array([0], dtype=np.int64),
        profile_L=np.array([0], dtype=np.int64),
    )
    profile = Profile(
        scale=1.0,
        offset=1.0,
        coeff=np.array([1.0], dtype=np.float64),
        envelope_power=PROFILE_STATIC_KWARGS["F"]["envelope_power"],
    )

    workspace.refresh_profile_slot(profile_id=0, profile=profile, grid_workspace=grid_workspace)
    f_fields = workspace.fields_for("F")
    convert_f_squared_fields_to_f(f_fields)

    assert_allclose(f_fields[0, -1], 1.0)
    assert abs(f_fields[1, -1]) > 1.0e-12


def test_degree_first_layout_encode_decode_and_active_metadata() -> None:
    profile_names = build_profile_names(2)
    profile_index = build_profile_index(profile_names)
    profile_L, coeff_index, order_offsets = build_profile_layout(
        {"h": [1.0, 2.0], "k": 3, "s1": [4.0]},
        profile_names=profile_names,
    )

    assert profile_L[profile_index["h"]] == 1
    assert profile_L[profile_index["k"]] == 2
    assert profile_L[profile_index["s1"]] == 0
    assert packed_size(coeff_index) == 6
    assert order_offsets.tolist() == [0, 3, 5, 6]

    x = encode_packed_state(
        {"h": [1.0, 2.0], "k": 3, "s1": [4.0]},
        profile_L,
        coeff_index,
        profile_names=profile_names,
    )
    assert_allclose(x, [1.0, 0.0, 4.0, 2.0, 0.0, 0.0])

    blocks = decode_packed_blocks(x, profile_L, coeff_index, profile_names=profile_names)
    assert_allclose(blocks[profile_index["h"]], [1.0, 2.0])
    assert_allclose(blocks[profile_index["k"]], [0.0, 0.0, 0.0])
    assert blocks[profile_index["v"]] is None

    active_mask, active_ids = build_active_profile_metadata(profile_L, profile_names=profile_names)
    assert active_mask.dtype == np.bool_
    assert active_ids.tolist() == [profile_index["h"], profile_index["k"], profile_index["s1"]]


def test_packed_layout_validation_errors() -> None:
    profile_names = build_profile_names(1)
    with pytest.raises(KeyError, match="Unknown profile names"):
        build_profile_layout({"unknown": [1.0]}, profile_names=profile_names)
    with pytest.raises(ValueError, match="At least one active profile"):
        build_profile_layout({"h": None}, profile_names=profile_names)
    with pytest.raises(TypeError, match="length indicator"):
        coeff_array_from_list("h", True)
    with pytest.raises(ValueError, match="shape"):
        validate_packed_state(np.zeros(2), np.array([[0, -1, -1]], dtype=np.int64))
