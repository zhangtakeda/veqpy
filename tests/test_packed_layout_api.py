from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from veqpy.engine import numba_residual
from veqpy.model import Grid
from veqpy.operator.packed_layout import (
    PROFILE_STATIC_KWARGS,
    build_active_profile_metadata,
    build_fourier_profile_names,
    build_profile_index,
    build_profile_layout,
    build_profile_names,
    build_shape_profile_names,
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
    workspace.refresh_profile_slot(
        profile_id=0,
        grid_workspace=grid_workspace,
        offset=1.0,
        scale=1.0,
        power=0,
        envelope_power=PROFILE_STATIC_KWARGS["F"]["envelope_power"],
        amplitude_power=PROFILE_STATIC_KWARGS["F"]["amplitude_power"],
        coeff=np.array([1.0], dtype=np.float64),
    )
    f_fields = workspace.fields_for("F")

    assert_allclose(f_fields[0, -1], 1.0)
    assert abs(f_fields[1, -1]) > 1.0e-12


def test_f_profile_amplitude_power_matches_f_squared_chain_rule() -> None:
    grid_workspace = GridWorkspace.from_grid(
        Grid(Nr=6, Nt=4, M_max=1, L_max=2, quadrature_scheme="uniform")
    )
    coeff = np.array([0.25, -0.1], dtype=np.float64)
    scale = 3.0

    workspace = ProfileWorkspace(
        nr=grid_workspace.Nr,
        m_max=grid_workspace.M_max,
        profile_names=("F",),
        profile_index={"F": 0},
        active_profile_ids=np.array([0], dtype=np.int64),
        profile_L=np.array([1], dtype=np.int64),
    )
    raw_workspace = ProfileWorkspace(
        nr=grid_workspace.Nr,
        m_max=grid_workspace.M_max,
        profile_names=("F",),
        profile_index={"F": 0},
        active_profile_ids=np.array([0], dtype=np.int64),
        profile_L=np.array([1], dtype=np.int64),
    )
    workspace.refresh_profile_slot(
        profile_id=0,
        grid_workspace=grid_workspace,
        offset=1.0,
        scale=scale,
        power=0,
        envelope_power=PROFILE_STATIC_KWARGS["F"]["envelope_power"],
        amplitude_power=PROFILE_STATIC_KWARGS["F"]["amplitude_power"],
        coeff=coeff,
    )
    raw_workspace.refresh_profile_slot(
        profile_id=0,
        grid_workspace=grid_workspace,
        offset=1.0,
        scale=1.0,
        power=0,
        envelope_power=PROFILE_STATIC_KWARGS["F"]["envelope_power"],
        amplitude_power=1.0,
        coeff=coeff,
    )

    f_fields = workspace.fields_for("F")
    amplitude_fields = raw_workspace.fields_for("F")
    amplitude = amplitude_fields[0]
    sqrt_amplitude = np.sqrt(amplitude)
    expected = np.empty_like(f_fields)
    expected[0] = scale * sqrt_amplitude
    expected[1] = scale * 0.5 * amplitude_fields[1] / sqrt_amplitude
    expected[2] = scale * (
        0.5 * amplitude_fields[2] / sqrt_amplitude
        - 0.25 * amplitude_fields[1] * amplitude_fields[1] / (amplitude * sqrt_amplitude)
    )

    assert_allclose(f_fields, expected)


def test_degree_first_layout_encode_decode_and_active_metadata() -> None:
    profile_names = build_profile_names(2)
    profile_index = build_profile_index(profile_names)
    active_profiles = {"h": 2, "k": 3, "s1": 1}
    coefficients = {
        "h": np.array([1.0, 2.0], dtype=np.float64),
        "k": np.zeros(3, dtype=np.float64),
        "s1": np.array([4.0], dtype=np.float64),
    }
    profile_L, coeff_index, order_offsets = build_profile_layout(
        active_profiles,
        profile_names=profile_names,
    )

    assert profile_L[profile_index["h"]] == 1
    assert profile_L[profile_index["k"]] == 2
    assert profile_L[profile_index["s1"]] == 0
    assert packed_size(coeff_index) == 6
    assert order_offsets.tolist() == [0, 3, 5, 6]

    x = encode_packed_state(
        coefficients,
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
        build_profile_layout(
            {"unknown": 1},
            profile_names=profile_names,
        )
    with pytest.raises(ValueError, match="At least one active profile"):
        build_profile_layout({}, profile_names=profile_names)
    with pytest.raises(TypeError, match="length must be int"):
        build_profile_layout({"h": True}, profile_names=profile_names)
    with pytest.raises(ValueError, match="length must be positive"):
        build_profile_layout({"h": 0}, profile_names=profile_names)
    with pytest.raises(ValueError, match="shape"):
        validate_packed_state(np.zeros(2), np.array([[0, -1, -1]], dtype=np.int64))


def test_residual_auto_packer_matches_legacy_high_block_path() -> None:
    rng = np.random.default_rng(20240611)
    nr = 5
    nt = 4
    block_count = 8
    residual_workspace = np.ascontiguousarray(rng.normal(size=(4, nr, nt)))
    scratch_legacy = np.empty(nr, dtype=np.float64)
    scratch_auto = np.empty(nr, dtype=np.float64)
    scratch_rows = np.empty((block_count + 5, nr), dtype=np.float64)
    block_codes = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    block_orders = np.array([0, 0, 0, 0, 1, 2, 0, 0], dtype=np.int64)
    block_radial_powers = np.array([0, 0, 0, 0, 1, 1, 0, 0], dtype=np.int64)
    coeff_index_rows = np.arange(block_count, dtype=np.int64).reshape(block_count, 1)
    lengths = np.ones(block_count, dtype=np.int64)
    grid_workspace = GridWorkspace.from_grid(
        Grid(Nr=nr, Nt=nt, L_max=0, M_max=3, K_max=2, quadrature_scheme="legendre")
    )
    weights = grid_workspace.weights
    out_legacy = np.zeros(block_count, dtype=np.float64)
    out_auto = np.zeros(block_count, dtype=np.float64)

    numba_residual.run_residual_blocks_packed_precomputed(
        out_legacy,
        scratch_legacy,
        block_codes,
        block_orders,
        block_radial_powers,
        coeff_index_rows,
        lengths,
        residual_workspace,
        grid_workspace.radial_fields,
        grid_workspace.poloidal_fields,
        grid_workspace.K_max,
        grid_workspace.L_max,
        weights,
        0.4,
        1.7,
        2.1,
    )
    numba_residual.run_residual_blocks_packed_precomputed_auto(
        out_auto,
        scratch_auto,
        scratch_rows,
        block_codes,
        block_orders,
        block_radial_powers,
        coeff_index_rows,
        lengths,
        residual_workspace,
        grid_workspace.radial_fields,
        grid_workspace.poloidal_fields,
        grid_workspace.K_max,
        grid_workspace.L_max,
        weights,
        0.4,
        1.7,
        2.1,
    )

    assert_allclose(out_auto, out_legacy, rtol=0.0, atol=1.0e-14)
