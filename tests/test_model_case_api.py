from __future__ import annotations

import warnings

import numpy as np
import pytest
from helpers import MU0, tiny_boundary
from numpy.testing import assert_allclose

from veqpy.model import Boundary, Profile
from veqpy.operator import OperatorCase


def test_boundary_normalizes_offsets_and_forces_s0_to_zero() -> None:
    boundary = Boundary(
        a=1,
        R0=2,
        Z0=0,
        B0=3,
        ka=1.5,
        c_offsets=[0.1, 0.2],
        s_offsets=[9.0, 0.3],
    )

    assert boundary.a == 1.0
    assert_allclose(boundary.c_offsets, [0.1, 0.2])
    assert_allclose(boundary.s_offsets, [0.0, 0.3])

    with pytest.raises(ValueError, match="c_offsets must be 1D"):
        Boundary(a=1.0, R0=1.0, Z0=0.0, B0=3.0, c_offsets=[[1.0]])
    with pytest.raises(ValueError, match="s_offsets must have at least one entry"):
        Boundary(a=1.0, R0=1.0, Z0=0.0, B0=3.0, s_offsets=[])


def test_profile_normalization_validation_and_copy_independence() -> None:
    coeff = np.array([1.0, 2.0])
    profile = Profile(scale=2, power=3.0, envelope_power=1.0, offset=None, coeff=coeff)

    assert profile.scale == 2.0
    assert profile.power == 3
    assert profile.offset == 0.0
    assert profile.coeff is coeff
    profile.check()

    clone = profile.copy()
    assert clone.coeff is not profile.coeff
    assert not np.shares_memory(clone.coeff, profile.coeff)
    assert not clone.coeff.flags.writeable
    assert_allclose(clone.coeff, profile.coeff)

    with pytest.raises(ValueError, match="coeff must be 1D"):
        Profile(coeff=np.ones((1, 1)))


def test_operator_case_setup_normalizes_then_copy_preserves_internal_state() -> None:
    heat_input = np.array([1.0e6, 1.2e6, 1.4e6], dtype=np.float64)
    current_input = np.array([0.0, 2.0e6, 3.0e6], dtype=np.float64)
    case = OperatorCase(
        route="pi",
        coordinate="RHO",
        nodes="UNIFORM",
        profile_coeffs={"h": 2},
        boundary=tiny_boundary(),
        heat_input=heat_input,
        current_input=current_input,
        Ip=3.0e6,
    )

    assert case.route == "PI"
    assert case.coordinate == "rho"
    assert case.nodes == "uniform"
    assert_allclose(case.heat_input, heat_input * MU0)
    assert_allclose(case.current_input, current_input * MU0)
    assert_allclose(case.Ip, 3.0e6 * MU0)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        clone = case.copy()
    assert not captured
    assert clone is not case
    assert_allclose(clone.current_input, case.current_input)
    clone.current_input[1] = -1.0
    assert case.current_input[1] != -1.0


def test_operator_case_rejects_ambiguous_setup_magnitudes() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="Rejected setup input magnitude"):
            OperatorCase(
                route="PF",
                coordinate="rho",
                profile_coeffs={"h": 2},
                boundary=tiny_boundary(),
                heat_input=np.full(3, 1.0e6, dtype=np.float64),
                current_input=np.ones(3, dtype=np.float64),
                Ip=MU0 * 3.0e6,
            )

    assert any("Pass unnormalized setup values" in str(item.message) for item in captured)
