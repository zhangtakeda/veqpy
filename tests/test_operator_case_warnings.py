from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from veqpy.model import Boundary
from veqpy.operator import OperatorCase

MU0 = 4.0e-7 * np.pi


def _boundary() -> Boundary:
    return Boundary(a=0.5, R0=1.0, Z0=0.0, B0=3.0)


def test_physical_ip_pressure_and_current_inputs_are_mu0_scaled() -> None:
    heat_input = np.array([1.0e6, 1.2e6, 1.4e6], dtype=np.float64)
    current_input = np.array([0.0, 2.0e6, 3.0e6], dtype=np.float64)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        case = OperatorCase(
            route="PI",
            coordinate="rho",
            profile_coeffs={"h": 2},
            boundary=_boundary(),
            heat_input=heat_input,
            current_input=current_input,
            Ip=3.0e6,
        )

    assert_allclose(case.heat_input, heat_input * MU0)
    assert_allclose(case.current_input, current_input * MU0)
    assert_allclose(case.Ip, 3.0e6 * MU0)
    assert not captured


def test_physical_ip_in_normal_setup_range_does_not_warn() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        case = OperatorCase(
            route="PF",
            coordinate="rho",
            profile_coeffs={"h": 2},
            boundary=_boundary(),
            heat_input=np.full(3, 1.0e6, dtype=np.float64),
            current_input=np.ones(3, dtype=np.float64),
            Ip=3.0e6,
        )

    assert_allclose(case.Ip, MU0 * 3.0e6)
    assert_allclose(case.heat_input, MU0 * np.full(3, 1.0e6, dtype=np.float64))
    assert_allclose(case.current_input, np.ones(3, dtype=np.float64))
    assert not captured


def test_mu0_scaled_ip_setup_input_is_rejected() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="Rejected setup input magnitude"):
            OperatorCase(
                route="PF",
                coordinate="rho",
                profile_coeffs={"h": 2},
                boundary=_boundary(),
                heat_input=np.full(3, 1.0e6, dtype=np.float64),
                current_input=np.ones(3, dtype=np.float64),
                Ip=MU0 * 3.0e6,
            )

    messages = [str(item.message) for item in captured]
    assert any(
        "Rejected setup input magnitude" in message and "Ip abs=" in message
        for message in messages
    )


def test_non_current_profile_outside_setup_range_is_rejected() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="Rejected setup input magnitude"):
            OperatorCase(
                route="PF",
                coordinate="rho",
                profile_coeffs={"h": 2},
                boundary=_boundary(),
                heat_input=np.full(3, 1.0e6, dtype=np.float64),
                current_input=np.full(3, 1.0e6, dtype=np.float64),
                Ip=3.0e6,
            )

    messages = [str(item.message) for item in captured]
    assert any("current_input max_abs=" in message for message in messages)
