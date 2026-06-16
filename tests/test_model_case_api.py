from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from helpers import MU0, profiles, tiny_boundary, tiny_grid
from numpy.testing import assert_allclose

from veqpy.model import Boundary, Problem, Profile
from veqpy.operator import Operator


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
    profile = Profile(scale=2, power=3.0, envelope_power=1.0, coeff=coeff)

    assert profile.scale == 2.0
    assert profile.power == 3
    assert profile.amplitude_power == 1.0
    assert profile.offset == 0.0
    assert profile.coeff is not coeff
    assert not np.shares_memory(profile.coeff, coeff)
    assert not profile.coeff.flags.writeable
    profile.check()

    clone = profile.copy()
    assert clone.coeff is not profile.coeff
    assert not np.shares_memory(clone.coeff, profile.coeff)
    assert not clone.coeff.flags.writeable

    with pytest.raises(FrozenInstanceError):
        profile.scale = 3.0

    with pytest.raises(TypeError):
        Profile(amplitude_power=None)
    assert_allclose(clone.coeff, profile.coeff)

    with pytest.raises(TypeError, match="offset"):
        Profile(offset=None)

    with pytest.raises(ValueError, match="coeff must be 1D"):
        Profile(coeff=np.ones((1, 1)))
    with pytest.raises(ValueError, match="coeff must be non-empty"):
        Profile(coeff=np.array([], dtype=np.float64))


def test_problem_is_the_public_problem_definition() -> None:
    problem = Problem(
        route="pf",
        coordinate="RHO",
        profiles=profiles({"h": 2}),
        boundary=tiny_boundary(),
        heat_input=np.full(3, 1.0e6, dtype=np.float64),
        current_input=np.ones(3, dtype=np.float64),
    )

    assert isinstance(problem, Problem)
    assert problem.route == "PF"
    assert problem.coordinate == "rho"
    assert problem.nodes == "uniform"


def test_problem_keeps_raw_inputs_and_copy_is_detached() -> None:
    heat_input = np.array([1.0e6, 1.2e6, 1.4e6], dtype=np.float64)
    current_input = np.array([0.0, 2.0e6, 3.0e6], dtype=np.float64)
    case = Problem(
        route="pi",
        coordinate="RHO",
        nodes="UNIFORM",
        profiles=profiles({"h": 2}),
        boundary=tiny_boundary(),
        heat_input=heat_input,
        current_input=current_input,
        Ip=3.0e6,
    )

    assert case.route == "PI"
    assert case.coordinate == "rho"
    assert case.nodes == "uniform"
    assert_allclose(case.heat_input, heat_input)
    assert_allclose(case.current_input, current_input)
    assert_allclose(case.Ip, 3.0e6)
    assert not case.heat_input.flags.writeable
    assert not case.current_input.flags.writeable

    heat_input[:] = -1.0
    current_input[:] = -2.0
    assert_allclose(case.heat_input, [1.0e6, 1.2e6, 1.4e6])
    assert_allclose(case.current_input, [0.0, 2.0e6, 3.0e6])

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        clone = case.copy()
    assert not captured
    assert clone is not case
    assert_allclose(clone.current_input, case.current_input)
    assert not np.shares_memory(clone.current_input, case.current_input)
    assert not clone.current_input.flags.writeable

    operator = Operator(tiny_grid(), case)
    source_plan = operator.plan.source_plan
    assert_allclose(source_plan.scaled_heat, case.heat_input * MU0)
    assert_allclose(source_plan.scaled_current, case.current_input * MU0)
    assert_allclose(source_plan.scaled_Ip, case.Ip * MU0)


def test_source_plan_rejects_ambiguous_setup_magnitudes() -> None:
    case = Problem(
        route="PF",
        coordinate="rho",
        profiles=profiles({"h": 2}),
        boundary=tiny_boundary(),
        heat_input=np.full(3, 1.0e6, dtype=np.float64),
        current_input=np.ones(3, dtype=np.float64),
        Ip=MU0 * 3.0e6,
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="Rejected setup input magnitude"):
            Operator(tiny_grid(), case)

    assert any("Pass unnormalized setup values" in str(item.message) for item in captured)


def test_boundary_and_problem_json_roundtrip(tmp_path) -> None:
    boundary = Boundary(
        a=0.5,
        R0=1.2,
        Z0=-0.1,
        B0=2.5,
        ka=1.6,
        c_offsets=np.array([0.1, 0.2], dtype=np.float64),
        s_offsets=np.array([0.0, -0.2], dtype=np.float64),
    )
    boundary_path = tmp_path / "boundary.json"
    boundary.write(str(boundary_path))
    loaded_boundary = Boundary.load(str(boundary_path))
    assert loaded_boundary.a == boundary.a
    assert loaded_boundary.R0 == boundary.R0
    assert loaded_boundary.Z0 == boundary.Z0
    assert loaded_boundary.B0 == boundary.B0
    assert loaded_boundary.ka == boundary.ka
    assert_allclose(loaded_boundary.c_offsets, boundary.c_offsets)
    assert_allclose(loaded_boundary.s_offsets, boundary.s_offsets)

    problem = Problem(
        route="pf",
        coordinate="rho",
        profiles=profiles({"h": np.array([0.0, 1.0], dtype=np.float64), "k": None}),
        boundary=boundary,
        heat_input=np.full(3, 1.0e6, dtype=np.float64),
        current_input=np.ones(3, dtype=np.float64),
        Ip=3.0e6,
    )
    problem_path = tmp_path / "problem.json"
    problem.write(str(problem_path))
    loaded_problem = Problem.load(str(problem_path))

    assert loaded_problem.route == "PF"
    assert loaded_problem.boundary.a == boundary.a
    assert_allclose(loaded_problem.boundary.c_offsets, boundary.c_offsets)
    assert_allclose(loaded_problem.profiles["h"].coeff, [0.0, 1.0])
    assert loaded_problem.profiles["k"].coeff is None
    assert_allclose(loaded_problem.heat_input, problem.heat_input)
    assert_allclose(loaded_problem.current_input, problem.current_input)
    assert_allclose(loaded_problem.Ip, 3.0e6)
    assert not loaded_problem.heat_input.flags.writeable
