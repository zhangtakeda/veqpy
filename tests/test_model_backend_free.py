from __future__ import annotations

import inspect

import numpy as np
import pytest
from helpers import tiny_boundary, tiny_grid, tiny_operator, tiny_pf_problem

from veqpy.model import Boundary, Equilibrium, Geqdsk, Grid, Problem, Profile


def test_model_constructor_signatures_do_not_accept_backend() -> None:
    for model_type in (Problem, Grid, Profile, Boundary, Geqdsk, Equilibrium):
        assert "backend" not in inspect.signature(model_type).parameters


def test_model_objects_do_not_store_backend() -> None:
    operator = tiny_operator()
    equilibrium = operator.build_equilibrium(operator.zero_state())
    instances = (
        tiny_grid(),
        Profile(),
        tiny_boundary(),
        Geqdsk(),
        tiny_pf_problem(),
        equilibrium,
    )

    for instance in instances:
        assert not hasattr(instance, "backend")


def test_model_constructors_reject_backend_keyword() -> None:
    with pytest.raises(TypeError):
        Grid(Nr=8, Nt=8, backend="jax")
    with pytest.raises(TypeError):
        Profile(backend="jax")
    with pytest.raises(TypeError):
        Boundary(a=0.5, R0=1.0, Z0=0.0, B0=3.0, backend="jax")
    with pytest.raises(TypeError):
        Geqdsk(backend="jax")
    with pytest.raises(TypeError):
        Problem(
            route="PF",
            coordinate="rho",
            nodes="grid",
            active_profiles={"h": 1},
            boundary=tiny_boundary(),
            heat_input=np.ones(8, dtype=np.float64),
            current_input=np.ones(8, dtype=np.float64),
            backend="jax",
        )
