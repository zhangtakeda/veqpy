from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from helpers import tiny_grid, tiny_operator, tiny_pf_problem
from numpy.testing import assert_allclose

from veqpy.engine.backend import (
    DEFAULT_BACKEND,
    InvalidBackendError,
    UnsupportedBackendFeature,
    normalize_backend,
)
from veqpy.operator import Operator


def test_backend_normalization_and_invalid_backend_error() -> None:
    assert normalize_backend(None) == DEFAULT_BACKEND
    assert normalize_backend("NUMBA") == "numba"
    assert normalize_backend(" jax ") == "jax"

    with pytest.raises(InvalidBackendError, match="Unsupported backend"):
        normalize_backend("cuda")


def test_operator_default_backend_is_numba() -> None:
    operator = tiny_operator()

    assert operator.backend == "numba"
    assert operator.backend == DEFAULT_BACKEND


def test_operator_numba_backend_matches_default_behavior() -> None:
    grid = tiny_grid()
    problem = tiny_pf_problem()
    default_operator = Operator(grid, problem)
    explicit_operator = Operator(grid, problem, backend="numba")
    x0 = default_operator.zero_state()

    assert explicit_operator.backend == "numba"
    assert explicit_operator.x_size == default_operator.x_size
    assert_allclose(explicit_operator.residual_var(x0), default_operator.residual_var(x0))


def test_operator_case_alias_accepts_numba_backend() -> None:
    problem = tiny_pf_problem()
    operator = Operator(tiny_grid(), case=problem, backend="numba")

    assert operator.problem is problem
    assert operator.case is problem
    assert operator.backend == "numba"


def test_operator_invalid_backend_raises_clear_error() -> None:
    with pytest.raises(InvalidBackendError, match="Unsupported backend"):
        Operator(tiny_grid(), tiny_pf_problem(), backend="not-a-backend")


def test_operator_jax_backend_is_explicitly_unsupported_in_phase1() -> None:
    with pytest.raises(UnsupportedBackendFeature, match="backend='jax'"):
        Operator(tiny_grid(), tiny_pf_problem(), backend="jax")


def test_default_and_numba_paths_do_not_import_jax() -> None:
    sys.modules.pop("jax", None)

    operator = Operator(tiny_grid(), tiny_pf_problem(), backend="numba")
    residual = operator.residual_var(np.zeros(operator.x_size, dtype=np.float64))

    assert np.all(np.isfinite(residual))
    assert "jax" not in sys.modules


def test_importing_public_packages_does_not_import_jax() -> None:
    script = (
        "import sys; "
        "import veqpy; "
        "import veqpy.model; "
        "import veqpy.operator; "
        "import veqpy.solver; "
        "print('jax' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )

    assert completed.stdout.strip() == "False"
