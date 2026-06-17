from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest
from helpers import tiny_grid, tiny_pf_problem
from jax_helpers import tiny_pf_rho_grid_problem

from veqpy.engine.backend import MissingOptionalBackendError, UnsupportedBackendFeature
from veqpy.operator import Operator


def test_public_imports_do_not_import_jax() -> None:
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


def test_unsupported_jax_route_fails_without_importing_jax() -> None:
    sys.modules.pop("jax", None)

    with pytest.raises(UnsupportedBackendFeature, match="backend='jax' does not support route"):
        Operator(tiny_grid(), tiny_pf_problem(), backend="jax")

    assert "jax" not in sys.modules


def test_missing_jax_dependency_error_for_supported_route() -> None:
    if importlib.util.find_spec("jax") is not None:
        pytest.skip("JAX is installed in this environment")

    grid = tiny_grid()
    with pytest.raises(MissingOptionalBackendError, match="requires the optional JAX dependency"):
        Operator(grid, tiny_pf_rho_grid_problem(grid), backend="jax")
