from __future__ import annotations

import numpy as np
import pytest
from helpers import tiny_boundary, tiny_grid, tiny_operator

from veqpy.engine import validate_route
from veqpy.engine.numba_abi import NumbaSourceBindingPlan
from veqpy.model import Problem
from veqpy.operator import Operator
from veqpy.operator.source_execution import SourceExecutionPlan
from veqpy.operator.source_plan import SourcePlan
from veqpy.operator.source_routes import validate_route_metadata


def test_source_plan_materializes_inputs_without_storing_numba_kernel() -> None:
    operator = tiny_operator()
    source_plan = operator.plan.source_plan

    assert isinstance(source_plan, SourcePlan)
    assert isinstance(operator.plan.source_execution, SourceExecutionPlan)
    assert "kernel" not in SourcePlan.__dataclass_fields__
    assert source_plan.route_key == ("PF", "psin", "uniform")
    assert source_plan.scaled_heat.flags.writeable is False
    assert source_plan.scaled_current.flags.writeable is False


def test_numba_source_binding_adapter_retrieves_kernel_callable() -> None:
    operator = tiny_operator()
    source_plan = operator.plan.source_plan
    binding = NumbaSourceBindingPlan.from_source_plan(source_plan)
    route_spec = validate_route(*source_plan.route_key)

    assert binding.route_key == source_plan.route_key
    assert binding.kernel is route_spec.implementation
    assert source_plan.kernel is binding.kernel
    assert source_plan.coordinate_code == binding.coordinate_code
    assert source_plan.parameterization_code == binding.parameterization_code


def test_backend_neutral_route_metadata_has_no_kernel_callable() -> None:
    metadata = validate_route_metadata("pf", "rho", "grid")

    assert metadata.route_key == ("PF", "rho", "grid")
    assert metadata.parameterization == "identity"
    assert not hasattr(metadata, "implementation")
    assert not hasattr(metadata, "kernel")


def test_engine_validate_route_compatibility_remains() -> None:
    route_spec = validate_route("pf", "rho", "grid")

    assert route_spec.route == "PF"
    assert route_spec.coordinate == "rho"
    assert route_spec.nodes == "grid"
    assert callable(route_spec.implementation)


def test_source_ownership_rejection_paths_keep_error_meaning() -> None:
    grid = tiny_grid()
    base_kwargs = {
        "route": "PF",
        "coordinate": "rho",
        "nodes": "grid",
        "boundary": tiny_boundary(),
        "heat_input": np.full(grid.Nr, 1.0e6, dtype=np.float64),
        "current_input": np.ones(grid.Nr, dtype=np.float64),
    }

    with pytest.raises(ValueError, match="active F profile.*only supported for PJ2"):
        Operator(
            grid,
            Problem(
                **base_kwargs,
                active_profiles={"h": 1, "F": 1},
            ),
        )

    with pytest.raises(ValueError, match="does not accept an active psin profile"):
        Operator(
            grid,
            Problem(
                **base_kwargs,
                active_profiles={"h": 1, "psin": 1},
            ),
        )
