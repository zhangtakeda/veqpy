from __future__ import annotations

import re
from pathlib import Path

from veqlib.facade import options
from veqlib.facade import types as facade_types


def test_python_abi_enum_codes_match_cxx_header() -> None:
    header_codes = _parse_veqlib_abi_codes()

    expected = {
        "solver_method_powell": options.SOLVER_METHOD_POWELL,
        "solver_method_levenberg_marquardt": options.SOLVER_METHOD_LEVENBERG_MARQUARDT,
        "solver_method_newton_krylov": options.SOLVER_METHOD_NEWTON_KRYLOV,
        "solver_method_newton_raphson": options.SOLVER_METHOD_NEWTON_RAPHSON,
        "initial_policy_cold_zeros": options.INITIAL_POLICY_COLD_ZEROS,
        "initial_policy_cold_geometric": options.INITIAL_POLICY_COLD_GEOMETRIC,
        "initial_policy_cold": options.INITIAL_POLICY_COLD,
        "continue_policy_cold_zeros": options.CONTINUE_POLICY_COLD_ZEROS,
        "continue_policy_cold_geometric": options.CONTINUE_POLICY_COLD_GEOMETRIC,
        "continue_policy_cold": options.CONTINUE_POLICY_COLD,
        "continue_policy_warm_fixed": options.CONTINUE_POLICY_WARM_FIXED,
        "continue_policy_warm_predict": options.CONTINUE_POLICY_WARM_PREDICT,
        "continue_policy_warm_chord": options.CONTINUE_POLICY_WARM_CHORD,
        "continue_policy_warm": options.CONTINUE_POLICY_WARM,
        "residual_normalization_none": options.RESIDUAL_NORMALIZATION_NONE,
        "residual_normalization_fast": options.RESIDUAL_NORMALIZATION_FAST,
        "residual_normalization_balanced": options.RESIDUAL_NORMALIZATION_BALANCED,
        "residual_normalization_safe": options.RESIDUAL_NORMALIZATION_SAFE,
        "source_route_pf": facade_types._SOURCE_ROUTE_CODES["PF"],
        "source_route_pp": facade_types._SOURCE_ROUTE_CODES["PP"],
        "source_route_pi": facade_types._SOURCE_ROUTE_CODES["PI"],
        "source_route_pj1": facade_types._SOURCE_ROUTE_CODES["PJ1"],
        "source_route_pj2": facade_types._SOURCE_ROUTE_CODES["PJ2"],
        "source_route_pq": facade_types._SOURCE_ROUTE_CODES["PQ"],
        "source_coordinate_rho": facade_types._SOURCE_COORDINATE_CODES["rho"],
        "source_coordinate_psin": facade_types._SOURCE_COORDINATE_CODES["psin"],
        "source_constraint_null": facade_types._SOURCE_CONSTRAINT_CODES["null"],
        "source_constraint_ip": facade_types._SOURCE_CONSTRAINT_CODES["Ip"],
        "source_constraint_beta": facade_types._SOURCE_CONSTRAINT_CODES["beta"],
        "source_constraint_ip_beta": facade_types._SOURCE_CONSTRAINT_CODES["Ip_beta"],
        "source_nodes_uniform": facade_types._SOURCE_NODES_CODES["uniform"],
        "source_nodes_grid": facade_types._SOURCE_NODES_CODES["grid"],
        "source_active_none": facade_types._SOURCE_ACTIVE_FAMILY_CODES["none"],
        "source_active_psin": facade_types._SOURCE_ACTIVE_FAMILY_CODES["psin"],
        "source_active_F": facade_types._SOURCE_ACTIVE_FAMILY_CODES["F"],
        "source_parameterization_identity": facade_types._SOURCE_PARAMETERIZATION_CODES[
            "identity"
        ],
        "source_parameterization_sqrt_psin": facade_types._SOURCE_PARAMETERIZATION_CODES[
            "sqrt_psin"
        ],
    }

    assert header_codes == expected


def _parse_veqlib_abi_codes() -> dict[str, int]:
    header = Path(__file__).resolve().parents[1] / "veqlib" / "core" / "abi_enums.h"
    text = header.read_text(encoding="utf-8")
    namespace_match = re.search(r"namespace veqlib_abi\s*\{(?P<body>.*?)\n\}", text, re.S)
    assert namespace_match is not None
    body = namespace_match.group("body")
    pairs = re.findall(r"inline constexpr int ([A-Za-z0-9_]+)\s*=\s*([0-9]+);", body)
    return {name: int(value) for name, value in pairs}
