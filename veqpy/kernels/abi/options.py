"""Python mirrors of VEQlib C++ ABI option codes.

Python facade inputs use canonical string tokens. Native kernels consume integer
ABI enum codes derived from those strings. Keep these constants synchronized with
``veqlib/cxx_core/core/abi_enums.h``.
"""

from __future__ import annotations

from typing import Final

SOLVER_METHOD_POWELL: Final[int] = 1
SOLVER_METHOD_LEVENBERG_MARQUARDT: Final[int] = 2
SOLVER_METHOD_NEWTON_KRYLOV: Final[int] = 4
SOLVER_METHOD_NEWTON_RAPHSON: Final[int] = 5

INITIAL_POLICY_COLD_ZEROS: Final[int] = 1
INITIAL_POLICY_COLD_GEOMETRIC: Final[int] = 2
INITIAL_POLICY_COLD: Final[int] = 3

CONTINUE_POLICY_COLD_ZEROS: Final[int] = 1
CONTINUE_POLICY_COLD_GEOMETRIC: Final[int] = 2
CONTINUE_POLICY_COLD: Final[int] = 3
CONTINUE_POLICY_WARM_FIXED: Final[int] = 4
CONTINUE_POLICY_WARM_PREDICT: Final[int] = 5
CONTINUE_POLICY_WARM_CHORD: Final[int] = 6
CONTINUE_POLICY_WARM: Final[int] = 7

RESIDUAL_NORMALIZATION_NONE: Final[int] = 0
RESIDUAL_NORMALIZATION_FAST: Final[int] = 1
RESIDUAL_NORMALIZATION_BALANCED: Final[int] = 2
RESIDUAL_NORMALIZATION_SAFE: Final[int] = 3

SOLVER_METHOD_CODES: Final[dict[str, int]] = {
    "powell": SOLVER_METHOD_POWELL,
    "levenberg-marquardt": SOLVER_METHOD_LEVENBERG_MARQUARDT,
    "newton-krylov": SOLVER_METHOD_NEWTON_KRYLOV,
    "newton-raphson": SOLVER_METHOD_NEWTON_RAPHSON,
}

INITIAL_POLICY_CODES: Final[dict[str, int]] = {
    "cold-zeros": INITIAL_POLICY_COLD_ZEROS,
    "cold-geometric": INITIAL_POLICY_COLD_GEOMETRIC,
    "cold": INITIAL_POLICY_COLD,
}

CONTINUE_POLICY_CODES: Final[dict[str, int]] = {
    "cold-zeros": CONTINUE_POLICY_COLD_ZEROS,
    "cold-geometric": CONTINUE_POLICY_COLD_GEOMETRIC,
    "cold": CONTINUE_POLICY_COLD,
    "warm-fixed": CONTINUE_POLICY_WARM_FIXED,
    "warm-predict": CONTINUE_POLICY_WARM_PREDICT,
    "warm-chord": CONTINUE_POLICY_WARM_CHORD,
    "warm": CONTINUE_POLICY_WARM,
}

RESIDUAL_NORMALIZATION_CODES: Final[dict[str, int]] = {
    "none": RESIDUAL_NORMALIZATION_NONE,
    "fast": RESIDUAL_NORMALIZATION_FAST,
    "balanced": RESIDUAL_NORMALIZATION_BALANCED,
    "safe": RESIDUAL_NORMALIZATION_SAFE,
}


def normalize_solver_method(value: str) -> str:
    return _option_token(value, SOLVER_METHOD_CODES, "solver method")


def normalize_initial_policy(value: str) -> str:
    return _option_token(value, INITIAL_POLICY_CODES, "initial policy")


def normalize_continue_policy(value: str) -> str:
    return _option_token(value, CONTINUE_POLICY_CODES, "continue policy")


def normalize_residual_normalization(value: str) -> str:
    return _option_token(value, RESIDUAL_NORMALIZATION_CODES, "residual normalization")


def solver_method_code(value: str) -> int:
    return SOLVER_METHOD_CODES[normalize_solver_method(value)]


def initial_policy_code(value: str) -> int:
    return INITIAL_POLICY_CODES[normalize_initial_policy(value)]


def continue_policy_code(value: str) -> int:
    return CONTINUE_POLICY_CODES[normalize_continue_policy(value)]


def residual_normalization_code(value: str) -> int:
    return RESIDUAL_NORMALIZATION_CODES[normalize_residual_normalization(value)]


def _option_token(value: str, table: dict[str, int], label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a canonical string token")
    key = value.strip().lower()
    if key in table:
        return key
    supported = ", ".join(sorted(table))
    raise ValueError(f"Unsupported {label} {value!r}; supported: {supported}")
