from __future__ import annotations

from typing import Final

SOLVER_METHOD_POWELL: Final[int] = 1
SOLVER_METHOD_LEVENBERG_MARQUARDT: Final[int] = 2
SOLVER_METHOD_NEWTON_KRYLOV: Final[int] = 4
SOLVER_METHOD_NEWTON_RAPHSON: Final[int] = 5
SOLVER_METHOD_SUNDIALS_NEWTON_KRYLOV: Final[int] = 6
SOLVER_METHOD_SUNDIALS_NEWTON_RAPHSON: Final[int] = 7

INITIAL_POLICY_COLD_ZEROS: Final[int] = 1
INITIAL_POLICY_COLD_GEOMETRIC: Final[int] = 2
INITIAL_POLICY_COLD: Final[int] = 3
INITIAL_POLICY_WARM_CLONE: Final[int] = 4

RESIDUAL_NORMALIZATION_NONE: Final[int] = 0
RESIDUAL_NORMALIZATION_FAST: Final[int] = 1
RESIDUAL_NORMALIZATION_BALANCED: Final[int] = 2
RESIDUAL_NORMALIZATION_SAFE: Final[int] = 3

SOLVER_METHOD_CODES: Final[dict[str, int]] = {
    "powell": SOLVER_METHOD_POWELL,
    "levenberg-marquardt": SOLVER_METHOD_LEVENBERG_MARQUARDT,
    "newton-krylov": SOLVER_METHOD_NEWTON_KRYLOV,
    "newton-raphson": SOLVER_METHOD_NEWTON_RAPHSON,
    "sundials-newton-krylov": SOLVER_METHOD_SUNDIALS_NEWTON_KRYLOV,
    "sundials-newton-raphson": SOLVER_METHOD_SUNDIALS_NEWTON_RAPHSON,
}

INITIAL_POLICY_CODES: Final[dict[str, int]] = {
    "cold-zeros": INITIAL_POLICY_COLD_ZEROS,
    "cold-geometric": INITIAL_POLICY_COLD_GEOMETRIC,
    "cold": INITIAL_POLICY_COLD,
    "warm-clone": INITIAL_POLICY_WARM_CLONE,
}

RESIDUAL_NORMALIZATION_CODES: Final[dict[str, int]] = {
    "none": RESIDUAL_NORMALIZATION_NONE,
    "fast": RESIDUAL_NORMALIZATION_FAST,
    "balanced": RESIDUAL_NORMALIZATION_BALANCED,
    "safe": RESIDUAL_NORMALIZATION_SAFE,
}


def solver_method_code(value: str | int) -> int:
    code = _option_code(value, SOLVER_METHOD_CODES, "solver method")
    if code not in (
        SOLVER_METHOD_POWELL,
        SOLVER_METHOD_LEVENBERG_MARQUARDT,
        SOLVER_METHOD_NEWTON_KRYLOV,
        SOLVER_METHOD_NEWTON_RAPHSON,
        SOLVER_METHOD_SUNDIALS_NEWTON_KRYLOV,
        SOLVER_METHOD_SUNDIALS_NEWTON_RAPHSON,
    ):
        raise ValueError(f"Unsupported solver method code {code!r}")
    return code


def initial_policy_code(value: str | int) -> int:
    code = _option_code(value, INITIAL_POLICY_CODES, "initial policy")
    if code not in (
        INITIAL_POLICY_COLD_ZEROS,
        INITIAL_POLICY_COLD_GEOMETRIC,
        INITIAL_POLICY_COLD,
        INITIAL_POLICY_WARM_CLONE,
    ):
        raise ValueError(f"Unsupported initial policy code {code!r}")
    return code


def residual_normalization_code(value: str | int) -> int:
    code = _option_code(value, RESIDUAL_NORMALIZATION_CODES, "residual normalization")
    if code not in (
        RESIDUAL_NORMALIZATION_NONE,
        RESIDUAL_NORMALIZATION_FAST,
        RESIDUAL_NORMALIZATION_BALANCED,
        RESIDUAL_NORMALIZATION_SAFE,
    ):
        raise ValueError(f"Unsupported residual normalization code {code!r}")
    return code


def _option_code(value: str | int, table: dict[str, int], label: str) -> int:
    if isinstance(value, int):
        return value
    key = _normalize_option_token(value)
    try:
        return table[key]
    except KeyError as exc:
        supported = ", ".join(sorted(table))
        raise ValueError(f"Unsupported {label} {value!r}; supported: {supported}") from exc


def _normalize_option_token(value: str) -> str:
    return value.strip()
