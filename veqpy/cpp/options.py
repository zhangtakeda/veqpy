from __future__ import annotations

from typing import Final

SOLVER_METHOD_POWELL: Final[int] = 1
SOLVER_METHOD_LEVENBERG_MARQUARDT: Final[int] = 2

INITIAL_POLICY_COLD_ZEROS: Final[int] = 1
INITIAL_POLICY_COLD_GEOMETRIC: Final[int] = 2
INITIAL_POLICY_COLD: Final[int] = 3
INITIAL_POLICY_WARM_CLONE: Final[int] = 4

RESIDUAL_NORMALIZATION_BLOCK_RMS: Final[int] = 1

SOLVER_METHOD_CODES: Final[dict[str, int]] = {
    "powell": SOLVER_METHOD_POWELL,
    "levenberg-marquardt": SOLVER_METHOD_LEVENBERG_MARQUARDT,
    "lm": SOLVER_METHOD_LEVENBERG_MARQUARDT,
}

INITIAL_POLICY_CODES: Final[dict[str, int]] = {
    "cold-zeros": INITIAL_POLICY_COLD_ZEROS,
    "cold-geometric": INITIAL_POLICY_COLD_GEOMETRIC,
    "cold": INITIAL_POLICY_COLD,
    "warm-clone": INITIAL_POLICY_WARM_CLONE,
    # Transitional aliases accepted at the Python boundary only; JSON payloads
    # sent to nanobind must carry integer *_code fields.
    "zeros": INITIAL_POLICY_COLD_ZEROS,
    "zero": INITIAL_POLICY_COLD_ZEROS,
    "geometric-refined": INITIAL_POLICY_COLD_GEOMETRIC,
    "geometric": INITIAL_POLICY_COLD_GEOMETRIC,
    "auto": INITIAL_POLICY_COLD,
    "warm": INITIAL_POLICY_WARM_CLONE,
    "warm-start": INITIAL_POLICY_WARM_CLONE,
    "warmstart": INITIAL_POLICY_WARM_CLONE,
}

RESIDUAL_NORMALIZATION_CODES: Final[dict[str, int]] = {
    "block-rms": RESIDUAL_NORMALIZATION_BLOCK_RMS,
}


def solver_method_code(value: str | int) -> int:
    code = _option_code(value, SOLVER_METHOD_CODES, "solver method")
    if code not in (SOLVER_METHOD_POWELL, SOLVER_METHOD_LEVENBERG_MARQUARDT):
        raise ValueError(f"Unsupported solver method code {code!r}")
    return code


def initial_policy_code(value: str | int, *, cold_prefers_geometric: bool | None = None) -> int:
    code = _option_code(value, INITIAL_POLICY_CODES, "initial policy")
    if code == INITIAL_POLICY_COLD and cold_prefers_geometric is not None:
        return (
            INITIAL_POLICY_COLD_GEOMETRIC
            if cold_prefers_geometric
            else INITIAL_POLICY_COLD_ZEROS
        )
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
    if code != RESIDUAL_NORMALIZATION_BLOCK_RMS:
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
    return value.strip().lower().replace("_", "-")
