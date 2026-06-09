"""
Module: model.profile

Role:
- Hold root parameters for one profile.

Public API:
- Profile

Notes:
- `Profile` is a reactive model-layer configuration object.
- Runtime fields are materialized in `ProfileWorkspace`, not on `Profile`.
- Does not own packed state, source scaling, or solver orchestration.
"""

from __future__ import annotations

from typing import Self

import numpy as np

from veqpy.base import Reactive, Serial


class Profile(Reactive, Serial):
    """Reactive passive root-parameter specification for one one-dimensional profile."""

    root_properties = {
        "scale",
        "power",
        "envelope_power",
        "offset",
        "coeff",
    }

    def __init__(
        self,
        scale: float = 1.0,
        power: int = 0,
        envelope_power: int = 1,
        offset: float | None = 0.0,
        coeff: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.power = power
        self.envelope_power = envelope_power
        self.offset = offset
        self.coeff = coeff

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        """Declare serializable root attributes."""
        return {
            "scale": float,
            "power": int,
            "envelope_power": int,
            "offset": float,
            "coeff": np.ndarray | None,
        }

    @classmethod
    def reactive_inspections(cls, name: str, value: object) -> object:
        """Normalize root writes while keeping Profile free of runtime fields."""

        match name:
            case "scale":
                return float(value)
            case "power" | "envelope_power":
                return int(value)
            case "offset":
                return 0.0 if value is None else float(value)
            case "coeff":
                return _coerce_optional_array(value, copy=False, name="coeff")
        return value

    def check(self) -> None:
        """Validate root parameters and serializable fields."""
        for key, expected in type(self).serial_attributes().items():
            value = getattr(self, key)
            if value is None:
                continue
            if expected in {float, int} and not isinstance(value, (expected, np.generic)):
                raise TypeError(
                    f"Attribute '{key}' must be {expected.__name__}, got {type(value).__name__}"
                )
            if isinstance(value, np.ndarray) and value.ndim != 1:
                raise ValueError(f"Attribute '{key}' must be 1D, got {value.shape}")

    def copy(self) -> Self:
        """Copy root parameters."""
        return Profile(
            scale=self.scale,
            power=self.power,
            envelope_power=self.envelope_power,
            offset=self.offset,
            coeff=_copy_optional_array(self.coeff),
        )


def _coerce_optional_array(value, *, copy: bool, name: str = "array") -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim == 0:
        scalar = value.item()
        if scalar is None or (isinstance(scalar, float) and np.isnan(scalar)):
            return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    return arr.copy() if copy else arr


def _copy_optional_array(value: np.ndarray | None) -> np.ndarray | None:
    return None if value is None else value.copy()
