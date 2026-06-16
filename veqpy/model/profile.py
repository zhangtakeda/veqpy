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
        "amplitude_power",
        "offset",
        "coeff",
    }

    def __init__(
        self,
        scale: float = 1.0,
        power: int = 0,
        envelope_power: int = 1,
        amplitude_power: float = 1.0,
        offset: float = 0.0,
        coeff: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        # Profile stores only the root parameters.  Derived value/derivative
        # arrays are allocated in ProfileWorkspace so one Profile can be reused
        # across grids and case refreshes.
        self.scale = scale
        self.power = power
        self.envelope_power = envelope_power
        self.amplitude_power = amplitude_power
        self.offset = offset
        self.coeff = coeff

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        """Declare serializable root attributes."""
        return {
            "scale": float,
            "power": int,
            "envelope_power": int,
            "amplitude_power": float,
            "offset": float,
            "coeff": np.ndarray | None,
        }

    @classmethod
    def reactive_inspections(cls, name: str, value: object) -> object:
        """Normalize root writes while keeping Profile free of runtime fields."""

        match name:
            case "scale":
                return float(value)
            case "amplitude_power":
                return float(value)
            case "power" | "envelope_power":
                return int(value)
            case "offset":
                if value is None:
                    raise TypeError("offset must be a float, got None")
                return float(value)
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
        kwargs = {
            "scale": self.scale,
            "power": self.power,
            "envelope_power": self.envelope_power,
            "offset": self.offset,
            "coeff": None if self.coeff is None else self.coeff.copy(),
        }
        amplitude_power = self.amplitude_power
        if amplitude_power is not None:
            kwargs["amplitude_power"] = amplitude_power
        return Profile(
            **kwargs,
        )


def _coerce_optional_array(value, *, copy: bool, name: str = "array") -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim == 0:
        scalar = value.item()
        if scalar is None or (isinstance(scalar, float) and np.isnan(scalar)):
            # Scalar object arrays from legacy serializers can encode "missing"
            # as None/NaN.  Preserve that as a passive profile marker.
            return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    return arr.copy() if copy else arr
