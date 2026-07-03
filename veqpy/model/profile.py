"""
Module: model.profile

Role:
- Hold root parameters for one profile.

Public API:
- Profile

Notes:
- `Profile` is an immutable model-layer profile-parameter snapshot.
- Runtime fields are materialized in `ProfileWorkspace`, not on `Profile`.
- Does not own packed state, source scaling, or solver orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np

from veqpy.base import SERIAL_TYPE_REGISTRY, Serial


@dataclass(frozen=True, slots=True)
class Profile(Serial):
    """Immutable root-parameter specification for one one-dimensional profile."""

    scale: float = 1.0
    power: int = 0
    envelope_power: int = 1
    amplitude_power: float = 1.0
    offset: float = 0.0
    coeff: np.ndarray | None = None

    def __post_init__(self) -> None:
        # Profile stores only root profile parameters. Derived value/derivative
        # arrays are allocated in ProfileWorkspace after Kernel runtime flattens setup.
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "power", int(self.power))
        object.__setattr__(self, "envelope_power", int(self.envelope_power))
        object.__setattr__(self, "amplitude_power", float(self.amplitude_power))
        if self.offset is None:
            raise TypeError("offset must be a float, got None")
        object.__setattr__(self, "offset", float(self.offset))
        object.__setattr__(self, "coeff", _coerce_optional_array(self.coeff, name="coeff"))

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

    def check(self) -> None:
        """Validate root parameters and serializable fields."""
        Serial.check(self)
        if self.coeff is not None and self.coeff.ndim != 1:
            raise ValueError(f"Attribute 'coeff' must be 1D, got {self.coeff.shape}")

    def copy(self) -> Self:
        """Copy root parameters."""
        return type(self)(
            scale=self.scale,
            power=self.power,
            envelope_power=self.envelope_power,
            amplitude_power=self.amplitude_power,
            offset=self.offset,
            coeff=None if self.coeff is None else self.coeff.copy(),
        )


def _coerce_optional_array(value, *, name: str = "array") -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim == 0:
        scalar = value.item()
        if scalar is None or (isinstance(scalar, float) and np.isnan(scalar)):
            # Scalar object arrays from older serializers can encode "missing"
            # as None/NaN.  Preserve that as a passive profile marker.
            return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty or None")
    out = arr.copy()
    out.setflags(write=False)
    return out


SERIAL_TYPE_REGISTRY[Profile.__name__] = Profile
