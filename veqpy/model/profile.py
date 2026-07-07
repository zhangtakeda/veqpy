"""
Module: veqpy.model.profile

Role:
- Hold root parameters for one profile.
- Materialize value and radial derivatives when a grid is bound.

Public API:
- Profile

Notes:
- ``grid`` is a reactive evaluation context, not a serialized field.
- ``value``, ``derivative``, and ``second_derivative`` require ``grid``.
- Solver packed workspaces still own backend runtime memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np

from veqpy.base import SERIAL_TYPE_REGISTRY, Reactive, Serial

if TYPE_CHECKING:
    from veqpy.model.grid import Grid

_AMPLITUDE_POWER_FLOOR = 1.0e-10
_GRID_UNSET = object()


class Profile(Reactive, Serial):
    """Reactive root-parameter specification for one one-dimensional profile."""

    root_properties = {
        "scale",
        "power",
        "envelope_power",
        "amplitude_power",
        "offset",
        "coeff",
        "grid",
    }

    def __init__(
        self,
        *,
        scale: float = 1.0,
        power: int = 0,
        envelope_power: int = 1,
        amplitude_power: float = 1.0,
        offset: float = 0.0,
        coeff: np.ndarray | None = None,
        grid: Grid | None = None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.power = power
        self.envelope_power = envelope_power
        self.amplitude_power = amplitude_power
        self.offset = offset
        self.coeff = coeff
        self.grid = grid

    @classmethod
    def reactive_inspections(cls, name: str, value: object) -> object:
        match name:
            case "scale" | "amplitude_power":
                return float(value)
            case "power" | "envelope_power":
                return int(value)
            case "offset":
                if value is None:
                    raise TypeError("offset must be a float, got None")
                return float(value)
            case "coeff":
                return _coerce_optional_array(value, name="coeff")
            case "grid":
                if value is None:
                    return None
                from veqpy.model.grid import Grid

                if not isinstance(value, Grid):
                    raise TypeError(f"grid must be Grid or None, got {type(value).__name__}")
                return value
        return value

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        """Declare persistent profile root parameters.

        ``grid`` is deliberately omitted because it is an evaluation context.
        """

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

    def copy(self, *, grid: Grid | None | object = _GRID_UNSET) -> Self:
        """Copy root parameters and optionally replace the evaluation grid."""

        return type(self)(
            scale=self.scale,
            power=self.power,
            envelope_power=self.envelope_power,
            amplitude_power=self.amplitude_power,
            offset=self.offset,
            coeff=None if self.coeff is None else self.coeff.copy(),
            grid=self.grid if grid is _GRID_UNSET else grid,
        )

    def with_grid(self, grid: Grid) -> Self:
        """Return a copy bound to ``grid`` for profile field evaluation."""

        return self.copy(grid=grid)

    @property
    def fields(self) -> np.ndarray:
        """Stack ``(value, derivative, second_derivative)`` on the bound grid."""

        grid = self.grid
        if grid is None:
            raise RuntimeError("Profile.grid is required to evaluate profile fields")
        fields = _profile_fields_on_grid(
            grid=grid,
            scale=self.scale,
            power=self.power,
            envelope_power=self.envelope_power,
            amplitude_power=self.amplitude_power,
            offset=self.offset,
            coeff=self.coeff,
        )
        fields.flags.writeable = False
        return fields

    @property
    def value(self) -> np.ndarray:
        """Profile values on ``grid.rho``."""

        return self.fields[0]

    @property
    def derivative(self) -> np.ndarray:
        """First radial derivative on ``grid.rho``."""

        return self.fields[1]

    @property
    def second_derivative(self) -> np.ndarray:
        """Second radial derivative on ``grid.rho``."""

        return self.fields[2]


def _profile_fields_on_grid(
    *,
    grid: Grid,
    scale: float,
    power: int,
    envelope_power: int,
    amplitude_power: float,
    offset: float,
    coeff: np.ndarray | None,
) -> np.ndarray:
    fields = np.empty((3, grid.Nr), dtype=np.float64)
    rp_fields = _power_terms(grid.rho, power)
    env_fields = _envelope_terms(
        grid.rho,
        grid.rho_powers[2],
        grid.y,
        envelope_power,
    )
    _update_profile_fields(
        fields,
        grid.T,
        grid.T_r,
        grid.T_rr,
        rp_fields,
        env_fields,
        float(offset),
        coeff,
        float(amplitude_power),
    )
    scale = float(scale)
    if scale != 1.0:
        np.multiply(fields, scale, out=fields)
    return fields


def _update_profile_fields(
    out_fields: np.ndarray,
    T: np.ndarray,
    T_r: np.ndarray,
    T_rr: np.ndarray,
    rp_fields: np.ndarray,
    env_fields: np.ndarray,
    offset: float,
    coeff: np.ndarray | None,
    amplitude_power: float,
) -> None:
    if coeff is None:
        if amplitude_power == 1.0:
            np.multiply(rp_fields, offset, out=out_fields)
            return
        amp, amp_r, amp_rr = _amplitude_power(offset, 0.0, 0.0, amplitude_power)
        out_fields[0] = amp * rp_fields[0]
        out_fields[1] = amp * rp_fields[1] + rp_fields[0] * amp_r
        out_fields[2] = amp * rp_fields[2] + 2.0 * rp_fields[1] * amp_r + rp_fields[0] * amp_rr
        return

    coeff_array = np.asarray(coeff, dtype=np.float64)
    coeff_size = coeff_array.size
    series = coeff_array @ T[:coeff_size]
    series_r = coeff_array @ T_r[:coeff_size]
    series_rr = coeff_array @ T_rr[:coeff_size]

    env = env_fields[0]
    env_r = env_fields[1]
    env_rr = env_fields[2]
    base = env * series
    base_r = env_r * series + env * series_r
    base_rr = env_rr * series + 2.0 * env_r * series_r + env * series_rr

    if amplitude_power == 1.0:
        amp = offset + base
        amp_r = base_r
        amp_rr = base_rr
    else:
        amp, amp_r, amp_rr = _amplitude_power(
            offset + base,
            base_r,
            base_rr,
            amplitude_power,
        )

    out_fields[0] = rp_fields[0] * amp
    out_fields[1] = rp_fields[1] * amp + rp_fields[0] * amp_r
    out_fields[2] = rp_fields[2] * amp + 2.0 * rp_fields[1] * amp_r + rp_fields[0] * amp_rr


def _amplitude_power(
    amp: float | np.ndarray,
    amp_r: float | np.ndarray,
    amp_rr: float | np.ndarray,
    amplitude_power: float,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    if amplitude_power == 1.0:
        return amp, amp_r, amp_rr

    amp_safe = np.maximum(amp, _AMPLITUDE_POWER_FLOOR)
    if amplitude_power == 0.5:
        value = np.sqrt(amp_safe)
        inv_value = 1.0 / value
        inv_value3 = inv_value / amp_safe
        return (
            value,
            0.5 * amp_r * inv_value,
            0.5 * amp_rr * inv_value - 0.25 * amp_r * amp_r * inv_value3,
        )

    value = amp_safe**amplitude_power
    value_r = amplitude_power * amp_safe ** (amplitude_power - 1.0) * amp_r
    value_rr = (
        amplitude_power * amp_safe ** (amplitude_power - 1.0) * amp_rr
        + amplitude_power
        * (amplitude_power - 1.0)
        * amp_safe ** (amplitude_power - 2.0)
        * amp_r
        * amp_r
    )
    return value, value_r, value_rr


def _power_terms(rho: np.ndarray, power: int) -> np.ndarray:
    power = int(power)
    out = np.empty((3, rho.shape[0]), dtype=np.float64)
    if power == 0:
        out[0].fill(1.0)
        out[1].fill(0.0)
        out[2].fill(0.0)
        return out
    out[0] = rho**power
    out[1] = power * rho ** (power - 1)
    if power == 1:
        out[2].fill(0.0)
    else:
        out[2] = power * (power - 1) * rho ** (power - 2)
    return out


def _envelope_terms(
    rho: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    envelope_power: int,
) -> np.ndarray:
    envelope_power = int(envelope_power)
    out = np.empty((3, rho.shape[0]), dtype=np.float64)
    if envelope_power == 0:
        out[0].fill(1.0)
        out[1].fill(0.0)
        out[2].fill(0.0)
        return out
    if envelope_power == 1:
        out[0] = y
        out[1] = -2.0 * rho
        out[2].fill(-2.0)
        return out
    out[0] = y**envelope_power
    out[1] = -2.0 * envelope_power * rho * y ** (envelope_power - 1)
    out[2] = -2.0 * envelope_power * y ** (envelope_power - 1) + 4.0 * envelope_power * (
        envelope_power - 1
    ) * rho2 * y ** (envelope_power - 2)
    return out


def _coerce_optional_array(value, *, name: str = "array") -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim == 0:
        scalar = value.item()
        if scalar is None or (isinstance(scalar, float) and np.isnan(scalar)):
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
