"""Reactive VEQ flux-surface geometry in a dense coefficient representation.

``Geometry`` owns the authoritative geometric roots of an equilibrium.  Radial
nodes, quadrature weights, calculus matrices, and basis tables are derived from
the declared discrete rules and are never serialized as independent state.

The dense coefficient layout deliberately gives every retained radial and
Fourier coefficient a physical tangent direction.  ``c_coeffs`` stores rows
``c0 .. cM`` while ``s_coeffs`` stores ``s1 .. sM``; all profile families share
one radial coefficient capacity ``L_max + 1``.
"""

from __future__ import annotations

from typing import Self

import numpy as np

from veqpy.base import Reactive, Serial
from veqpy.model.grid import Grid
from veqpy.model.profile import Profile

_RADIAL_RULES = frozenset({"legendre", "lobatto", "radau", "chebyshev", "uniform"})
_RADIAL_CALCULUS = frozenset({"spectral", "cfd33", "cfd35", "cfd55"})


class Geometry(Reactive, Serial):
    """Continuous VEQ geometry with a derived radial-poloidal evaluation grid."""

    root_properties = {
        "Nr",
        "Nt",
        "radial_rule",
        "radial_calculus",
        "K_max",
        "R0",
        "Z0",
        "a",
        "kappa_lcfs",
        "c_lcfs",
        "s_lcfs",
        "h_coeffs",
        "v_coeffs",
        "kappa_coeffs",
        "c_coeffs",
        "s_coeffs",
    }

    def __init__(
        self,
        *,
        Nr: int,
        Nt: int,
        R0: float,
        Z0: float,
        a: float,
        kappa_lcfs: float,
        c_lcfs: np.ndarray,
        s_lcfs: np.ndarray,
        h_coeffs: np.ndarray,
        v_coeffs: np.ndarray,
        kappa_coeffs: np.ndarray,
        c_coeffs: np.ndarray,
        s_coeffs: np.ndarray,
        radial_rule: str = "legendre",
        radial_calculus: str = "spectral",
        K_max: int | None = None,
    ) -> None:
        super().__init__()
        self.Nr = Nr
        self.Nt = Nt
        self.radial_rule = radial_rule
        self.radial_calculus = radial_calculus
        self.K_max = K_max
        self.R0 = R0
        self.Z0 = Z0
        self.a = a
        self.kappa_lcfs = kappa_lcfs
        self.c_lcfs = c_lcfs
        self.s_lcfs = s_lcfs
        self.h_coeffs = h_coeffs
        self.v_coeffs = v_coeffs
        self.kappa_coeffs = kappa_coeffs
        self.c_coeffs = c_coeffs
        self.s_coeffs = s_coeffs
        self.check()

    @classmethod
    def reactive_inspections(cls, name: str, value: object) -> object:
        """Normalize individual roots before Reactive stores immutable values."""

        match name:
            case "Nr" | "Nt":
                if isinstance(value, bool):
                    raise TypeError(f"{name} must be an integer")
                integer = int(value)
                if integer < 4:
                    raise ValueError(f"{name} must be at least 4")
                return integer
            case "K_max":
                if value is None:
                    return None
                if isinstance(value, bool):
                    raise TypeError("K_max must be an integer or None")
                integer = int(value)
                if integer < 0:
                    raise ValueError("K_max must be non-negative")
                return integer
            case "radial_rule":
                token = str(value).lower()
                if token not in _RADIAL_RULES:
                    available = ", ".join(sorted(_RADIAL_RULES))
                    raise ValueError(
                        f"unsupported radial_rule {token!r}; expected one of {available}"
                    )
                return token
            case "radial_calculus":
                token = str(value).lower()
                if token == "compact":
                    token = "cfd33"
                if token not in _RADIAL_CALCULUS:
                    available = ", ".join(sorted(_RADIAL_CALCULUS))
                    raise ValueError(
                        f"unsupported radial_calculus {token!r}; expected one of {available}"
                    )
                return token
            case "R0" | "Z0" | "a" | "kappa_lcfs":
                scalar = float(value)
                if not np.isfinite(scalar):
                    raise ValueError(f"{name} must be finite")
                if name in {"R0", "a", "kappa_lcfs"} and scalar <= 0.0:
                    raise ValueError(f"{name} must be positive")
                return scalar
            case "c_lcfs" | "s_lcfs" | "h_coeffs" | "v_coeffs" | "kappa_coeffs":
                return _owned_finite_array(value, name=name, ndim=1)
            case "c_coeffs" | "s_coeffs":
                return _owned_finite_array(value, name=name, ndim=2)
        return value

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        """Declare the authoritative geometry roots in stable persistence order."""

        return {
            "Nr": int,
            "Nt": int,
            "radial_rule": str,
            "radial_calculus": str,
            "K_max": int | None,
            "R0": float,
            "Z0": float,
            "a": float,
            "kappa_lcfs": float,
            "c_lcfs": np.ndarray,
            "s_lcfs": np.ndarray,
            "h_coeffs": np.ndarray,
            "v_coeffs": np.ndarray,
            "kappa_coeffs": np.ndarray,
            "c_coeffs": np.ndarray,
            "s_coeffs": np.ndarray,
        }

    def check(self) -> None:
        """Validate the dense harmonic and radial coefficient layout."""

        Serial.check(self)
        radial_size = int(self.c_coeffs.shape[1])
        if radial_size < 2:
            raise ValueError("coefficient radial axis must contain at least two entries")
        expected_radial = (radial_size,)
        for name in ("h_coeffs", "v_coeffs", "kappa_coeffs"):
            value = getattr(self, name)
            if value.shape != expected_radial:
                raise ValueError(f"{name} must have shape {expected_radial}, got {value.shape}")
        if self.s_coeffs.shape[1] != radial_size:
            raise ValueError(
                "c_coeffs and s_coeffs must share one radial coefficient axis, "
                f"got {self.c_coeffs.shape} and {self.s_coeffs.shape}"
            )

        c_count = int(self.c_coeffs.shape[0])
        s_count = int(self.s_coeffs.shape[0])
        if c_count < 2 or s_count < 1 or c_count != s_count + 1:
            raise ValueError(
                "c_coeffs must store c0..cM and s_coeffs must store s1..sM; "
                f"got {c_count} cosine rows and {s_count} sine rows"
            )
        if self.c_lcfs.shape != (c_count,):
            raise ValueError(f"c_lcfs must have shape {(c_count,)}, got {self.c_lcfs.shape}")
        if self.s_lcfs.shape != (s_count,):
            raise ValueError(f"s_lcfs must have shape {(s_count,)}, got {self.s_lcfs.shape}")

        if self.Nr < radial_size:
            raise ValueError(
                f"Nr={self.Nr} must be at least L_max + 1 = {radial_size}"
            )
        if self.Nt < 2 * self.M_max + 1:
            raise ValueError(
                f"Nt={self.Nt} must satisfy the Fourier Nyquist bound "
                f"2*M_max+1={2 * self.M_max + 1}"
            )
        if self.radial_calculus in {"cfd35", "cfd55"} and self.Nr < 5:
            raise ValueError(f"radial_calculus={self.radial_calculus!r} requires Nr >= 5")

    def freeze(self) -> Self:
        """Validate the complete cross-root layout before recursive freezing."""

        self.check()
        return super().freeze()

    @property
    def L_max(self) -> int:
        """Highest retained radial Chebyshev order."""

        return int(self.c_coeffs.shape[1]) - 1

    @property
    def M_max(self) -> int:
        """Highest retained poloidal Fourier order."""

        return int(self.c_coeffs.shape[0]) - 1

    @property
    def _grid(self) -> Grid:
        """Legacy numerical helper retained only as a derived cache."""

        return Grid(
            Nr=self.Nr,
            Nt=self.Nt,
            L_max=self.L_max,
            M_max=self.M_max,
            K_max=self.K_max,
            quadrature_scheme=self.radial_rule,
            calculus_scheme=self.radial_calculus,
        )

    @property
    def r(self) -> np.ndarray:
        """Derived radial nodes on the normalized geometric radius."""

        return self._grid.r

    @property
    def radial_weights(self) -> np.ndarray:
        """Derived full-domain radial quadrature weights."""

        return self._grid.weights

    @property
    def theta(self) -> np.ndarray:
        """Derived uniform periodic poloidal nodes."""

        return self._grid.theta

    @property
    def accumulator(self) -> np.ndarray:
        """Derived radial prefix-integration matrix."""

        return self._grid.accumulator

    @property
    def differentiator(self) -> np.ndarray:
        """Derived radial differentiation matrix."""

        return self._grid.differentiator

    @property
    def axis_interpolation_weights(self) -> np.ndarray:
        """Weights evaluating a radial nodal field at ``r=0``."""

        return self._grid.axis_interpolation_weights

    @property
    def lcfs_interpolation_weights(self) -> np.ndarray:
        """Weights evaluating a radial nodal field at ``r=1``."""

        return self._grid.edge_interpolation_weights

    @property
    def h_fields(self) -> np.ndarray:
        """Packed ``h``, ``dh/dr``, and ``d2h/dr2`` fields."""

        return _profile_fields(self._grid, self.h_coeffs, offset=0.0, power=0)

    @property
    def v_fields(self) -> np.ndarray:
        """Packed ``v``, ``dv/dr``, and ``d2v/dr2`` fields."""

        return _profile_fields(self._grid, self.v_coeffs, offset=0.0, power=0)

    @property
    def kappa_fields(self) -> np.ndarray:
        """Packed elongation profile and its first two radial derivatives."""

        return _profile_fields(
            self._grid,
            self.kappa_coeffs,
            offset=self.kappa_lcfs,
            power=0,
        )

    @property
    def c_fields(self) -> np.ndarray:
        """Cosine-angle profiles with shape ``(M_max+1, 3, Nr)``."""

        out = np.empty((self.M_max + 1, 3, self.Nr), dtype=np.float64)
        powers = self._grid.K_values
        for order in range(self.M_max + 1):
            out[order] = _profile_fields(
                self._grid,
                self.c_coeffs[order],
                offset=float(self.c_lcfs[order]),
                power=int(powers[order]),
            )
        return _readonly(out)

    @property
    def s_fields(self) -> np.ndarray:
        """Sine-angle profiles indexed by harmonic, with structural ``s0=0``."""

        out = np.zeros((self.M_max + 1, 3, self.Nr), dtype=np.float64)
        powers = self._grid.K_values
        for order in range(1, self.M_max + 1):
            out[order] = _profile_fields(
                self._grid,
                self.s_coeffs[order - 1],
                offset=float(self.s_lcfs[order - 1]),
                power=int(powers[order]),
            )
        return _readonly(out)

    @property
    def h(self) -> np.ndarray:
        return self.h_fields[0]

    @property
    def v(self) -> np.ndarray:
        return self.v_fields[0]

    @property
    def kappa(self) -> np.ndarray:
        return self.kappa_fields[0]

    @property
    def c(self) -> np.ndarray:
        return self.c_fields[:, 0]

    @property
    def s(self) -> np.ndarray:
        return self.s_fields[:, 0]

    @property
    def R(self) -> np.ndarray:
        """Major-radius coordinates on the default ``(r, theta)`` grid."""

        eta = self.theta[None, :] + self.c[0, :, None]
        if self.M_max:
            eta = eta + self.c[1:].T @ self._grid.cos_mtheta[1:]
            eta = eta + self.s[1:].T @ self._grid.sin_mtheta[1:]
        result = self.R0 + self.a * (self.h[:, None] + self.r[:, None] * np.cos(eta))
        if np.any(result <= 0.0) or np.any(~np.isfinite(result)):
            raise ValueError("Geometry produces a non-positive or non-finite major radius")
        return _readonly(result)

    @property
    def Z(self) -> np.ndarray:
        """Vertical coordinates on the default ``(r, theta)`` grid."""

        result = self.Z0 + self.a * (
            self.v[:, None]
            - self.r[:, None] * self.kappa[:, None] * np.sin(self.theta)[None, :]
        )
        if np.any(~np.isfinite(result)):
            raise ValueError("Geometry produces a non-finite vertical coordinate")
        return _readonly(result)

    @property
    def R_lcfs(self) -> np.ndarray:
        """Analytic LCFS major-radius coordinates on ``theta``."""

        eta = self.theta + float(self.c_lcfs[0])
        if self.M_max:
            eta = eta + self.c_lcfs[1:] @ self._grid.cos_mtheta[1:]
            eta = eta + self.s_lcfs @ self._grid.sin_mtheta[1:]
        result = self.R0 + self.a * np.cos(eta)
        if np.any(result <= 0.0) or np.any(~np.isfinite(result)):
            raise ValueError("Geometry LCFS has a non-positive or non-finite major radius")
        return _readonly(result)

    @property
    def Z_lcfs(self) -> np.ndarray:
        """Analytic LCFS vertical coordinates on ``theta``."""

        result = self.Z0 - self.a * self.kappa_lcfs * np.sin(self.theta)
        return _readonly(result)


def _owned_finite_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _profile_fields(grid: Grid, coeffs: np.ndarray, *, offset: float, power: int) -> np.ndarray:
    return Profile(offset=offset, power=power, coeff=coeffs, grid=grid).fields


def _readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array
