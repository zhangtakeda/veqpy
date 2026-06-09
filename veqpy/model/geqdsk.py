"""
Module: model.geqdsk

Role:
- Hold passive GEQDSK payload data.
- Read and write canonical GEQDSK text files.

Public API:
- Geqdsk

Notes:
- Boundary fitting is owned by ``veqpy.model.boundary``.
- ``Geqdsk`` stores GEQDSK data and serialization behavior only.
"""

from __future__ import annotations

import os
import re
from dataclasses import InitVar, dataclass, field
from typing import Self

import numpy as np

from veqpy.base import Serial, read_serializer, write_serializer


@dataclass(slots=True)
class Geqdsk(Serial):
    """Passive GEQDSK payload with canonical text serialization helpers."""

    path: InitVar[str | os.PathLike[str] | None] = None
    header: str = ""

    NR: int = 0
    NZ: int = 0

    R0: float = 0.0
    Z0: float = 0.0
    dR: float = 0.0
    dZ: float = 0.0
    Rmin: float = 0.0
    Rmax: float = 0.0
    Zmin: float = 0.0
    Zmax: float = 0.0

    boundary: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    limiter: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))

    Bt0: float = 0.0
    Raxis: float = 0.0
    Zaxis: float = 0.0
    Ip: float = 0.0
    psi_axis: float = 0.0
    psi_bound: float = 0.0

    F: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    P: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    FF_psi: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    P_psi: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    q: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    psi: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        """Declare the passive GEQDSK fields persisted by ``Serial``."""
        return {
            "header": str,
            "NR": int,
            "NZ": int,
            "R0": float,
            "Z0": float,
            "dR": float,
            "dZ": float,
            "Rmin": float,
            "Rmax": float,
            "Zmin": float,
            "Zmax": float,
            "boundary": np.ndarray,
            "limiter": np.ndarray,
            "Bt0": float,
            "Raxis": float,
            "Zaxis": float,
            "Ip": float,
            "psi_axis": float,
            "psi_bound": float,
            "F": np.ndarray,
            "P": np.ndarray,
            "FF_psi": np.ndarray,
            "P_psi": np.ndarray,
            "q": np.ndarray,
            "psi": np.ndarray,
        }

    def __post_init__(self, path: str | os.PathLike[str] | None) -> None:
        self.header = str(self.header)
        self.NR = int(self.NR)
        self.NZ = int(self.NZ)

        for name in (
            "R0",
            "Z0",
            "dR",
            "dZ",
            "Rmin",
            "Rmax",
            "Zmin",
            "Zmax",
            "Bt0",
            "Raxis",
            "Zaxis",
            "Ip",
            "psi_axis",
            "psi_bound",
        ):
            setattr(self, name, float(getattr(self, name)))

        self.boundary = _coerce_point_array(self.boundary, name="boundary")
        self.limiter = _coerce_point_array(self.limiter, name="limiter")
        self.F = _coerce_vector(self.F, name="F")
        self.P = _coerce_vector(self.P, name="P")
        self.FF_psi = _coerce_vector(self.FF_psi, name="FF_psi")
        self.P_psi = _coerce_vector(self.P_psi, name="P_psi")
        self.q = _coerce_vector(self.q, name="q")
        self.psi = _coerce_matrix(self.psi, name="psi")

        if path is not None:
            self.read(os.fspath(path))
            return

        self._refresh_spacing()

    def check(self) -> None:
        """Validate GEQDSK array ranks and dimensions against ``NR``/``NZ``."""
        if self.NR < 0 or self.NZ < 0:
            raise ValueError("NR and NZ must be non-negative")

        for name in ("boundary", "limiter"):
            arr = getattr(self, name)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"{name} must have shape (N, 2), got {arr.shape}")

        for name in ("F", "P", "FF_psi", "P_psi", "q"):
            arr = getattr(self, name)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1D, got {arr.shape}")

        if self.psi.ndim != 2:
            raise ValueError(f"psi must be 2D, got {self.psi.shape}")

        if self.NR == 0 or self.NZ == 0:
            return

        expected_radial = self.NR
        expected_psi = (self.NR, self.NZ)
        for name in ("F", "P", "FF_psi", "P_psi", "q"):
            arr = getattr(self, name)
            if arr.size != expected_radial:
                raise ValueError(f"{name} must have length {expected_radial}, got {arr.size}")

        if self.psi.shape != expected_psi:
            raise ValueError(f"psi must have shape {expected_psi}, got {self.psi.shape}")

    @read_serializer("txt", "geqdsk", "gfile")
    def read_geqdsk(self, file: str) -> Self:
        """Read a canonical GEQDSK/g-file text payload into this instance."""
        with open(file, "r", encoding="utf-8") as handle:
            self._read_header(handle)
            self._read_geometry(handle)
            self._read_axfig_and_current(handle)
            self._read_profiles_and_boundary(handle)
        self._refresh_spacing()
        return self

    @write_serializer("txt", "geqdsk", "gfile")
    def write_geqdsk(self, file: str) -> None:
        """Write this instance using the fixed-width GEQDSK text layout."""
        self.check()
        with open(file, "w", encoding="utf-8") as handle:
            handle.write(_header_line(self.header, self.NR, self.NZ))
            handle.write(
                _float_line(
                    [self.Rmax - self.Rmin, self.Zmax - self.Zmin, self.R0, self.Rmin, self.Z0]
                )
            )
            handle.write(
                _float_line([self.Raxis, self.Zaxis, self.psi_axis, self.psi_bound, self.Bt0])
            )
            handle.write(_float_line([self.Ip, self.psi_axis, 0.0, self.Raxis, 0.0]))
            handle.write(_float_line([self.Zaxis, 0.0, self.psi_bound, 0.0, 0.0]))
            handle.write(_format_float_fields(self.F))
            handle.write(_format_float_fields(self.P))
            handle.write(_format_float_fields(self.FF_psi))
            handle.write(_format_float_fields(self.P_psi))
            # GEQDSK stores psirz with Z as the leading dimension in file order.
            handle.write(_format_float_fields(self.psi.T.reshape(-1)))
            handle.write(_format_float_fields(self.q))
            handle.write(f"{int(self.boundary.shape[0])} {int(self.limiter.shape[0])}\n")
            handle.write(_format_float_fields(self.boundary.reshape(-1)))
            handle.write(_format_float_fields(self.limiter.reshape(-1)))

    def _read_header(self, file) -> None:
        line = file.readline()
        match = re.match(r"^\s*(.*)\s+(\d+)\s+(\d+)\s*$", line, re.I)
        if match is None:
            raise ValueError(f"Error reading header from line: {line}")
        self.header = match.group(1).rstrip()
        self.NR = int(match.group(2))
        self.NZ = int(match.group(3))

    def _read_geometry(self, file) -> None:
        line = _sanitize_line(file.readline())
        match = re.match(r"^\s*(\S+)\s*(\S+)\s*(\S+)\s*(\S+)\s*(\S+)", line)
        if match is None:
            raise ValueError(f"Error reading geometry from line: {line}")

        r_box_len, z_box_len, self.R0, self.Rmin, self.Z0 = map(float, match.groups())
        self.Rmax = self.Rmin + r_box_len
        self.Zmin = self.Z0 - 0.5 * z_box_len
        self.Zmax = self.Z0 + 0.5 * z_box_len

    def _read_axfig_and_current(self, file) -> None:
        for index in range(2):
            line = _sanitize_line(file.readline())
            values = list(map(float, re.split(r"\s+", line.strip())))
            if index == 0:
                self.Raxis, self.Zaxis, self.psi_axis, self.psi_bound, self.Bt0 = values
            else:
                self.Ip = values[0]

    def _read_profiles_and_boundary(self, file) -> None:
        file.readline()
        payload = _sanitize_line(file.read().replace("\n", " "))
        fields = re.split(r"\s+", payload.strip())
        data = np.array(
            [_safe_float_conversion(value) for value in fields if value], dtype=np.float64
        )

        nr = self.NR
        nz = self.NZ
        index = 0

        def take(count: int, *, label: str) -> np.ndarray:
            nonlocal index
            end = index + count
            if data.size < end:
                raise ValueError(f"GEQDSK payload ended before {label}.")
            out = data[index:end]
            index = end
            return out

        self.F = take(nr, label="F profile").copy()
        self.P = take(nr, label="P profile").copy()
        self.FF_psi = take(nr, label="FF_psi profile").copy()
        self.P_psi = take(nr, label="P_psi profile").copy()
        # GEQDSK stores psirz with Z as the leading dimension in file order.
        self.psi = take(nr * nz, label="psi grid").reshape(nz, nr).T.copy()
        self.q = take(nr, label="q profile").copy()

        counts = take(2, label="boundary metadata")
        nbound = int(counts[0])
        nlimiter = int(counts[1])

        boundary_flat = take(2 * nbound, label="LCFS boundary points")
        limiter_flat = take(2 * nlimiter, label="limiter points")
        self.boundary = boundary_flat.reshape(nbound, 2).copy()
        self.limiter = limiter_flat.reshape(nlimiter, 2).copy()

    def _refresh_spacing(self) -> None:
        if self.NR > 1 and np.isfinite(self.Rmax) and np.isfinite(self.Rmin):
            self.dR = (self.Rmax - self.Rmin) / (self.NR - 1)
        else:
            self.dR = 0.0

        if self.NZ > 1 and np.isfinite(self.Zmax) and np.isfinite(self.Zmin):
            self.dZ = (self.Zmax - self.Zmin) / (self.NZ - 1)
        else:
            self.dZ = 0.0


def _coerce_vector(value, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    return arr.copy()


def _coerce_matrix(value, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {arr.shape}")
    return arr.copy()


def _coerce_point_array(value, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {arr.shape}")
    return arr.copy()


def _sanitize_line(line: str) -> str:
    return re.sub(r"([^Ee])-", r"\1 -", line)


def _safe_float_conversion(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return np.nan


def _header_line(header: str, nr: int, nz: int) -> str:
    title = (header or "VEQPy GEQDSK").strip()
    return f"{title} {int(nr)} {int(nz)}\n"


def _float_line(values: list[float] | tuple[float, ...]) -> str:
    return "".join(f"{float(value):16.9E}" for value in values) + "\n"


def _format_float_fields(values: np.ndarray, *, columns: int = 5) -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return ""
    lines = []
    for start in range(0, arr.size, columns):
        chunk = arr[start : start + columns]
        lines.append("".join(f"{float(value):16.9E}" for value in chunk))
    return "\n".join(lines) + "\n"
