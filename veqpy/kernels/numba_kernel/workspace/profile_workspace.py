"""
Module: veqpy.kernels.numba_kernel.workspace.profile_workspace

Role:
- Own profile-stage runtime memory and profile metadata arrays.

Public API:
- ProfileWorkspace

Notes:
- Profile field storage is keyed by stable runtime profile ids.
- This module does not own profile object construction semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from veqpy.kernels.numba_kernel.profile_stage import update_profile
from veqpy.kernels.numba_kernel.workspace.field_rows import PROFILE_R, PROFILE_RR, PROFILE_VALUE

if TYPE_CHECKING:
    from veqpy.kernels.numba_kernel.workspace.grid_workspace import GridWorkspace


@dataclass(init=False, slots=True)
class ProfileWorkspace:
    """Profile stage memory owner.

    Profile field arrays have shape ``(n_profiles, 3, Nr)``.  The first axis is
    the stable ``profile_id`` from the operator plan; active profiles are a list
    of profile ids, not a second owner of profile field storage.

    Derivative axis ``0/1/2`` means value, radial first derivative, radial second
    derivative.

    Fourier family field arrays have shape ``(M_max + 1, 3, Nr)``.
    Each named ``*_fields`` array is directly owned by this workspace; there is
    no hidden packing axis or first-axis row contract.
    """

    profile_names: tuple[str, ...]
    profile_index: dict[str, int]
    profile_fields: np.ndarray
    profile_rp_fields: np.ndarray
    profile_env_fields: np.ndarray
    active_profile_ids: np.ndarray
    active_offsets: np.ndarray
    active_scales: np.ndarray
    active_amplitude_powers: np.ndarray
    active_lengths: np.ndarray
    active_coeff_index_rows: np.ndarray
    _active_profile_ids_buffer: np.ndarray
    _active_offsets_buffer: np.ndarray
    _active_scales_buffer: np.ndarray
    _active_amplitude_powers_buffer: np.ndarray
    _active_lengths_buffer: np.ndarray
    _active_coeff_index_rows_buffer: np.ndarray
    c_family_fields: np.ndarray
    s_family_fields: np.ndarray
    c_family_base_fields: np.ndarray
    s_family_base_fields: np.ndarray
    c_family_source_profile_ids: np.ndarray
    s_family_source_profile_ids: np.ndarray

    def __init__(
        self,
        *,
        nr: int,
        m_max: int,
        profile_names: tuple[str, ...],
        profile_index: dict[str, int],
        active_profile_ids: np.ndarray,
        profile_L: np.ndarray,
        active_slot_capacity: int | None = None,
        active_coeff_capacity: int | None = None,
    ) -> None:
        """Allocate profile-stage runtime memory and profile-slot metadata."""

        n_profiles = len(profile_names)
        n_active = int(active_profile_ids.size)
        slot_capacity = n_profiles if active_slot_capacity is None else int(active_slot_capacity)
        coeff_capacity = 0 if active_coeff_capacity is None else int(active_coeff_capacity)
        if active_coeff_capacity is None and n_active > 0:
            coeff_capacity = max(int(profile_L[int(p)]) + 1 for p in active_profile_ids)
        if slot_capacity < n_active:
            raise ValueError("active profile capacity is smaller than current active profile count")

        self.profile_names = tuple(profile_names)
        self.profile_index = dict(profile_index)
        self.profile_fields = np.empty((n_profiles, 3, nr), dtype=np.float64)
        self.profile_rp_fields = np.empty((n_profiles, 3, nr), dtype=np.float64)
        self.profile_env_fields = np.empty((n_profiles, 3, nr), dtype=np.float64)
        self._active_profile_ids_buffer = np.empty(slot_capacity, dtype=np.int64)
        self._active_offsets_buffer = np.empty(slot_capacity, dtype=np.float64)
        self._active_scales_buffer = np.empty(slot_capacity, dtype=np.float64)
        self._active_amplitude_powers_buffer = np.empty(slot_capacity, dtype=np.float64)
        self._active_lengths_buffer = np.empty(slot_capacity, dtype=np.int64)
        self._active_coeff_index_rows_buffer = np.full(
            (slot_capacity, coeff_capacity),
            -1,
            dtype=np.int64,
        )
        self.configure_active_metadata(active_profile_ids)

        self.c_family_fields = np.empty((m_max + 1, 3, nr), dtype=np.float64)
        self.s_family_fields = np.zeros((m_max + 1, 3, nr), dtype=np.float64)
        self.c_family_base_fields = np.zeros((m_max + 1, 3, nr), dtype=np.float64)
        self.s_family_base_fields = np.zeros((m_max + 1, 3, nr), dtype=np.float64)

        self.c_family_source_profile_ids = np.full(m_max + 1, -1, dtype=np.int64)
        self.s_family_source_profile_ids = np.full(m_max + 1, -1, dtype=np.int64)
        for order in range(m_max + 1):
            c_name = f"c{order}"
            if c_name in profile_index:
                self.c_family_source_profile_ids[order] = profile_index[c_name]
            if order == 0:
                # s0 is structurally absent; -1 keeps family update code on the
                # base zero field for that row.
                continue
            s_name = f"s{order}"
            if s_name in profile_index:
                self.s_family_source_profile_ids[order] = profile_index[s_name]

    def configure_active_metadata(self, active_profile_ids: np.ndarray) -> None:
        """Expose active metadata views backed by capacity-sized buffers."""

        active_ids = np.asarray(active_profile_ids, dtype=np.int64)
        n_active = int(active_ids.size)
        if n_active > self._active_profile_ids_buffer.size:
            raise ValueError("active profile count exceeds workspace capacity")
        self._active_profile_ids_buffer[:n_active] = active_ids
        self.active_profile_ids = self._active_profile_ids_buffer[:n_active]
        self.active_offsets = self._active_offsets_buffer[:n_active]
        self.active_scales = self._active_scales_buffer[:n_active]
        self.active_amplitude_powers = self._active_amplitude_powers_buffer[:n_active]
        self.active_lengths = self._active_lengths_buffer[:n_active]
        self.active_coeff_index_rows = self._active_coeff_index_rows_buffer[:n_active]
        self.active_lengths.fill(0)
        self.active_coeff_index_rows.fill(-1)

    def refresh_profile_slot(
        self,
        *,
        profile_id: int,
        grid_workspace: GridWorkspace,
        offset: float,
        scale: float,
        power: int,
        envelope_power: int,
        amplitude_power: float,
        coeff: np.ndarray | None,
    ) -> None:
        """Refresh one workspace-owned profile slot from flat profile metadata."""

        p = int(profile_id)
        self.profile_rp_fields.flags.writeable = True
        self.profile_env_fields.flags.writeable = True
        rp_fields = self.profile_rp_fields[p]
        env_fields = self.profile_env_fields[p]
        # Power/envelope terms depend only on static profile shape and grid, not
        # on packed coefficients.  Refresh them when the profile spec changes.
        _fill_power_terms(rp_fields, grid_workspace.rho, int(power))
        _fill_envelope_terms(
            env_fields,
            grid_workspace.rho,
            grid_workspace.rho_powers[2],
            grid_workspace.y,
            int(envelope_power),
        )
        self.profile_rp_fields.flags.writeable = False
        self.profile_env_fields.flags.writeable = False
        # Passive and active profiles both materialize through the same helper so
        # derivative row semantics stay identical.
        _fill_profile_outputs(
            self.profile_fields[p],
            grid_workspace.T,
            grid_workspace.T_r,
            grid_workspace.T_rr,
            rp_fields,
            env_fields,
            float(offset),
            coeff,
            float(scale),
            float(amplitude_power),
        )

    def refresh_profile_fields(
        self,
        *,
        profile_id: int,
        offset: float,
        scale: float,
        amplitude_power: float,
        coeff: np.ndarray | None,
        grid_workspace: GridWorkspace,
    ) -> None:
        """Refresh one workspace-owned value/derivative field set from existing auxiliary fields."""

        p = int(profile_id)
        _fill_profile_outputs(
            self.profile_fields[p],
            grid_workspace.T,
            grid_workspace.T_r,
            grid_workspace.T_rr,
            self.profile_rp_fields[p],
            self.profile_env_fields[p],
            float(offset),
            coeff,
            float(scale),
            float(amplitude_power),
        )

    def profile_id_for(self, name: str) -> int:
        """Return the stable plan profile id for ``name``."""

        try:
            return int(self.profile_index[name])
        except KeyError as exc:
            raise KeyError(f"Unknown profile name {name!r}") from exc

    def fields_for(self, name: str) -> np.ndarray:
        """Return workspace-owned ``(3, Nr)`` fields for ``name``."""

        return self.profile_fields[self.profile_id_for(name)]

    def values_for(self, name: str) -> np.ndarray:
        """Return workspace-owned value row for ``name``."""

        return self.fields_for(name)[PROFILE_VALUE]

    def radial_derivative_for(self, name: str) -> np.ndarray:
        """Return workspace-owned first radial derivative row for ``name``."""

        return self.fields_for(name)[PROFILE_R]

    def radial_second_derivative_for(self, name: str) -> np.ndarray:
        """Return workspace-owned second radial derivative row for ``name``."""

        return self.fields_for(name)[PROFILE_RR]

    def has_fields_for(self, name: str) -> bool:
        """Return whether a named profile has a workspace field slot."""

        return name in self.profile_index

    def active_slot_for_profile_id(self, profile_id: int) -> int:
        """Return active slot for ``profile_id`` or ``-1`` when fixed/inactive."""

        p = int(profile_id)
        for slot, active_profile_id in enumerate(self.active_profile_ids):
            if int(active_profile_id) == p:
                return int(slot)
        return -1

    def residual_block_lengths(self) -> np.ndarray:
        """Return a copy of active residual block lengths for solver normalization."""

        return self.active_lengths.copy()

    def active_profile_blocks(self) -> tuple[tuple[int, str, np.ndarray, float, float], ...]:
        """Return copy-based packed-profile metadata for solver scaling."""

        blocks: list[tuple[int, str, np.ndarray, float, float]] = []
        for slot, profile_id in enumerate(self.active_profile_ids):
            length = int(self.active_lengths[slot])
            if length <= 0:
                continue
            p = int(profile_id)
            profile_scale = float(self.active_scales[slot])
            amplitude_power = float(self.active_amplitude_powers[slot])
            if amplitude_power != 1.0 and profile_scale != 0.0:
                # Solver x-scale is a coefficient-space conditioning hint.  When
                # the physical profile output applies an amplitude power, undo it
                # so F coefficients still scale like the F**2 amplitude they own.
                profile_scale = abs(profile_scale) ** (1.0 / amplitude_power)
            blocks.append(
                (
                    p,
                    self.profile_names[p],
                    self.active_coeff_index_rows[slot, :length].copy(),
                    float(self.active_offsets[slot]),
                    profile_scale,
                )
            )
        return tuple(blocks)

def _fill_profile_outputs(
    u_fields: np.ndarray,
    T: np.ndarray,
    T_r: np.ndarray,
    T_rr: np.ndarray,
    rp_fields: np.ndarray,
    env_fields: np.ndarray,
    offset: float,
    coeff: np.ndarray | None,
    scale: float,
    amplitude_power: float,
) -> None:
    """Refresh one profile field set from coefficients."""

    update_profile(
        u_fields,
        T,
        T_r,
        T_rr,
        rp_fields,
        env_fields,
        offset,
        coeff,
        amplitude_power,
    )
    if scale != 1.0:
        # Scale is applied uniformly to value and derivative rows after any
        # amplitude transform; for F this restores physical F units.
        np.multiply(u_fields, scale, out=u_fields)


def _fill_power_terms(out: np.ndarray, rho: np.ndarray, power: int) -> None:
    power = int(power)
    if power == 0:
        out[0].fill(1.0)
        out[1].fill(0.0)
        out[2].fill(0.0)
        return

    out[0] = rho**power
    out[1] = power * rho ** (power - 1)
    if power == 1:
        # Avoid rho**-1 at the axis; the second derivative is analytically zero.
        out[2].fill(0.0)
    else:
        out[2] = power * (power - 1) * rho ** (power - 2)


def _fill_envelope_terms(
    out: np.ndarray,
    rho: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    envelope_power: int,
) -> None:
    envelope_power = int(envelope_power)
    if envelope_power == 0:
        out[0].fill(1.0)
        out[1].fill(0.0)
        out[2].fill(0.0)
        return

    if envelope_power == 1:
        # y is the standard edge envelope 1-rho**2.
        out[0] = y
        out[1] = -2.0 * rho
        out[2].fill(-2.0)
        return

    out[0] = y**envelope_power
    out[1] = -2.0 * envelope_power * rho * y ** (envelope_power - 1)
    out[2] = (
        -2.0 * envelope_power * y ** (envelope_power - 1)
        + 4.0 * envelope_power * (envelope_power - 1) * rho2 * y ** (envelope_power - 2)
    )
