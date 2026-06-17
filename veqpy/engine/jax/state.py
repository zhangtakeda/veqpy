"""Private JAX backend state containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class JaxStaticSpec:
    """Hashable static signature for JAX compile-cache keys."""

    route_key: tuple[str, str, str]
    nr: int
    nt: int
    k_max: int
    l_max: int
    m_max: int
    x_size: int
    profile_names: tuple[str, ...]
    active_profile_ids: tuple[int, ...]
    active_lengths: tuple[int, ...]
    residual_block_codes: tuple[int, ...]
    residual_block_orders: tuple[int, ...]
    residual_block_radial_powers: tuple[int, ...]
    active_amplitude_powers: tuple[float, ...] = ()
    c_effective_order: int = 0
    s_effective_order: int = 0
    n_axis_fix: int = 0
    has_Ip: bool = False
    has_beta: bool = False
    enable_x64: bool = True
    donate_x: bool = False


@dataclass(frozen=True, slots=True)
class JaxDeviceState:
    """Private device-state leaf container.

    Leaves are typed as ``Any`` so importing this module never requires JAX.
    """

    leaves: dict[str, Any]


@dataclass(slots=True)
class JaxRuntime:
    """Private JAX runtime bundle for one operator static signature."""

    static_spec: JaxStaticSpec
    device_state: JaxDeviceState
    compiled_residual: Any | None = None
