"""
Module: operator.packed_layout

Role:
- Build packed layout and profile metadata.
- Encode and decode packed state vectors at the operator boundary.
- Provide packed-state shape validation helpers.

Public API:
- PACKED_PROFILE_FAMILY_ORDER
- PACKED_LAYOUT_PROFILE_FIRST
- INTERLEAVE_SHAPE_COEFFS_BY_ORDER
- get_prefix_profile_names
- build_fourier_profile_names
- build_shape_profile_names
- build_profile_names
- build_profile_index
- build_profile_layout
- build_active_profile_metadata
- packed_size
- validate_packed_state
- encode_packed_state
- decode_packed_blocks
- coeff_array_from_list

Notes:
- Profile family ordering and residual block metadata are declared here.
- This module defines packed layout rules.
- It also handles state codecs for the same packed layout.
- Does not own numerical kernels, residual assembly, or solver iteration control.
- Treat this file as the single source of truth for both packed x ordering and
  residual block ordering.  Engine kernels consume the integer metadata built
  here and should not infer profile meaning from names.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

ProfileCoeffValue = list[float] | np.ndarray | int

RESIDUAL_BLOCK_CODE_BY_NAME = {
    "h": 0,
    "v": 1,
    "k": 2,
    "c0": 3,
    "c_family": 4,
    "s_family": 5,
    "psin": 6,
    "F": 7,
}

# Family order determines both model-facing coefficient dictionaries and the
# row order of layout metadata arrays.  Reordering it is a file-format/runtime
# compatibility change, not a local implementation detail.
PACKED_PROFILE_FAMILY_ORDER = ("h", "v", "k", "c0", "c", "s", "psin", "F")
PREFIX_PROFILE_FAMILIES = ("psin", "F")
SHAPE_PROFILE_FAMILIES = ("h", "v", "k", "c0", "c", "s")
ALL_PROFILE_FAMILIES = SHAPE_PROFILE_FAMILIES + PREFIX_PROFILE_FAMILIES

PROFILE_STATIC_KWARGS: dict[str, dict[str, float | int]] = {
    "psin": {"power": 2},
    # F coefficients parameterize the dimensionless F**2 amplitude.  The generic
    # profile evaluator takes the square root before source kernels consume F.
    "F": {"envelope_power": 1, "amplitude_power": 0.5},
}
PROFILE_OFFSET_SPECS: dict[str, float | str] = {
    "h": 0.0,
    "v": 0.0,
    "k": "ka",
    "psin": 1.0,
    "F": 1.0,
}


def validate_profile_family_order(
    family_order: tuple[str, ...] = PACKED_PROFILE_FAMILY_ORDER,
) -> tuple[str, ...]:
    """Validate and return the canonical packed profile-family order."""
    family_order = tuple(family_order)
    if len(family_order) != len(ALL_PROFILE_FAMILIES) or set(family_order) != set(
        ALL_PROFILE_FAMILIES
    ):
        raise ValueError(f"Invalid PACKED_PROFILE_FAMILY_ORDER {family_order!r}")
    return family_order


def get_prefix_profile_names(
    family_order: tuple[str, ...] = PACKED_PROFILE_FAMILY_ORDER,
) -> tuple[str, ...]:
    """Return non-shape prefix profile families in packed-layout order."""
    return tuple(
        family
        for family in validate_profile_family_order(family_order)
        if family in PREFIX_PROFILE_FAMILIES
    )


def expand_profile_family(family: str, M_max: int) -> tuple[str, ...]:
    """Expand one profile family token into concrete profile names."""
    if family == "psin":
        return ("psin",)
    if family == "F":
        return ("F",)
    if family == "h":
        return ("h",)
    if family == "v":
        return ("v",)
    if family == "k":
        return ("k",)
    if family == "c0":
        return ("c0",)
    if family == "c":
        return tuple(f"c{k}" for k in range(1, M_max + 1))
    if family == "s":
        return tuple(f"s{k}" for k in range(1, M_max + 1))
    raise KeyError(f"Unknown profile family {family!r}")


def build_fourier_profile_names(
    M_max: int,
    family_order: tuple[str, ...] = PACKED_PROFILE_FAMILY_ORDER,
) -> tuple[str, ...]:
    """Return concrete Fourier profile names up to order ``M_max``."""
    M_max = int(M_max)
    if M_max < 0:
        raise ValueError(f"M_max must be non-negative, got {M_max}")

    fourier_names: list[str] = []
    for family in validate_profile_family_order(family_order):
        if family not in ("c0", "c", "s"):
            continue
        fourier_names.extend(expand_profile_family(family, M_max))
    return tuple(fourier_names)


def build_shape_profile_names(
    M_max: int,
    family_order: tuple[str, ...] = PACKED_PROFILE_FAMILY_ORDER,
) -> tuple[str, ...]:
    """Return shape-profile names, excluding prefix/source profiles."""
    shape_profile_names: list[str] = []
    for family in validate_profile_family_order(family_order):
        if family in PREFIX_PROFILE_FAMILIES:
            continue
        shape_profile_names.extend(expand_profile_family(family, int(M_max)))
    return tuple(shape_profile_names)


def build_profile_names(
    M_max: int,
    family_order: tuple[str, ...] = PACKED_PROFILE_FAMILY_ORDER,
) -> tuple[str, ...]:
    """Return all concrete profile names in packed-layout order."""
    M_max = int(M_max)
    profile_names: list[str] = []
    for family in validate_profile_family_order(family_order):
        profile_names.extend(expand_profile_family(family, M_max))
    return tuple(profile_names)


def _decode_residual_block_code(name: str) -> tuple[int, int]:
    # c/s residual blocks share kernels across Fourier orders.  The returned
    # order is a kernel argument; the code selects the projection formula.
    if name.startswith("c") and name[1:].isdigit():
        order = int(name[1:])
        if order == 0:
            return (RESIDUAL_BLOCK_CODE_BY_NAME["c0"], 0)
        return (RESIDUAL_BLOCK_CODE_BY_NAME["c_family"], order)
    if name.startswith("s") and name[1:].isdigit():
        order = int(name[1:])
        if order == 0:
            raise KeyError("s0 is not a valid residual block")
        return (RESIDUAL_BLOCK_CODE_BY_NAME["s_family"], order)
    try:
        return (RESIDUAL_BLOCK_CODE_BY_NAME[name], 0)
    except KeyError as exc:
        supported = ", ".join(RESIDUAL_BLOCK_CODE_BY_NAME)
        raise KeyError(f"Unknown residual block {name!r}. Supported blocks: {supported}") from exc


def build_residual_block_metadata(profile_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Build residual block codes and Fourier orders for profile names."""
    block_codes = np.empty(len(profile_names), dtype=np.int64)
    block_orders = np.zeros(len(profile_names), dtype=np.int64)
    for i, name in enumerate(profile_names):
        block_codes[i], block_orders[i] = _decode_residual_block_code(name)
    return block_codes, block_orders


def build_residual_block_radial_powers(
    profile_names: tuple[str, ...],
    *,
    K_values: np.ndarray,
) -> np.ndarray:
    """Map profile names to residual radial powers used by Fourier blocks."""
    radial_powers = np.zeros(len(profile_names), dtype=np.int64)
    for i, name in enumerate(profile_names):
        if name.startswith(("c", "s")) and name[1:].isdigit():
            order = int(name[1:])
            if order < K_values.size:
                radial_powers[i] = int(K_values[order])
    return radial_powers


# Global packed-vector ordering switch.
# True  -> profile/name-first: h[0:L], v[0:L], ..., psin[0:L], F[0:L].
# False -> degree-first: all active profile 0th coefficients, then 1st coefficients, etc.
PACKED_LAYOUT_PROFILE_FIRST = False

# Backwards-readable alias for the previous degree-first switch name.
INTERLEAVE_SHAPE_COEFFS_BY_ORDER = not PACKED_LAYOUT_PROFILE_FIRST


def build_profile_index(profile_names: tuple[str, ...]) -> dict[str, int]:
    """Build a name-to-row lookup for packed profile metadata arrays."""
    return {name: i for i, name in enumerate(profile_names)}


def build_profile_layout(
    profile_coeffs: dict[str, ProfileCoeffValue | None],
    *,
    profile_names: tuple[str, ...],
    prefix_profile_names: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the packed state layout from a profile-coefficient dictionary."""
    profile_names = tuple(profile_names)
    if prefix_profile_names is None:
        prefix_profile_names = get_prefix_profile_names()
    prefix_profile_names = tuple(prefix_profile_names)
    profile_index = build_profile_index(profile_names)
    profile_count = len(profile_names)
    profile_L = np.full(profile_count, -1, dtype=np.int64)

    unknown = set(profile_coeffs) - set(profile_index)
    if unknown:
        raise KeyError(f"Unknown profile names: {sorted(unknown)}")

    for name, coeff in profile_coeffs.items():
        if coeff is None:
            continue
        profile_L[profile_index[name]] = coeff_array_from_list(name, coeff).size - 1

    # profile_L == -1 marks a passive profile.  Active profiles get a row in
    # coeff_index, with -1 cells left wherever that profile has no coefficient
    # at a higher degree.
    max_L = int(np.max(profile_L))
    if max_L < 0:
        raise ValueError("At least one active profile is required")

    coeff_index = -np.ones((profile_count, max_L + 1), dtype=np.int64)
    order_offsets = np.full(max_L + 2, -1, dtype=np.int64)
    # coeff_index[p, k] is the only supported map from (profile, degree) to x.
    # Its -1 padding must never be dereferenced; active_lengths guards hot loops.

    x_pos = 0
    if not PACKED_LAYOUT_PROFILE_FIRST:
        # Degree-first ordering keeps same-degree coefficients adjacent across
        # profile families, which improves the current fused residual workload.
        for k in range(max_L + 1):
            order_offsets[k] = x_pos
            for name in profile_names:
                p = profile_index[name]
                if profile_L[p] >= k:
                    coeff_index[p, k] = x_pos
                    x_pos += 1
    else:
        # Profile-first remains supported for codecs and benchmarks, even though
        # it is not the default runtime ordering.
        for name in profile_names:
            p = profile_index[name]
            L = int(profile_L[p])
            if L < 0:
                continue
            for k in range(L + 1):
                if order_offsets[k] < 0:
                    order_offsets[k] = x_pos
                coeff_index[p, k] = x_pos
                x_pos += 1
        for k in range(max_L + 1):
            if order_offsets[k] < 0:
                order_offsets[k] = x_pos
    order_offsets[max_L + 1] = x_pos
    # order_offsets[k] gives the first packed position for degree k in the
    # degree-first layout.  For profile-first it is retained for compatibility
    # checks and monotone layout fingerprints.

    return profile_L, coeff_index, order_offsets


def build_active_profile_metadata(
    profile_L: np.ndarray,
    *,
    profile_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract active profile metadata from the profile-degree vector."""
    profile_L = np.asarray(profile_L, dtype=np.int64)
    expected_size = len(tuple(profile_names))
    if profile_L.ndim != 1 or profile_L.shape[0] != expected_size:
        raise ValueError(
            f"Expected profile_L to have shape ({expected_size},), got {profile_L.shape}"
        )

    active_profile_mask = profile_L >= 0
    active_profile_ids = np.flatnonzero(active_profile_mask).astype(np.int64, copy=False)
    return active_profile_mask, active_profile_ids


def packed_size(coeff_index: np.ndarray) -> int:
    """Return the effective packed-vector length."""
    if coeff_index.size == 0:
        return 0
    return int(np.count_nonzero(coeff_index >= 0))


def validate_packed_state(x: np.ndarray, coeff_index: np.ndarray) -> np.ndarray:
    """Validate packed state vector shape and return a float64 view."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected x to be 1D, got {x.shape}")

    x_size = packed_size(coeff_index)
    if x.shape[0] != x_size:
        raise ValueError(f"Expected x to have shape ({x_size},), got {x.shape}")
    return x


def encode_packed_state(
    profile_coeffs: dict[str, ProfileCoeffValue | None],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    *,
    profile_names: tuple[str, ...],
) -> np.ndarray:
    """Encode a profile-coefficient dictionary into a packed state vector."""
    x = np.empty(packed_size(coeff_index), dtype=np.float64)

    for p, name in enumerate(profile_names):
        L = int(profile_L[p])
        coeff = profile_coeffs.get(name)

        if L < 0:
            # Passive profiles are reconstructed from offsets/static coeffs; they
            # intentionally occupy no packed x positions.
            continue
        if coeff is None:
            raise ValueError(f"{name} is active in layout but coeff is None")

        coeff_arr = coeff_array_from_list(name, coeff)
        if coeff_arr.size != L + 1:
            raise ValueError(
                f"{name} coeff shape mismatch: expected {(L + 1,)}, got {coeff_arr.shape}"
            )

        for k in range(L + 1):
            # Always encode through coeff_index so profile-first and degree-first
            # layouts share the same codec.
            x[coeff_index[p, k]] = coeff_arr[k]

    return x


def decode_packed_blocks(
    x: np.ndarray,
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    *,
    profile_names: tuple[str, ...],
) -> tuple[np.ndarray | None, ...]:
    """Decode a packed state vector into per-profile coefficient copies."""
    x = validate_packed_state(x, coeff_index)

    blocks: list[np.ndarray | None] = []
    for p, _ in enumerate(profile_names):
        L = int(profile_L[p])
        if L < 0:
            # Preserve passive profile markers in the decoded tuple; callers use
            # None to distinguish absent packed state from zero coefficients.
            blocks.append(None)
        else:
            block = np.empty(L + 1, dtype=np.float64)
            for k in range(L + 1):
                block[k] = x[coeff_index[p, k]]
            blocks.append(block)
    return tuple(blocks)


def coeff_array_from_list(name: str, coeff: ProfileCoeffValue) -> np.ndarray:
    """Convert profile coefficient input into a constrained one-dimensional array."""
    if isinstance(coeff, bool):
        raise TypeError(f"{name} coeff length indicator must be an integer, got bool")
    if isinstance(coeff, Integral):
        length = int(coeff)
        if length <= 0:
            raise ValueError(f"{name} coeff length indicator must be positive, got {coeff}")
        return np.zeros(length, dtype=np.float64)
    if not isinstance(coeff, (list, np.ndarray)):
        raise TypeError(f"Invalid {name} coeff type {type(coeff).__name__}")
    coeff_arr = np.asarray(coeff, dtype=np.float64)
    if coeff_arr.ndim != 1:
        raise ValueError(f"{name} coeff must be 1D, got {coeff_arr.shape}")
    if coeff_arr.size == 0:
        raise ValueError(f"{name} coeff must be non-empty or None")
    return coeff_arr
