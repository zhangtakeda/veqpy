"""Backend-neutral coercion for explicit nonlinear-solver initial states."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .types import KernelRecipe, KernelTopology, SolveResult

KernelInitial: TypeAlias = ArrayLike | Mapping[str, ArrayLike] | SolveResult


def materialize_initial_state(
    value: KernelInitial,
    topology: KernelTopology,
    recipe: KernelRecipe,
) -> NDArray[np.float64]:
    """Return one owned packed initial state in the recipe's layout."""

    if isinstance(value, SolveResult):
        packed = np.asarray(value.x, dtype=np.float64)
    elif isinstance(value, Mapping):
        packed = _pack_named_coefficients(value, topology, recipe)
    else:
        packed = np.asarray(value, dtype=np.float64)
    if packed.ndim != 1 or packed.shape != (topology.x_size,):
        raise ValueError(f"x0 must have shape ({topology.x_size},), got {packed.shape}")
    if not np.all(np.isfinite(packed)):
        raise ValueError("x0 must contain only finite values")
    return np.ascontiguousarray(packed, dtype=np.float64).copy()


def _pack_named_coefficients(
    coefficients: Mapping[str, ArrayLike],
    topology: KernelTopology,
    recipe: KernelRecipe,
) -> NDArray[np.float64]:
    active = dict(topology.active_profiles)
    supplied = set(coefficients)
    unknown = supplied - set(active)
    if unknown:
        raise KeyError(f"Unknown or inactive profile names in x0: {sorted(unknown)}")
    missing = set(active) - supplied
    if missing:
        raise ValueError(f"Missing active profile coefficients in x0: {sorted(missing)}")

    blocks: dict[str, NDArray[np.float64]] = {}
    for name, count in topology.active_profiles:
        block = np.asarray(coefficients[name], dtype=np.float64)
        if block.ndim != 1 or block.shape != (count,):
            raise ValueError(f"x0[{name!r}] must have shape ({count},), got {block.shape}")
        blocks[name] = block

    packed = np.empty(topology.x_size, dtype=np.float64)
    position = 0
    if recipe.layout_profile_first:
        for name, _ in topology.active_profiles:
            block = blocks[name]
            packed[position : position + block.size] = block
            position += block.size
    else:
        max_count = max(active.values(), default=0)
        for degree in range(max_count):
            for name, count in topology.active_profiles:
                if degree < count:
                    packed[position] = blocks[name][degree]
                    position += 1
    return packed
