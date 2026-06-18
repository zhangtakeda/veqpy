"""Private JAX compile-cache shell."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from veqpy.engine.jax.residual import fused_residual_pf_rho_grid
from veqpy.engine.jax.snapshot import fused_snapshot_pf_rho_grid
from veqpy.engine.jax.state import JaxStaticSpec


@dataclass(frozen=True, slots=True)
class JaxCompileKey:
    """Cache key that keeps residual and snapshot executable signatures separate."""

    spec: JaxStaticSpec
    execution_kind: str


@dataclass(slots=True)
class JaxCompileCache:
    """Small compile-cache container keyed by static spec and execution kind."""

    _cache: dict[JaxCompileKey, Any] = field(default_factory=dict)

    def get(self, spec: JaxStaticSpec, *, execution_kind: str = "residual") -> Any | None:
        return self._cache.get(JaxCompileKey(spec, execution_kind))

    def put(
        self,
        spec: JaxStaticSpec,
        compiled: Any,
        *,
        execution_kind: str = "residual",
    ) -> None:
        self._cache[JaxCompileKey(spec, execution_kind)] = compiled

    def __len__(self) -> int:
        return len(self._cache)


_GLOBAL_RESIDUAL_CACHE = JaxCompileCache()
_GLOBAL_SNAPSHOT_CACHE = JaxCompileCache()


def compile_residual_pf_rho_grid(jax_module: Any, spec: JaxStaticSpec) -> Any:
    """Compile or retrieve the PF/rho/grid residual graph for one static signature."""

    cached = _GLOBAL_RESIDUAL_CACHE.get(spec)
    if cached is not None:
        return cached

    def residual_fn(leaves: dict[str, Any], x: Any) -> Any:
        return fused_residual_pf_rho_grid(jax_module, leaves, spec, x)

    compiled = jax_module.jit(residual_fn, donate_argnums=(1,) if spec.donate_x else ())
    _GLOBAL_RESIDUAL_CACHE.put(spec, compiled)
    return compiled


def compile_snapshot_pf_rho_grid(jax_module: Any, spec: JaxStaticSpec) -> Any:
    """Compile or retrieve the PF/rho/grid explicit snapshot graph."""

    cached = _GLOBAL_SNAPSHOT_CACHE.get(spec, execution_kind="snapshot")
    if cached is not None:
        return cached

    def snapshot_fn(leaves: dict[str, Any], x: Any) -> dict[str, Any]:
        return fused_snapshot_pf_rho_grid(jax_module, leaves, spec, x)

    compiled = jax_module.jit(snapshot_fn)
    _GLOBAL_SNAPSHOT_CACHE.put(spec, compiled, execution_kind="snapshot")
    return compiled


def global_residual_cache_size() -> int:
    """Return the process-local residual compile cache size for tests."""

    return len(_GLOBAL_RESIDUAL_CACHE)


def global_snapshot_cache_size() -> int:
    """Return the process-local snapshot compile cache size for tests."""

    return len(_GLOBAL_SNAPSHOT_CACHE)
