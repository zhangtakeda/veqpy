"""Private JAX residual compile-cache shell."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from veqpy.engine.jax.residual import fused_residual_pf_rho_grid
from veqpy.engine.jax.state import JaxStaticSpec


@dataclass(slots=True)
class JaxCompileCache:
    """Small compile-cache container keyed by ``JaxStaticSpec``."""

    _cache: dict[JaxStaticSpec, Any] = field(default_factory=dict)

    def get(self, spec: JaxStaticSpec) -> Any | None:
        return self._cache.get(spec)

    def put(self, spec: JaxStaticSpec, compiled: Any) -> None:
        self._cache[spec] = compiled

    def __len__(self) -> int:
        return len(self._cache)


_GLOBAL_RESIDUAL_CACHE = JaxCompileCache()


def compile_residual_pf_rho_grid(jax_module: Any, spec: JaxStaticSpec) -> Any:
    """Compile or retrieve the PF/rho/grid residual graph for one static signature."""

    cached = _GLOBAL_RESIDUAL_CACHE.get(spec)
    if cached is not None:
        return cached

    def residual_fn(leaves: dict[str, Any], x: Any) -> tuple[Any, dict[str, Any]]:
        return fused_residual_pf_rho_grid(jax_module, leaves, spec, x)

    compiled = jax_module.jit(residual_fn, donate_argnums=(1,) if spec.donate_x else ())
    _GLOBAL_RESIDUAL_CACHE.put(spec, compiled)
    return compiled


def global_residual_cache_size() -> int:
    """Return the process-local residual compile cache size for tests."""

    return len(_GLOBAL_RESIDUAL_CACHE)
