"""
Module: base.registry

Role:
- Provide small decorator-backed registries.

Public API:
- Registry

Notes:
- This generalizes decorators shaped like ``read_serializer(*exts)``:
  calling the decorator with one or more keys returns a wrapper that stores
  the decorated function in a mapping table and returns the function unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Mapping
from types import MappingProxyType
from typing import Generic, TypeVar

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------


Key = TypeVar("Key", bound=Hashable)
Value = TypeVar("Value", bound=Callable)


class Registry(Generic[Key, Value]):
    """Callable decorator with an attached key -> function registry."""

    def __init__(
        self,
        key_type: type | tuple[type, ...],
        value_type: type,
    ) -> None:
        _validate_key_type(key_type)
        _validate_value_type(value_type)
        self._key_type = key_type
        self._value_type = value_type
        self._registry: dict[Key, Value] = {}

    @property
    def key_type(self) -> type | tuple[type, ...]:
        """Runtime type accepted for registration keys."""
        return self._key_type

    @property
    def value_type(self) -> type:
        """Runtime type accepted for registered values."""
        return self._value_type

    @property
    def registry(self) -> Mapping[Key, Value]:
        """Read-only view of the registry mapping."""
        return MappingProxyType(self._registry)

    def __call__(self, *keys: Key) -> Callable[[Value], Value]:
        if not keys:
            raise ValueError("At least one registry key is required")
        normalized_keys = tuple(self._normalize_key(key) for key in keys)
        for key in normalized_keys:
            if not isinstance(key, self.key_type):
                raise TypeError(
                    f"Registry key must be {_type_name(self.key_type)}, "
                    f"got {type(key).__name__}: {key!r}"
                )

        def wrapper(func: Value) -> Value:
            if not isinstance(func, self.value_type):
                raise TypeError(
                    f"Registry value must be {_type_name(self.value_type)}, "
                    f"got {type(func).__name__}: {func!r}"
                )
            for key in normalized_keys:
                self._registry[key] = func
            return func

        return wrapper

    def __contains__(self, key: object) -> bool:
        return self._normalize_key(key) in self._registry

    def __getitem__(self, key: Key) -> Value:
        return self._registry[self._normalize_key(key)]

    def __iter__(self) -> Iterator[Key]:
        return iter(self._registry)

    def _normalize_key(self, key: object) -> object:
        if isinstance(key, str):
            return key.lower()
        return key


# -----------------------------------------------------------------------------
# Private implementation
# -----------------------------------------------------------------------------


def _validate_key_type(value: type | tuple[type, ...]) -> None:
    if isinstance(value, type):
        return
    if isinstance(value, tuple) and value and all(isinstance(item, type) for item in value):
        return
    raise TypeError("Registry key_type must be a type or a non-empty tuple of types")


def _validate_value_type(value: type) -> None:
    if isinstance(value, type):
        return
    raise TypeError("Registry value_type must be a type")


def _type_name(value: type | tuple[type, ...]) -> str:
    if isinstance(value, tuple):
        return " | ".join(item.__name__ for item in value)
    return value.__name__
