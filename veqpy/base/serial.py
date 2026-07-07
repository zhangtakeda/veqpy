"""
Module: veqpy.base.serial

Role:
- Provide the shared persistence framework for public root-state objects.
- Handle JSON, Pickle, numpy arrays, nested ``Serial`` objects, and type-tagged
  reconstruction through one registry-backed path.

Public API:
- Serial
- read_serializer
- write_serializer

Design notes:
- Subclasses declare persistent fields through ``serial_attributes()``.  Dataclass
  subclasses can infer those fields from annotations, while non-dataclass models
  normally provide the schema explicitly.
- Files store independent root state, not derived caches.  Runtime workspaces,
  backend temporaries, residual buffers, and other hot-path arrays must stay out
  of ``serial_attributes()`` so serialization remains stable across internal
  performance changes.
- ``load`` creates a new object from a file.  ``read`` mutates an existing object
  from a compatible file payload.  ``write`` emits the root-state representation.
- ``Serial`` pairs with ``Reactive``: files persist the minimal state, and loaded
  objects rebuild derived quantities through formulas instead of replaying stale
  cache contents.
"""
from __future__ import annotations

import inspect
import os
import pickle
import types
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Literal, Self, get_args, get_origin, get_type_hints

import numpy as np
import orjson

from veqpy.base.registry import Registry

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------


serializer_handler = Callable[..., Any]
serializer_registry = Mapping[str, serializer_handler]

read_serializer: Registry[str, serializer_handler] = Registry(str, Callable)
write_serializer: Registry[str, serializer_handler] = Registry(str, Callable)
SERIAL_TYPE_REGISTRY: dict[str, type] = {}

READ_SERIALIZER_HANDLERS = read_serializer.registry
WRITE_SERIALIZER_HANDLERS = write_serializer.registry

ORJSON_LOADS = orjson.loads
ORJSON_DUMPS = orjson.dumps
ORJSON_WRITE_OPTIONS = orjson.OPT_SERIALIZE_NUMPY


class Serial:
    """Base class for unified serialization."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return
        SERIAL_TYPE_REGISTRY.setdefault(cls.__name__, cls)

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        """Return the serializable attribute schema for this concrete type.

        Subclasses may override this explicitly.  Dataclass subclasses can rely
        on field annotations as the default schema.
        """
        if is_dataclass(cls):
            return _dataclass_attribute_types(cls)
        raise NotImplementedError

    def copy(self) -> Self:
        """Return a deep copy that preserves nested serial/reactive state."""
        return deepcopy(self)

    @classmethod
    def load(cls, file: str, **kwargs) -> Self:
        """Deserialize a file into a new instance."""
        ext = _resolve_ext(file, READ_SERIALIZER_HANDLERS)

        if ext in ("json", "jsonl"):
            with open(file, "rb") as f:
                data = ORJSON_LOADS(f.read())
            instance = _json_to_python(data, cls)
            if not _check_type(instance, cls):
                raise TypeError(
                    f"Deserialized object is {type(instance).__name__}, expected {cls.__name__}"
                )
            return instance

        if ext in ("pkl", "pickle"):
            with open(file, "rb") as f:
                data = pickle.load(f)
            instance = _instantiate_serial(cls, data)
            if not _check_type(instance, cls):
                raise TypeError(
                    f"Deserialized object is {type(instance).__name__}, expected {cls.__name__}"
                )
            return instance

        instance = cls.__new__(cls)
        instance.read(file, **kwargs)
        return instance

    def read(self, file: str, func: serializer_handler | str | None = None, **kwargs) -> Self:
        """Read a file into the current instance."""
        if func is None:
            _dispatch("read", self, file, **kwargs)
        elif isinstance(func, str):
            getattr(self, func)(file, **kwargs)
        else:
            func(self, file, **kwargs)
        return self

    def write(self, file: str, func: serializer_handler | str | None = None, **kwargs) -> None:
        """Write the current instance to a file."""
        if func is None:
            _dispatch("write", self, file, **kwargs)
        elif isinstance(func, str):
            getattr(self, func)(file, **kwargs)
        else:
            func(self, file, **kwargs)

    def check(self) -> None:
        """Validate that serial attributes exist, match their types, and are non-empty."""
        spec = type(self).serial_attributes()
        cls = type(self)
        prop_names = {n for n, o in cls.__dict__.items() if isinstance(o, property)}
        actual = {key for key in spec if hasattr(self, key)} | prop_names

        missing = set(spec.keys()) - actual
        if missing:
            raise AttributeError(f"Missing attributes: {missing}")

        for key, expected in spec.items():
            value = getattr(self, key)

            if not _check_type(value, expected):
                raise TypeError(
                    f"Attribute '{key}': expected {_type_name(expected)}, "
                    f"got {type(value).__name__}"
                )

            if value is None and _allows_none(expected):
                continue

            if isinstance(value, np.ndarray):
                if value.size == 0:
                    raise ValueError(f"Attribute '{key}' is empty array")
            elif not isinstance(value, (int, float, bool, np.number)):
                if not value and value != "":
                    raise ValueError(f"Attribute '{key}' is empty")

    @read_serializer("json", "jsonl")
    def read_json(self, file: str) -> Self:
        """Read JSON content into an existing instance.

        Use ``Serial.load`` for frozen dataclasses.
        """
        with open(file, "rb") as f:
            data = ORJSON_LOADS(f.read())

        attrs_data = _unwrap_typed_dict(data, type(self).__name__)
        _restore_serial_fields(self, attrs_data, decoder=_json_to_python)
        return self

    @write_serializer("json", "jsonl")
    def write_json(self, file: str) -> None:
        """Write the current instance as typed JSON or JSONL-compatible content."""
        self.check()
        data = _python_to_json(self)
        with open(file, "wb") as f:
            f.write(ORJSON_DUMPS(data, option=ORJSON_WRITE_OPTIONS))

    @read_serializer("pkl", "pickle")
    def read_pickle(self, file: str) -> Self:
        """Read Pickle content into an existing mutable instance."""
        with open(file, "rb") as f:
            data = pickle.load(f)
        _restore_serial_fields(self, data, decoder=lambda value, _expected: value)
        return self

    @write_serializer("pkl", "pickle")
    def write_pickle(self, file: str) -> None:
        """Write the current instance as a Pickle payload."""
        self.check()
        data = {k: getattr(self, k) for k in type(self).serial_attributes()}
        with open(file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


# -----------------------------------------------------------------------------
# Private implementation
# -----------------------------------------------------------------------------

# Dispatch helpers


def _resolve_ext(file: str, registry: serializer_registry) -> str:
    """Resolve a file extension and verify that it exists in the registry."""
    suffix = Path(file).suffix.lstrip(".")
    if suffix in registry:
        return suffix
    for ext in registry:
        if file.endswith("." + ext):
            return ext
    raise ValueError(f"Unsupported file format: {file}")


def _dispatch(op: str, instance: Serial, file: str, **kwargs) -> None:
    registry = READ_SERIALIZER_HANDLERS if op == "read" else WRITE_SERIALIZER_HANDLERS
    if op == "read" and not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
    ext = _resolve_ext(file, registry)
    registry[ext](instance, file, **kwargs)


# Type helpers


def _normalize_union(t) -> tuple | None:
    """Normalize tuple and ``types.UnionType`` values to tuple members."""
    if isinstance(t, tuple):
        return t
    if isinstance(t, types.UnionType):
        return get_args(t)
    return None


def _allows_none(t) -> bool:
    members = _normalize_union(t)
    if members is None:
        return t is type(None)
    return type(None) in members


def _type_name(t) -> str:
    members = _normalize_union(t)
    if members is not None:
        return f"Union[{', '.join(_type_name(x) for x in members)}]"
    return getattr(t, "__name__", str(t))


def _check_type(value: Any, expected: type | tuple) -> bool:
    """Recursively check values against Union, list, tuple, Literal, and Serial types."""
    if expected is Any:
        return True

    members = _normalize_union(expected)
    if members is not None:
        return any(_check_type(value, t) for t in members)

    origin = get_origin(expected)
    if origin is list:
        if not isinstance(value, list):
            return False
        args = get_args(expected)
        if args and value:
            return all(_check_type(v, args[0]) for v in value)
        return True

    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        args = get_args(expected)
        if args:
            if len(args) != len(value):
                return False
            return all(_check_type(v, t) for v, t in zip(value, args))
        return True

    if origin is dict:
        if not isinstance(value, dict):
            return False
        args = get_args(expected)
        if len(args) == 2 and value:
            key_t, value_t = args
            return all(_check_type(k, key_t) and _check_type(v, value_t) for k, v in value.items())
        return True

    if origin is Literal:
        return value in get_args(expected)

    if expected is np.ndarray:
        return isinstance(value, np.ndarray)

    if expected in {int, float, str, bool}:
        return isinstance(value, (expected, np.generic))

    try:
        if isinstance(value, expected):
            return True
    except TypeError:
        pass

    vt = type(value)
    if vt.__name__ == expected.__name__ and getattr(vt, "__module__", None) == getattr(
        expected, "__module__", None
    ):
        return True

    try:
        return issubclass(vt, expected)
    except TypeError:
        return False


# JSON conversion helpers


def _unwrap_typed_dict(data: Any, expected_name: str) -> dict:
    """Remove the outer type tag when data is ``{TypeName: {attrs}}``."""
    if isinstance(data, dict) and len(data) == 1:
        key = next(iter(data))
        if key == expected_name:
            return data[key]
    return data if isinstance(data, dict) else {}


def _python_to_json(value: Any) -> Any:
    """Recursively convert a Python object into a JSON-serializable structure."""
    if hasattr(value, "serial_attributes"):
        spec = type(value).serial_attributes()
        return {type(value).__name__: {k: _python_to_json(getattr(value, k)) for k in spec}}

    if is_dataclass(value) and not isinstance(value, type):
        spec = _dataclass_attribute_types(type(value))
        return {type(value).__name__: {k: _python_to_json(getattr(value, k)) for k in spec}}

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, list):
        return [_python_to_json(v) for v in value]

    if isinstance(value, tuple):
        return [_python_to_json(v) for v in value]

    if isinstance(value, dict):
        return {k: _python_to_json(v) for k, v in value.items()}

    return value


def _json_to_python(data: Any, expected: type | tuple) -> Any:
    """Recursively convert JSON data into the expected Python type."""
    if expected is Any:
        return data

    members = _normalize_union(expected)
    if members is not None:
        return _json_to_python_union(data, members)

    if isinstance(data, dict) and len(data) == 1:
        result = _try_instantiate_from_tagged_dict(data)
        if result is not None:
            return result
        expected_name = getattr(expected, "__name__", None)
        if expected_name == next(iter(data)):
            # Support both nested typed payloads and already-unwrapped attribute
            # dictionaries; older files may contain either representation.
            data = data[expected_name]

    origin = get_origin(expected)
    if origin is list and isinstance(data, list):
        inner = get_args(expected)[0] if get_args(expected) else Any
        return [_json_to_python(item, inner) for item in data]

    if origin is tuple and isinstance(data, (list, tuple)):
        args = get_args(expected)
        if args:
            return tuple(_json_to_python(item, t) for item, t in zip(data, args))
        return tuple(data)

    if origin is dict and isinstance(data, dict):
        args = get_args(expected)
        if len(args) == 2:
            key_t, value_t = args
            return {
                _json_to_python(key, key_t): _json_to_python(value, value_t)
                for key, value in data.items()
            }
        return dict(data)

    if origin is Literal:
        return data

    if expected is np.ndarray:
        return np.array(data) if not isinstance(data, np.ndarray) else data

    if expected in {int, float, str, bool}:
        return data.item() if isinstance(data, np.generic) else data

    if hasattr(expected, "serial_attributes") and isinstance(data, dict):
        return _instantiate_serial(expected, data)

    if inspect.isclass(expected) and is_dataclass(expected) and isinstance(data, dict):
        return _instantiate_dataclass(expected, data)

    return data


def _json_to_python_union(data: Any, members: tuple) -> Any:
    """Convert JSON data for a Union type."""
    if isinstance(data, dict) and len(data) == 1:
        key = next(iter(data))
        for t in members:
            if getattr(t, "__name__", None) == key:
                return _json_to_python(data, t)

    for t in members:
        if t is np.ndarray and isinstance(data, list):
            return np.array(data)
        if t in {int, float, str, bool} and isinstance(data, (int, float, str, bool)):
            return t(data)

    for t in members:
        try:
            candidate = _json_to_python(data, t)
            if _check_type(candidate, t):
                return candidate
        except (TypeError, ValueError, AttributeError):
            continue

    return data


# Object construction helpers


def _try_instantiate_from_tagged_dict(data: dict) -> Any | None:
    """Try to instantiate an object from a ``{"TypeName": {attrs}}`` structure."""
    type_name = next(iter(data))
    cls = SERIAL_TYPE_REGISTRY.get(type_name)
    if cls is None:
        return None

    attrs = data[type_name]
    # The type registry is intentionally name-based so JSON payloads do not need
    # to persist import paths that can change across package refactors.
    if hasattr(cls, "serial_attributes"):
        return _instantiate_serial(cls, attrs)

    return cls(attrs)


def _instantiate_serial(cls: type, attrs: dict) -> Any:
    """Instantiate a Serial subclass from an attribute dictionary."""
    spec = cls.serial_attributes()
    field_values: dict[str, Any] = {}
    for key, expected in spec.items():
        field_values[key] = _json_to_python(attrs.get(key), expected)
    return _construct_object(cls, field_values)


def _instantiate_dataclass(cls: type, attrs: dict) -> Any:
    """Instantiate a dataclass object from an attribute dictionary."""

    spec = _dataclass_attribute_types(cls)
    field_values = {k: _json_to_python(attrs.get(k), t) for k, t in spec.items()}
    return _construct_object(cls, field_values)


def _construct_object(cls: type, field_values: dict[str, Any]) -> Any:
    try:
        return cls(**field_values)
    except TypeError:
        pass

    # Some serialized classes do not accept every serial attribute in
    # __init__; construct with the supported subset and restore the rest.
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        raise TypeError(f"Unable to instantiate {cls.__name__}: cannot inspect __init__")

    init_params = {
        name
        for name, p in sig.parameters.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    init_kwargs = {k: v for k, v in field_values.items() if k in init_params}

    try:
        instance = cls(**init_kwargs)
    except TypeError as exc:
        raise TypeError(f"Unable to instantiate {cls.__name__}: {exc}") from exc

    for key, value in field_values.items():
        if key in init_kwargs:
            continue
        desc = getattr(cls, key, None)
        if isinstance(desc, property) and desc.fset is None:
            continue
        _set_attribute(instance, key, value)

    return instance


def _dataclass_attribute_types(cls: type) -> dict[str, type]:
    hints = get_type_hints(cls)
    return {field.name: hints.get(field.name, Any) for field in dataclass_fields(cls)}


def _set_attribute(instance: Any, key: str, value: Any) -> None:
    try:
        setattr(instance, key, value)
    except (AttributeError, TypeError):
        object.__setattr__(instance, key, value)


def _restore_serial_fields(
    instance: Any,
    attrs: dict[str, Any],
    *,
    decoder: Callable[[Any, type | tuple], Any],
) -> None:
    for key, expected in type(instance).serial_attributes().items():
        desc = getattr(type(instance), key, None)
        if isinstance(desc, property) and desc.fset is None:
            continue
        if key in attrs:
            _set_attribute(instance, key, decoder(attrs[key], expected))
