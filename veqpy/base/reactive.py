"""Push-invalidated reactive properties for read-heavy model objects.

Subclasses declare independent roots explicitly; dependencies of derived
properties are inferred from their AST or declared with ``depends_on``. The
runtime moves dependency bookkeeping from reads to writes:

- a root write marks its transitive derived dependents dirty;
- a clean derived read is one bit test plus one list lookup;
- nested ``Reactive`` objects notify parents through weak subscriptions;
- derived values are recomputed lazily and keep the existing snapshot value
  semantics.

The implementation accepts native scalar/container values, arrays, nested
``Reactive`` objects, and externally owned snapshot objects. Mutable native
containers are frozen internally and exposed as fresh values of their original
built-in type. External snapshot objects are passed through unchanged and are
therefore expected not to mutate during the owning object's lifetime.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import weakref
from collections.abc import Callable
from copy import deepcopy
from types import MappingProxyType
from typing import Any

import numpy as np

_MISSING = object()
_SCALAR_TYPES = {type(None), bool, int, float, complex, str, bytes}
_PICKLE_ROOTS_KEY = "__reactive_roots__"
_PICKLE_EXTRAS_KEY = "__reactive_extras__"
_PICKLE_VERSION_KEY = "__reactive_state_version__"
_PICKLE_STATE_VERSION = 1
_PLAN_LIST = "list"
_PLAN_TUPLE = "tuple"
_PLAN_DICT = "dict"
_PLAN_SET = "set"
_PLAN_BYTEARRAY = "bytearray"


class Reactive:
    """Reactive cache with push invalidation and constant-time clean reads.

    Arrays are frozen on assignment. Mutable built-in containers are converted
    to immutable internal forms and reconstructed only when exposed. Only
    nested ``Reactive`` instances propagate internal updates. Initialization
    is idempotent so existing subclasses may assign roots before or after their
    ``super().__init__()`` call.
    """

    dependency_graph: dict[str, set[str]] = {}
    root_properties: set[str]
    _reactive_derived_properties: frozenset[str] = frozenset()
    _reactive_all_properties: frozenset[str] = frozenset()
    _reactive_indices: dict[str, int] = {}
    _reactive_root_indices: dict[str, int] = {}
    _reactive_dependent_masks: dict[str, int] = {}
    _reactive_all_dirty_mask: int = 0

    def __init__(self) -> None:
        if "_reactive_values" in self.__dict__:
            return
        object.__setattr__(
            self,
            "_reactive_values",
            [None] * len(self._reactive_derived_properties),
        )
        object.__setattr__(
            self,
            "_reactive_value_plans",
            [None] * len(self._reactive_derived_properties),
        )
        object.__setattr__(
            self,
            "_reactive_root_plans",
            [None] * len(self.root_properties),
        )
        object.__setattr__(self, "_reactive_dirty_mask", self._reactive_all_dirty_mask)
        object.__setattr__(self, "_reactive_revision", 0)
        object.__setattr__(self, "_reactive_observers", {})

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        roots = cls._validate_root_properties()
        dependency_graph = cls._build_dependency_graph(roots)
        reverse_adj = _build_reverse_adj(dependency_graph)
        _validate_dependency_graph(roots, dependency_graph, reverse_adj)

        derived_names = tuple(sorted(dependency_graph))
        cls.dependency_graph = dependency_graph
        cls._reactive_derived_properties = frozenset(derived_names)
        cls._reactive_all_properties = frozenset(roots | set(derived_names))
        cls._reactive_indices = {name: i for i, name in enumerate(derived_names)}
        cls._reactive_root_indices = {name: i for i, name in enumerate(sorted(roots))}
        cls._reactive_all_dirty_mask = (1 << len(derived_names)) - 1
        cls._reactive_dependent_masks = _build_dependent_masks(
            roots=roots,
            derived_names=derived_names,
            reverse_adj=reverse_adj,
        )

        cls._setup_root_properties(roots)
        cls._wrap_derived_properties(dependency_graph)

    def __deepcopy__(self, memo):
        cls = self.__class__
        cloned = cls.__new__(cls)
        memo[id(self)] = cloned
        Reactive.__init__(cloned)

        internal = {
            "_reactive_values",
            "_reactive_value_plans",
            "_reactive_root_plans",
            "_reactive_dirty_mask",
            "_reactive_revision",
            "_reactive_observers",
        }
        cached_roots = {f"cached_{name}" for name in cls.root_properties}
        for name, value in self.__dict__.items():
            if name not in internal and name not in cached_roots:
                object.__setattr__(cloned, name, deepcopy(value, memo))

        for root in sorted(cls.root_properties):
            if f"cached_{root}" in self.__dict__:
                setattr(cloned, root, deepcopy(getattr(self, root), memo))
        return cloned

    def __copy__(self):
        raise TypeError(f"{type(self).__name__} does not support shallow copy; use copy.deepcopy()")

    def __getstate__(self) -> dict[str, Any]:
        """Serialize authoritative roots, never caches or observer state."""

        roots = {
            name: getattr(self, name)
            for name in sorted(self.root_properties)
            if f"cached_{name}" in self.__dict__
        }
        internal = {
            "_reactive_values",
            "_reactive_value_plans",
            "_reactive_root_plans",
            "_reactive_dirty_mask",
            "_reactive_revision",
            "_reactive_observers",
        }
        cached_roots = {f"cached_{name}" for name in self.root_properties}
        extras = {
            name: value
            for name, value in self.__dict__.items()
            if name not in internal and name not in cached_roots
        }
        return {
            _PICKLE_VERSION_KEY: _PICKLE_STATE_VERSION,
            _PICKLE_ROOTS_KEY: roots,
            _PICKLE_EXTRAS_KEY: extras,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore roots and rebuild an empty derived cache plus subscriptions."""

        Reactive.__init__(self)
        if _PICKLE_ROOTS_KEY in state:
            roots = state[_PICKLE_ROOTS_KEY]
            extras = state.get(_PICKLE_EXTRAS_KEY, {})
        else:
            # Best-effort support for objects pickled from the pull-based
            # implementation, whose authoritative roots lived in cached_*.
            roots = {}
            extras = {
                name: value
                for name, value in state.items()
                if not name.startswith("cached_")
                and name not in {"cache", "_version", "_revision"}
            }
            for name in self.root_properties:
                cached_name = f"cached_{name}"
                if cached_name in state:
                    roots[name] = state[cached_name]
                elif name in state:
                    roots[name] = state[name]
        for name, value in extras.items():
            object.__setattr__(self, name, value)
        for name in sorted(self.root_properties):
            if name in roots:
                setattr(self, name, roots[name])

    def invalidate(self, *names: str) -> None:
        """Mark selected nodes, or every derived node, dirty."""

        if not names:
            for name, index in self._reactive_indices.items():
                value = self._reactive_values[index]
                if value is not None:
                    self._unsubscribe_nested(name, value)
                self._reactive_values[index] = None
                self._reactive_value_plans[index] = None
            self._reactive_dirty_mask = self._reactive_all_dirty_mask
        else:
            dirty = 0
            for name in names:
                if name not in self._reactive_all_properties:
                    raise KeyError(f"unknown Reactive node {name!r}")
                dirty |= self._reactive_dependent_masks.get(name, 0)
                index = self._reactive_indices.get(name)
                if index is not None:
                    dirty |= 1 << index
            self._reactive_dirty_mask |= dirty
        self._state_changed()

    @classmethod
    def _validate_root_properties(cls) -> set[str]:
        roots = getattr(cls, "root_properties", None)
        if roots is None:
            raise TypeError(f"{cls.__name__} must define root_properties explicitly")
        if not isinstance(roots, set) or any(not isinstance(name, str) for name in roots):
            raise TypeError(f"{cls.__name__}.root_properties must be a set[str]")
        if not roots:
            raise ValueError(f"{cls.__name__}.root_properties must not be empty")
        return roots

    @classmethod
    def _setup_root_properties(cls, roots: set[str]) -> None:
        for name in roots:
            attr = cls._find_property(name)
            if attr is None:
                cls._install_default_root_property(name)
            else:
                cls._wrap_existing_root_property(name, attr)

    @classmethod
    def _install_default_root_property(cls, name: str) -> None:
        cached_name = f"cached_{name}"
        root_slot = cls._reactive_root_indices[name]

        def fget(self):
            self._ensure_reactive_state()
            value = self.__dict__.get(cached_name)
            plan = self._reactive_root_plans[root_slot]
            return value if plan is None else _expose_reactive_value(value, plan)

        def fset(self, value):
            self._ensure_reactive_state()
            value = self._prepare_root_value(name, value)
            value, plan = _prepare_reactive_value(value, path=name)
            old = self.__dict__.get(cached_name, _MISSING)
            self._reject_nested_cycle(value, node=name)
            if old is not _MISSING:
                self._unsubscribe_nested(name, old)
            object.__setattr__(self, cached_name, value)
            self._reactive_root_plans[root_slot] = plan
            self._subscribe_nested(name, value)
            self._reactive_dirty_mask |= self._reactive_dependent_masks.get(name, 0)
            self._state_changed()

        fget._reactive_root_wrapped = True  # type: ignore[attr-defined]
        fset._reactive_root_wrapped = True  # type: ignore[attr-defined]
        setattr(cls, name, property(fget=fget, fset=fset))

    @classmethod
    def _wrap_existing_root_property(cls, name: str, attr: property) -> None:
        if getattr(attr.fget, "_reactive_root_wrapped", False) or getattr(
            attr.fset, "_reactive_root_wrapped", False
        ):
            return

        original_fget = attr.fget
        original_fset = attr.fset
        root_slot = cls._reactive_root_indices[name]

        def fget(self):
            self._ensure_reactive_state()
            if original_fget is None:
                raise AttributeError(f"unreadable attribute {name!r}")
            value = original_fget(self)
            plan = self._reactive_root_plans[root_slot]
            return value if plan is None else _expose_reactive_value(value, plan)

        if original_fset is None:
            fset = None
        else:

            def fset(self, value):
                self._ensure_reactive_state()
                old = original_fget(self) if original_fget is not None else _MISSING
                value = self._prepare_root_value(name, value)
                value, plan = _prepare_reactive_value(value, path=name)
                self._reject_nested_cycle(value, node=name)
                if old is not _MISSING:
                    self._unsubscribe_nested(name, old)
                original_fset(self, value)
                self._reactive_root_plans[root_slot] = plan
                self._subscribe_nested(name, value)
                self._reactive_dirty_mask |= self._reactive_dependent_masks.get(name, 0)
                self._state_changed()

            fset._reactive_root_wrapped = True  # type: ignore[attr-defined]
            fset.__wrapped__ = original_fset  # type: ignore[attr-defined]

        fget._reactive_root_wrapped = True  # type: ignore[attr-defined]
        if original_fget is not None:
            fget.__wrapped__ = original_fget  # type: ignore[attr-defined]
        setattr(cls, name, property(fget=fget, fset=fset, fdel=attr.fdel, doc=attr.__doc__))

    @classmethod
    def _wrap_derived_properties(
        cls,
        dependency_graph: dict[str, set[str]],
    ) -> None:
        del dependency_graph
        for name, index in cls._reactive_indices.items():
            attr = cls._find_property(name)
            if attr is None or attr.fget is None:
                continue
            original_fget = _unwrap_function(attr.fget)
            bit = 1 << index

            def make_lazy_fget(orig, node, slot, node_bit):
                def lazy_fget(self):
                    self._ensure_reactive_state()
                    if not self._reactive_dirty_mask & node_bit:
                        value = self._reactive_values[slot]
                        plan = self._reactive_value_plans[slot]
                        return value if plan is None else _expose_reactive_value(value, plan)

                    value = orig(self)
                    value, plan = _prepare_reactive_value(value, path=node)
                    self._reject_nested_cycle(value, node=node)
                    old = self._reactive_values[slot]
                    if old is not None:
                        self._unsubscribe_nested(node, old)
                    self._reactive_values[slot] = value
                    self._reactive_value_plans[slot] = plan
                    self._subscribe_nested(node, value)
                    self._reactive_dirty_mask &= ~node_bit
                    return value if plan is None else _expose_reactive_value(value, plan)

                lazy_fget.__wrapped__ = orig
                return lazy_fget

            setattr(
                cls,
                name,
                property(
                    fget=make_lazy_fget(original_fget, name, index, bit),
                    fset=attr.fset,
                    fdel=attr.fdel,
                    doc=attr.__doc__,
                ),
            )

    def _ensure_reactive_state(self) -> None:
        if "_reactive_values" not in self.__dict__:
            Reactive.__init__(self)

    def _prepare_root_value(self, name: str, value: Any) -> Any:
        return type(self).reactive_inspections(name, value)

    @classmethod
    def reactive_inspections(cls, name: str, value: Any) -> Any:
        return value

    def _state_changed(self) -> None:
        self._reactive_revision += 1
        self._notify_observers()

    def _nested_dependency_changed(self, node: str) -> None:
        self._reactive_dirty_mask |= self._reactive_dependent_masks.get(node, 0)
        self._state_changed()

    def _subscribe_nested(self, node: str, value: Any) -> None:
        for child in _iter_nested_reactives(value):
            child._add_observer(self, node)

    def _unsubscribe_nested(self, node: str, value: Any) -> None:
        for child in _iter_nested_reactives(value):
            child._remove_observer(self, node)

    def _add_observer(self, parent: Reactive, node: str) -> None:
        key = (id(parent), node)
        self._reactive_observers[key] = weakref.ref(parent)

    def _remove_observer(self, parent: Reactive, node: str) -> None:
        self._reactive_observers.pop((id(parent), node), None)

    def _notify_observers(self) -> None:
        stale: list[tuple[int, str]] = []
        for key, parent_ref in tuple(self._reactive_observers.items()):
            parent = parent_ref()
            if parent is None:
                stale.append(key)
            else:
                parent._nested_dependency_changed(key[1])
        for key in stale:
            self._reactive_observers.pop(key, None)

    def _reject_nested_cycle(self, value: Any, *, node: str) -> None:
        for child in _iter_nested_reactives(value):
            if child is self or child._contains_reactive(self, set()):
                raise ValueError(
                    f"Reactive containment must be acyclic; node {node!r} creates a cycle"
                )

    def _contains_reactive(self, target: Reactive, seen: set[int]) -> bool:
        if self is target:
            return True
        identity = id(self)
        if identity in seen:
            return False
        seen.add(identity)
        values: list[Any] = []
        for root in self.root_properties:
            cached_name = f"cached_{root}"
            if cached_name in self.__dict__:
                values.append(self.__dict__[cached_name])
        for index, value in enumerate(self._reactive_values):
            if not self._reactive_dirty_mask & (1 << index):
                values.append(value)
        for value in values:
            for child in _iter_nested_reactives(value):
                if child._contains_reactive(target, seen):
                    return True
        return False

    @classmethod
    def _build_dependency_graph(cls, roots: set[str]) -> dict[str, set[str]]:
        valid_nodes = set(roots)
        props: dict[str, property] = {}
        base_props = set(Reactive.__dict__)

        for name in dir(cls):
            if name.startswith("__") or name in base_props:
                continue
            attr = cls._find_property(name)
            if attr is None:
                continue
            valid_nodes.add(name)
            if name not in roots:
                props[name] = attr

        graph: dict[str, set[str]] = {}
        for name, prop in props.items():
            if prop.fget is None:
                continue
            original_func = _unwrap_function(prop.fget)
            raw_deps = _parse_dependency(original_func)
            explicit = getattr(original_func, "_reactive_deps", None)
            if explicit is not None:
                raw_deps |= explicit
            graph[name] = {dep for dep in raw_deps if dep in valid_nodes and dep != name}
        return graph

    @classmethod
    def _find_property(cls, name: str) -> property | None:
        for klass in cls.__mro__:
            if name in klass.__dict__:
                value = klass.__dict__[name]
                return value if isinstance(value, property) else None
        return None


def depends_on(*deps: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Declare dependencies that cannot be inferred from ``self.attr`` reads."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "_reactive_deps", set(deps))
        return func

    return decorator


def _build_dependent_masks(
    *,
    roots: set[str],
    derived_names: tuple[str, ...],
    reverse_adj: dict[str, set[str]],
) -> dict[str, int]:
    indices = {name: i for i, name in enumerate(derived_names)}
    memo: dict[str, int] = {}

    def descendants(node: str) -> int:
        cached = memo.get(node)
        if cached is not None:
            return cached
        mask = 0
        for child in reverse_adj.get(node, ()):
            mask |= 1 << indices[child]
            mask |= descendants(child)
        memo[node] = mask
        return mask

    return {name: descendants(name) for name in roots | set(derived_names)}


def _prepare_reactive_value(value: Any, *, path: str) -> tuple[Any, Any]:
    """Freeze a value and return its minimal public-exposure plan."""

    value_type = type(value)
    if value_type in _SCALAR_TYPES or isinstance(value, np.generic):
        return value, None
    if value_type is np.ndarray:
        if value.dtype.hasobject:
            raise TypeError(f"{path} cannot contain an object-dtype ndarray")
        if value.flags.writeable:
            value.flags.writeable = False
        return value, None
    if isinstance(value, Reactive):
        return value, None
    if value_type is bytearray:
        return bytes(value), _PLAN_BYTEARRAY
    if value_type is list:
        prepared_items: list[Any] = []
        child_plans: list[Any] = []
        for index, item in enumerate(value):
            prepared, plan = _prepare_reactive_value(item, path=f"{path}[{index}]")
            prepared_items.append(prepared)
            child_plans.append(plan)
        return tuple(prepared_items), (_PLAN_LIST, tuple(child_plans))
    if value_type is tuple:
        prepared_items: list[Any] = []
        child_plans: list[Any] = []
        changed = False
        for index, item in enumerate(value):
            prepared, plan = _prepare_reactive_value(item, path=f"{path}[{index}]")
            prepared_items.append(prepared)
            child_plans.append(plan)
            changed |= prepared is not item or plan is not None
        if not changed:
            return value, None
        return tuple(prepared_items), (_PLAN_TUPLE, tuple(child_plans))
    if value_type is dict:
        prepared: dict[Any, Any] = {}
        child_plans: list[tuple[Any, Any]] = []
        for key, item in value.items():
            if type(key) not in _SCALAR_TYPES - {type(None)}:
                raise TypeError(f"{path} has unsupported dictionary key {type(key).__name__}")
            prepared_item, plan = _prepare_reactive_value(item, path=f"{path}[{key!r}]")
            prepared[key] = prepared_item
            child_plans.append((key, plan))
        return MappingProxyType(prepared), (_PLAN_DICT, tuple(child_plans))
    if value_type is set:
        prepared_items = []
        for item in value:
            prepared, plan = _prepare_reactive_value(item, path=f"{path}[set-item]")
            if plan is not None:
                raise TypeError(f"{path} set items must already be immutable")
            prepared_items.append(prepared)
        return frozenset(prepared_items), _PLAN_SET
    if value_type is frozenset:
        prepared_items = []
        changed = False
        for item in value:
            prepared, plan = _prepare_reactive_value(item, path=f"{path}[set-item]")
            if plan is not None:
                raise TypeError(f"{path} frozenset items must already be immutable")
            prepared_items.append(prepared)
            changed |= prepared is not item
        return (frozenset(prepared_items) if changed else value), None
    # External data contracts (for example frozen kernel dataclasses or an
    # equilibrium supplied by another package) are snapshot roots. They do not
    # participate in nested invalidation and must be replaced as a whole when
    # their state changes.
    return value, None


def _expose_reactive_value(value: Any, plan: Any) -> Any:
    """Restore the public built-in type only where freezing changed it."""

    if plan is None:
        return value
    kind = plan if isinstance(plan, str) else plan[0]
    if kind == _PLAN_BYTEARRAY:
        return bytearray(value)
    if kind == _PLAN_SET:
        return set(value)
    if kind in {_PLAN_LIST, _PLAN_TUPLE}:
        child_plans = plan[1]
        exposed = (
            _expose_reactive_value(item, child_plan)
            for item, child_plan in zip(value, child_plans, strict=True)
        )
        return list(exposed) if kind == _PLAN_LIST else tuple(exposed)
    if kind == _PLAN_DICT:
        return {key: _expose_reactive_value(value[key], child_plan) for key, child_plan in plan[1]}
    raise RuntimeError(f"unknown Reactive exposure plan {kind!r}")


def _iter_nested_reactives(value: Any):
    if isinstance(value, Reactive):
        yield value
    elif type(value) in {tuple, frozenset}:
        for item in value:
            yield from _iter_nested_reactives(item)
    elif type(value) in {dict, MappingProxyType}:
        for item in value.values():
            yield from _iter_nested_reactives(item)


def _build_reverse_adj(dependency_graph: dict[str, set[str]]) -> dict[str, set[str]]:
    reverse: dict[str, set[str]] = {}
    for property_name, dependencies in dependency_graph.items():
        for dependency in dependencies:
            reverse.setdefault(dependency, set()).add(property_name)
    return reverse


def _validate_dependency_graph(
    roots: set[str],
    dependency_graph: dict[str, set[str]],
    reverse_adj: dict[str, set[str]],
) -> None:
    nodes = roots | set(dependency_graph)
    in_degree = {node: 0 for node in nodes}
    for property_name, dependencies in dependency_graph.items():
        in_degree[property_name] = len(dependencies)

    ready = sorted(node for node, degree in in_degree.items() if degree == 0)
    processed = 0
    while ready:
        next_ready: list[str] = []
        for node in ready:
            processed += 1
            for dependent in reverse_adj.get(node, ()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    next_ready.append(dependent)
        ready = sorted(next_ready)
    if processed != len(nodes):
        cyclic = sorted(name for name, degree in in_degree.items() if degree > 0)
        raise ValueError(f"Circular dependency detected among properties: {cyclic}")


def _unwrap_function(function: Callable[..., Any]) -> Callable[..., Any]:
    return getattr(function, "__wrapped__", function)


def _parse_dependency(function: Callable[..., Any]) -> set[str]:
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        return set()
    tree = ast.parse(textwrap.dedent(source))
    names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Attribute(self, node: ast.Attribute) -> None:
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                names.add(node.attr)
            self.generic_visit(node)

    Visitor().visit(tree)
    return names
