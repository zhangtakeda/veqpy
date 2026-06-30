"""
Module: base.reactive

Role:
- Provide reactive caching and pull-based dependency freshness checks for model
  objects and solved snapshots.
- Keep public objects in a minimal-root-state form while rebuilding derived
  quantities lazily from formulas.

Public API:
- Reactive
- depends_on

Design notes:
- Subclasses declare independent root fields in ``root_properties``.  Assigning
  a root value only updates that field and bumps a version token; it does not
  walk the dependency graph or eagerly recompute downstream properties.
- Derived ``@property`` values are wrapped at class creation time.  Dependencies
  are inferred from the property AST when possible, and non-obvious dependencies
  can be declared explicitly with ``depends_on``.
- Derived reads compare cached dependency tokens and recompute only when a direct
  dependency changed.  If a dependency value is itself ``Reactive``, the nested
  object's revision participates in the token so child root writes invalidate
  parent-derived values that depend on that child.
- ``Reactive`` is intentionally not used for operator hot paths.  Runtime solver
  state uses explicit workspaces and in-place arrays; snapshots use this formula
  system because they need interpretable, serializable, on-demand diagnostics.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------


class Reactive:
    """Base class for reactive caching.

    Each reactive node, root or derived, owns a monotonically increasing version.
    Root setters only assign and bump their own version plus the object's state
    revision. Derived recomputation bumps only the derived node version.
    Derived getters compare the versions of their direct dependencies and
    recompute lazily when at least one token changed.

    If a dependency value is itself ``Reactive``, its object-level state
    revision is included in the token. This is what makes
    ``parent.child.root = value`` invalidate parent-derived properties that
    depend on ``parent.child``.
    """

    dependency_graph: dict[str, set[str]] = {}
    root_properties: set[str]
    _reactive_derived_properties: frozenset[str] = frozenset()
    _reactive_all_properties: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self._init_reactive_state()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        roots = cls._validate_root_properties()
        dependency_graph = cls._build_dependency_graph(roots)
        reverse_adj = _build_reverse_adj(dependency_graph)
        _validate_dependency_graph(roots, dependency_graph, reverse_adj)

        cls.dependency_graph = dependency_graph
        cls._reactive_derived_properties = frozenset(dependency_graph.keys())
        cls._reactive_all_properties = frozenset(roots | set(dependency_graph.keys()))

        cls._setup_root_properties(roots)
        cls._wrap_derived_properties(dependency_graph)

    def __deepcopy__(self, memo):
        cls = self.__class__
        cloned = cls.__new__(cls)
        memo[id(self)] = cloned

        for k, v in self.__dict__.items():
            if k == "cache":
                object.__setattr__(cloned, k, {})
            else:
                object.__setattr__(cloned, k, deepcopy(v, memo))

        cloned._init_reactive_state()
        return cloned

    def invalidate(self, *names: str) -> None:
        """Invalidate cached values.

        With no names, clear this instance's derived cache and bump the object
        revision so parents that depend on this object will revalidate.

        With names, each named root/derived node has its version bumped. Named
        derived caches are removed; downstream caches are left in place and will
        self-detect staleness on next read.
        """

        self._init_reactive_state()

        if not names:
            self.cache.clear()
            self._bump_revision()
            return

        for name in names:
            self.cache.pop(name, None)
            self._bump_version(name)

    @dataclass(frozen=True)
    class _CacheEntry:
        value: Any
        dependency_tokens: tuple[tuple[str, tuple[int, int | None]], ...]

    def _init_reactive_state(self) -> None:
        """Initialize per-instance reactive bookkeeping.

        This method is intentionally idempotent so root setters still work in
        subclasses that assign roots before calling ``super().__init__()``.
        """

        if "cache" not in self.__dict__:
            object.__setattr__(self, "cache", {})

        if "_version" not in self.__dict__:
            object.__setattr__(
                self,
                "_version",
                {name: 0 for name in self._reactive_all_properties},
            )

        if "_revision" not in self.__dict__:
            object.__setattr__(self, "_revision", 0)

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
        """Create or wrap version-bumping properties for root attributes."""

        for name in roots:
            attr = cls._find_property(name)
            if attr is None:
                # Most model classes declare roots as plain assignments; install
                # descriptors so __init__ assignment still bumps reactive state.
                cls._install_default_root_property(name)
            else:
                cls._wrap_existing_root_property(name, attr)

    @classmethod
    def _install_default_root_property(cls, name: str) -> None:
        cached_name = f"cached_{name}"

        def fget(self):
            self._init_reactive_state()
            v = getattr(self, cached_name, None)
            return _freeze_ndarray(v)

        def fset(self, value):
            self._init_reactive_state()
            value = self._prepare_root_value(name, value)
            object.__setattr__(self, cached_name, _freeze_ndarray(value))
            self._bump_version(name)

        fget._reactive_root_wrapped = True  # type: ignore[attr-defined]
        fset._reactive_root_wrapped = True  # type: ignore[attr-defined]

        setattr(cls, name, property(fget=fget, fset=fset))

    @classmethod
    def _wrap_existing_root_property(cls, name: str, attr: property) -> None:
        """Wrap a user-defined root property so its setter bumps the version."""

        if getattr(attr.fget, "_reactive_root_wrapped", False) or getattr(
            attr.fset, "_reactive_root_wrapped", False
        ):
            return

        original_fget = attr.fget
        original_fset = attr.fset

        def fget(self):
            self._init_reactive_state()
            if original_fget is None:
                raise AttributeError(f"unreadable attribute {name!r}")
            return _freeze_ndarray(original_fget(self))

        if original_fset is None:
            fset = None
        else:

            def fset(self, value):
                self._init_reactive_state()
                value = self._prepare_root_value(name, value)
                original_fset(self, _freeze_ndarray(value))
                self._bump_version(name)

            fset._reactive_root_wrapped = True  # type: ignore[attr-defined]
            fset.__wrapped__ = original_fset  # type: ignore[attr-defined]

        fget._reactive_root_wrapped = True  # type: ignore[attr-defined]
        if original_fget is not None:
            fget.__wrapped__ = original_fget  # type: ignore[attr-defined]

        setattr(
            cls,
            name,
            property(fget=fget, fset=fset, fdel=attr.fdel, doc=attr.__doc__),
        )

    @classmethod
    def _wrap_derived_properties(
        cls,
        dependency_graph: dict[str, set[str]],
    ) -> None:
        """Wrap derived properties with version-token based lazy cache lookups."""

        for name, deps in dependency_graph.items():
            attr = cls._find_property(name)
            if attr is None or attr.fget is None:
                continue

            original_fget = _unwrap_function(attr.fget)
            ordered_deps = tuple(sorted(deps))

            def make_lazy_fget(orig, n, dep_names):
                def lazy_fget(self):
                    self._init_reactive_state()

                    tokens = tuple((dep, self._dependency_token(dep)) for dep in dep_names)

                    entry = self.cache.get(n)
                    if entry is not None and entry.dependency_tokens == tokens:
                        return entry.value

                    # Cache entries are validated by direct dependency tokens,
                    # so downstream properties can stay cached until read.
                    value = orig(self)
                    self.cache[n] = self._CacheEntry(value=value, dependency_tokens=tokens)
                    self._bump_version(n, bump_object_revision=False)
                    return value

                lazy_fget.__wrapped__ = orig
                return lazy_fget

            lazy_prop = property(
                fget=make_lazy_fget(original_fget, name, ordered_deps),
                fset=attr.fset,
                fdel=attr.fdel,
                doc=attr.__doc__,
            )
            setattr(cls, name, lazy_prop)

    def _dependency_token(self, name: str) -> tuple[int, int | None]:
        """Return the current version token for a direct dependency."""

        self._init_reactive_state()

        if name in self._reactive_derived_properties:
            # Force the dependency to validate/recompute before reading its
            # version. Without this, A -> B -> root can return stale A because
            # B's version would not change until B is read.
            value = getattr(self, name)
        else:
            value = getattr(self, name, None)

        own_version = self._version.get(name, 0)
        nested_revision = _nested_reactive_revision(value)
        return (own_version, nested_revision)

    def _prepare_root_value(self, name: str, value: Any) -> Any:
        """Normalize and validate a root assignment before storage.

        Subclasses can override ``reactive_inspections`` to centralize root
        coercion and validation while keeping ``__init__`` as plain assignment.
        """

        return type(self).reactive_inspections(name, value)

    @classmethod
    def reactive_inspections(cls, name: str, value: Any) -> Any:
        """Subclass hook for root normalization/validation."""

        return value

    def _bump_version(self, name: str, *, bump_object_revision: bool = True) -> None:
        self._init_reactive_state()
        self._version[name] = self._version.get(name, 0) + 1
        if bump_object_revision:
            self._bump_revision()

    def _bump_revision(self) -> None:
        self._init_reactive_state()
        object.__setattr__(self, "_revision", self._revision + 1)

    @classmethod
    def _build_dependency_graph(cls, roots: set[str]) -> dict[str, set[str]]:
        """Build the dependency graph for all reactive derived properties."""

        valid_nodes = set(roots)
        props: dict[str, property] = {}
        base_props = set(Reactive.__dict__.keys())

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
                # Explicit dependencies augment AST inference for dynamic reads
                # that cannot be found from ``self.attr`` syntax.
                raw_deps = raw_deps | explicit

            graph[name] = {d for d in raw_deps if d in valid_nodes and d != name}

        return graph

    @classmethod
    def _find_property(cls, name: str) -> property | None:
        for klass in cls.__mro__:
            if name in klass.__dict__:
                obj = klass.__dict__[name]
                if isinstance(obj, property):
                    return obj
                return None
        return None


def depends_on(*deps: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Declare additional dependencies for a property.

    The dependency names should be first-level reactive attribute names, e.g.
    ``"profile"`` or ``"layout"``. For nested Reactive objects, depending on the
    first-level attribute is sufficient because its nested revision is included
    in the dependency token.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "_reactive_deps", set(deps))
        return func

    return decorator


# -----------------------------------------------------------------------------
# Private implementation
# -----------------------------------------------------------------------------


def _freeze_ndarray(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.flags.writeable:
        value.flags.writeable = False
    return value


def _nested_reactive_revision(value: Any) -> int | None:
    if isinstance(value, Reactive):
        value._init_reactive_state()
        return value._revision

    if isinstance(value, dict):
        return _nested_reactive_collection_revision(value.values())

    if isinstance(value, (tuple, list)):
        return _nested_reactive_collection_revision(value)

    return None


def _nested_reactive_collection_revision(values) -> int | None:
    revisions: list[int] = []
    for item in values:
        revision = _nested_reactive_revision(item)
        if revision is not None:
            revisions.append(revision)

    if not revisions:
        return None
    return hash(tuple(revisions))


def _build_reverse_adj(dependency_graph: dict[str, set[str]]) -> dict[str, set[str]]:
    """Build a one-hop reverse adjacency map: property -> direct dependents."""

    rev: dict[str, set[str]] = {}
    for prop_name, deps in dependency_graph.items():
        for dep in deps:
            rev.setdefault(dep, set()).add(prop_name)
    return rev


def _validate_dependency_graph(
    roots: set[str],
    dependency_graph: dict[str, set[str]],
    reverse_adj: dict[str, set[str]],
) -> None:
    """Validate that the dependency graph is acyclic."""

    all_props = set(roots) | set(dependency_graph.keys())
    if not all_props:
        return

    in_degree: dict[str, int] = {p: 0 for p in all_props}

    for prop, deps in dependency_graph.items():
        in_degree[prop] = len(deps)

    for root in roots:
        in_degree.setdefault(root, 0)

    queue = sorted([p for p, d in in_degree.items() if d == 0])

    processed = 0
    while queue:
        next_queue = []
        for node in queue:
            processed += 1
            for child in reverse_adj.get(node, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_queue.append(child)
        queue = sorted(next_queue)

    if processed != len(all_props):
        remaining = [p for p in all_props if in_degree.get(p, 0) > 0]
        cycles = _detect_cycles(remaining, dependency_graph)
        raise ValueError(
            "Circular dependency detected:\n"
            + f"-- involved properties: {remaining}\n"
            + f"-- dependency cycles: {cycles}"
        )


def _detect_cycles(
    nodes: list[str],
    dependency_graph: dict[str, set[str]],
) -> list[list[str]]:
    """Detect and return all cyclic dependency paths."""

    cycles: list[list[str]] = []
    node_set = set(nodes)

    def dfs(node: str, path: list[str], visited: set[str]):
        if node in path:
            cycle_start = path.index(node)
            cycles.append(path[cycle_start:] + [node])
            return

        if node in visited:
            return

        visited.add(node)
        path.append(node)

        for dep in dependency_graph.get(node, set()):
            if dep in node_set:
                dfs(dep, path.copy(), visited)

    visited_global: set[str] = set()
    for node in nodes:
        if node not in visited_global:
            dfs(node, [], visited_global)

    return cycles


def _unwrap_function(func):
    if hasattr(func, "__wrapped__"):
        return func.__wrapped__
    return func


def _parse_dependency(func) -> set[str]:
    """Parse names of ``self.xxx`` attributes accessed by a function."""

    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return set()

    source = textwrap.dedent(source)
    tree = ast.parse(source)
    names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        """AST node visitor."""

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                names.add(node.attr)
            self.generic_visit(node)

    Visitor().visit(tree)
    return names
