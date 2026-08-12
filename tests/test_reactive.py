from __future__ import annotations

import copy
import pickle
from copy import deepcopy
from types import MappingProxyType
from typing import Self

import numpy as np
import pytest

from veqpy.base import Reactive
from veqpy.base.serial import Serial


class SplitGraph(Reactive):
    root_properties = {"x", "y"}

    def __init__(self, x: float, y: float) -> None:
        super().__init__()
        self._left_calls = 0
        self._right_calls = 0
        self._total_calls = 0
        self.x = x
        self.y = y

    @property
    def left(self) -> float:
        self._left_calls += 1
        return 2.0 * self.x

    @property
    def right(self) -> float:
        self._right_calls += 1
        return 3.0 * self.y

    @property
    def total(self) -> float:
        self._total_calls += 1
        return self.left + self.right


class ParentGraph(Reactive):
    root_properties = {"child"}

    def __init__(self, child: SplitGraph) -> None:
        super().__init__()
        self._value_calls = 0
        self.child = child

    @property
    def value(self) -> float:
        self._value_calls += 1
        return self.child.total + 1.0


class DerivedChildGraph(Reactive):
    root_properties = {"x"}

    def __init__(self, x: float) -> None:
        super().__init__()
        self.x = x

    @property
    def child(self) -> SplitGraph:
        return SplitGraph(self.x, self.x + 1.0)


class MappingGraph(Reactive):
    root_properties = {"children"}

    def __init__(self, child: SplitGraph) -> None:
        super().__init__()
        self.children = {"main": child}

    @property
    def value(self) -> float:
        return self.children["main"].total


class ArrayGraph(Reactive):
    root_properties = {"source"}

    def __init__(self, source: np.ndarray) -> None:
        super().__init__()
        self._calls = 0
        self.source = source

    @property
    def doubled(self) -> np.ndarray:
        self._calls += 1
        return 2.0 * self.source


class PickleGraph(Reactive):
    root_properties = {"x"}

    def __init__(self, x: float) -> None:
        super().__init__()
        self.x = x

    @property
    def doubled(self) -> float:
        return 2.0 * self.x


class NativeContainersGraph(Reactive):
    root_properties = {"buffer", "children", "items", "tags", "nested"}

    def __init__(self, child: SplitGraph) -> None:
        super().__init__()
        self.items = [1, 2]
        self.tags = {"a", "b"}
        self.buffer = bytearray(b"ab")
        self.children = [child]
        self.nested = ([1, 2], {"x": [3, 4]})

    @property
    def value(self) -> float:
        return self.children[0].total


class SerializableContainersGraph(Reactive, Serial):
    root_properties = {
        "blob",
        "buffer",
        "frozen_tags",
        "items",
        "mapping",
        "pair",
        "tags",
    }

    def __init__(
        self,
        *,
        items: list[int],
        tags: set[str],
        frozen_tags: frozenset[str],
        buffer: bytearray,
        blob: bytes,
        pair: tuple[int, str],
        mapping: dict[str, list[int]],
    ) -> None:
        super().__init__()
        self.items = items
        self.tags = tags
        self.frozen_tags = frozen_tags
        self.buffer = buffer
        self.blob = blob
        self.pair = pair
        self.mapping = mapping

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        return {
            "items": list[int],
            "tags": set[str],
            "frozen_tags": frozenset[str],
            "buffer": bytearray,
            "blob": bytes,
            "pair": tuple[int, str],
            "mapping": dict[str, list[int]],
        }

    @classmethod
    def example(cls) -> Self:
        return cls(
            items=[1, 2],
            tags={"a", "b"},
            frozen_tags=frozenset({"c", "d"}),
            buffer=bytearray(b"ab"),
            blob=b"cd",
            pair=(3, "e"),
            mapping={"f": [4, 5]},
        )


def test_clean_reads_are_cached_and_root_writes_only_dirty_descendants() -> None:
    graph = SplitGraph(2.0, 5.0)

    assert graph.total == 19.0
    assert graph.total == 19.0
    assert (graph._left_calls, graph._right_calls, graph._total_calls) == (1, 1, 1)

    graph.x = 4.0
    assert graph.total == 23.0
    assert (graph._left_calls, graph._right_calls, graph._total_calls) == (2, 1, 2)


def test_nested_reactive_changes_push_invalidation_to_parents() -> None:
    child = SplitGraph(2.0, 5.0)
    parent = ParentGraph(child)
    mapping = MappingGraph(child)

    assert parent.value == 20.0
    assert mapping.value == 19.0
    child.y = 7.0
    assert parent.value == 26.0
    assert mapping.value == 25.0
    assert parent._value_calls == 2


def test_root_dictionary_structure_is_externally_immutable() -> None:
    original = SplitGraph(2.0, 5.0)
    replacement = SplitGraph(7.0, 11.0)
    graph = MappingGraph(original)

    exposed = graph.children
    assert type(graph.__dict__["cached_children"]) is MappingProxyType
    exposed["main"] = replacement
    assert graph.children["main"] is original
    assert graph.value == 19.0

    # Nested Reactive values intentionally remain live: only container
    # membership is immutable without replacing the complete root dictionary.
    original.x = 4.0
    assert graph.value == 23.0


def test_native_mutable_containers_are_frozen_and_exposed_as_copies() -> None:
    child = SplitGraph(2.0, 5.0)
    graph = NativeContainersGraph(child)

    assert type(graph.__dict__["cached_items"]) is tuple
    assert type(graph.__dict__["cached_tags"]) is frozenset
    assert type(graph.__dict__["cached_buffer"]) is bytes
    assert type(graph.items) is list
    assert type(graph.tags) is set
    assert type(graph.buffer) is bytearray

    items = graph.items
    tags = graph.tags
    buffer = graph.buffer
    items.append(3)
    tags.add("c")
    buffer[0] = ord("z")
    assert graph.items == [1, 2]
    assert graph.tags == {"a", "b"}
    assert graph.buffer == bytearray(b"ab")

    nested = graph.nested
    assert type(nested) is tuple
    assert type(nested[0]) is list
    assert type(nested[1]) is dict
    assert type(nested[1]["x"]) is list
    nested[0].append(3)
    nested[1]["x"].append(5)
    assert graph.nested == ([1, 2], {"x": [3, 4]})

    assert graph.value == 19.0
    child.x = 4.0
    assert graph.value == 23.0


def test_immutable_values_are_returned_without_reconstruction() -> None:
    source = np.arange(4.0)
    graph = ArrayGraph(source)
    plain_tuple = (source, "stable")

    assert graph.source is graph.source
    assert graph.doubled is graph.doubled

    class TupleGraph(Reactive):
        root_properties = {"value"}

        def __init__(self, value: tuple[object, ...]) -> None:
            super().__init__()
            self.value = value

    tuple_graph = TupleGraph(plain_tuple)
    assert tuple_graph.value is plain_tuple


def test_pickle_stores_roots_and_rebuilds_derived_state() -> None:
    graph = PickleGraph(3.0)
    assert graph.doubled == 6.0

    state = graph.__getstate__()
    assert state["__reactive_roots__"] == {"x": 3.0}
    assert "_reactive_values" not in state
    restored = pickle.loads(pickle.dumps(graph))
    assert restored.x == 3.0
    assert restored.doubled == 6.0


def test_freeze_is_in_place_and_keeps_lazy_derived_evaluation() -> None:
    graph = SplitGraph(2.0, 5.0)

    assert graph.freeze() is graph
    assert graph.is_frozen
    with pytest.raises(AttributeError, match="root property 'x'"):
        graph.x = 4.0

    assert graph.total == 19.0
    assert graph.total == 19.0
    assert (graph._left_calls, graph._right_calls, graph._total_calls) == (1, 1, 1)


def test_freeze_recursively_freezes_nested_reactive_roots() -> None:
    child = SplitGraph(2.0, 5.0)
    parent = ParentGraph(child)

    parent.freeze()
    assert parent.is_frozen
    assert child.is_frozen
    with pytest.raises(AttributeError, match="root property 'y'"):
        child.y = 7.0


def test_reactive_derived_after_freeze_is_also_frozen() -> None:
    graph = DerivedChildGraph(2.0).freeze()

    child = graph.child
    assert child.is_frozen
    with pytest.raises(AttributeError, match="root property 'x'"):
        child.x = 3.0


def test_thaw_returns_an_independent_mutable_snapshot_with_empty_caches() -> None:
    original = ParentGraph(SplitGraph(2.0, 5.0))
    assert original.value == 20.0
    original.freeze()

    thawed = original.thaw()
    assert thawed is not original
    assert thawed.child is not original.child
    assert not thawed.is_frozen
    assert not thawed.child.is_frozen
    assert thawed._value_calls == 1

    thawed.child.x = 6.0
    assert thawed.value == 28.0
    assert original.value == 20.0


def test_pickle_preserves_frozen_state_without_persisting_caches() -> None:
    graph = PickleGraph(3.0).freeze()

    restored = pickle.loads(pickle.dumps(graph))
    assert restored.is_frozen
    assert restored.doubled == 6.0
    with pytest.raises(AttributeError, match="root property 'x'"):
        restored.x = 4.0


def test_pull_reactive_pickle_state_remains_readable() -> None:
    restored = PickleGraph.__new__(PickleGraph)
    restored.__setstate__(
        {
            "cached_x": 4.0,
            "cache": {"doubled": "stale-derived-value"},
            "_version": {"x": 2, "doubled": 1},
            "_revision": 2,
        }
    )

    assert restored.x == 4.0
    assert restored.doubled == 8.0


def test_shallow_copy_is_rejected() -> None:
    with pytest.raises(TypeError, match="does not support shallow copy"):
        copy.copy(PickleGraph(3.0))


@pytest.mark.parametrize("extension", ["json", "pkl"])
def test_native_root_container_serial_roundtrip(tmp_path, extension: str) -> None:
    source = SerializableContainersGraph.example()
    path = tmp_path / f"containers.{extension}"
    source.write(str(path))
    restored = SerializableContainersGraph.load(str(path))

    assert type(restored.items) is list
    assert type(restored.tags) is set
    assert type(restored.frozen_tags) is frozenset
    assert type(restored.buffer) is bytearray
    assert type(restored.blob) is bytes
    assert type(restored.pair) is tuple
    assert type(restored.mapping) is dict
    assert type(restored.mapping["f"]) is list
    assert restored.items == source.items
    assert restored.tags == source.tags
    assert restored.frozen_tags == source.frozen_tags
    assert restored.buffer == source.buffer
    assert restored.blob == source.blob
    assert restored.pair == source.pair
    assert restored.mapping == source.mapping


def test_arrays_are_frozen_and_recomputation_returns_a_new_snapshot() -> None:
    source = np.arange(4.0)
    graph = ArrayGraph(source)
    first = graph.doubled

    assert not graph.source.flags.writeable
    assert not first.flags.writeable
    assert graph.doubled is first

    graph.source = np.arange(4.0) + 1.0
    second = graph.doubled
    assert second is not first
    np.testing.assert_array_equal(first, [0.0, 2.0, 4.0, 6.0])
    np.testing.assert_array_equal(second, [2.0, 4.0, 6.0, 8.0])


def test_deepcopy_resets_caches_and_reconnects_nested_notifications() -> None:
    original = ParentGraph(SplitGraph(2.0, 5.0))
    assert original.value == 20.0

    cloned = deepcopy(original)
    assert cloned.value == 20.0
    cloned.child.x = 6.0
    assert cloned.value == 28.0
    assert original.value == 20.0


def test_object_arrays_are_rejected_but_external_snapshots_are_supported() -> None:
    graph = ArrayGraph(np.arange(2.0))

    with pytest.raises(TypeError, match="object-dtype ndarray"):
        graph.source = np.asarray([object()], dtype=object)
    snapshot = object()
    graph.source = snapshot
    assert graph.source is snapshot


def test_nested_reactive_containment_must_be_acyclic() -> None:
    left = ParentGraph(SplitGraph(1.0, 2.0))
    right = ParentGraph(SplitGraph(3.0, 4.0))
    original_right_child = right.child
    assert right.value == 19.0

    left.child = right
    with pytest.raises(ValueError, match="must be acyclic"):
        right.child = left
    original_right_child.x = 5.0
    assert right.value == 23.0


def test_root_assignment_before_super_init_remains_compatible() -> None:
    class LegacyOrder(Reactive):
        root_properties = {"x"}

        def __init__(self) -> None:
            self.x = 1.0
            super().__init__()

        @property
        def doubled(self) -> float:
            return 2.0 * self.x

    graph = LegacyOrder()
    assert graph.doubled == 2.0
    graph.x = 3.0
    assert graph.doubled == 6.0
