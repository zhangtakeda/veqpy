# Serial

`Serial` is VEQPy's unified serialization base class. It stores declarative root state that can reconstruct a model object, not a full memory image of the object. JSON, pickle, GEQDSK, and other formats share the `load/read/write` entry points; concrete readers and writers are dispatched through the serializer registry.

Source location: `veqpy/base/serial.py`; GEQDSK-related implementation lives in `veqpy/model/geqdsk.py`.

## Serializable Attributes

Each `Serial` subclass declares serializable fields:

```python
@classmethod
def serial_attributes(cls) -> dict[str, type]:
    return {"field": float, "array": np.ndarray}
```

Dataclass subclasses can infer fields from annotations. Derived caches, workspace buffers, and engine temporaries that are not listed in `serial_attributes()` are not persistent state.

## Format Semantics

JSON writes a type-tagged structure and recursively handles nested `Serial` objects, dataclasses, NumPy arrays, and common containers. Pickle stores the field dictionary. Other formats are registered through the registry, such as GEQDSK text I/O.

After reading, reactive objects rebuild derived caches from root state rather than restoring old caches from the file.

## load/read/write

`Serial.load(file)` creates a new object and reads the file. `read(file)` restores fields into an existing object. `write(file)` selects the writer by extension. Objects with complex constructors or frozen dataclasses are usually better loaded with `load()`.

## Boundary

`Serial` and `Reactive` form a clean boundary together: files store independent state, and the object read from the file reconstructs derived quantities by formula. Public file formats therefore do not change with runtime workspace optimization and do not treat a cache created during one solve or plot as a physical fact.
