"""VEQ Kernel report I/O.

This module intentionally contains no physical-state or GEQDSK conversion.
Reports are independent JSON snapshots of the four internal ABI records.  A
report name is the local-time microsecond timestamp; an exact timestamp
collision intentionally overwrites the earlier file per the repository
contract.
"""

from __future__ import annotations

import json
import os
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .kernels.contracts import KernelInput


def write_report(
    *,
    topology: object,
    config: object,
    input_buffer: object,
    output: object,
    report_dir: str | os.PathLike[str] | None,
    backend: str,
) -> Path:
    """Write one complete KT/KC/KI/KO semantic report and return its path."""

    directory = Path.cwd() / "report" if report_dir is None else Path(report_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone()
    path = directory / f"veqpy-{stamp:%Y%m%d-%H%M%S-%f}.json"
    payload = {
        "schema": "veqpy.kernel-report.v2",
        "created_at": stamp.isoformat(),
        "backend": str(backend),
        "KT": _semantic(topology),
        "KC": _semantic(config),
        "KI": _semantic(input_buffer),
        "KO": _semantic(output),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    path.write_bytes(encoded)
    return path


def load_report(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load one VEQ report JSON payload."""

    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _semantic(value: object) -> Any:
    if is_dataclass(value):
        semantic = {field.name: _semantic(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, KernelInput):
            semantic["source_capacity"] = value.source_capacity
        return semantic
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_semantic(item) for item in value]
    if isinstance(value, list):
        return [_semantic(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _semantic(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


__all__ = ["load_report", "write_report"]
