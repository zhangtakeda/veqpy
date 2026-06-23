from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

from veqpy.cpp import KernelRegistry, SolverThreadError, ThreadOwnedKernelSolver, VEQlibSolver
from veqpy.cpp.kernel_registry import _load_artifact_module
from veqpy.topology import Topology


def make_topology(**overrides: object) -> Topology:
    params: dict[str, object] = {
        "h_count": 3,
        "v_count": 0,
        "kappa_count": 6,
        "psin_count": 6,
        "F_count": 0,
        "c_counts": (),
        "s_counts": (3,),
        "Nr": 32,
        "Nt": 16,
        "route": "PF",
        "coordinate": "psin",
        "constraint": "Ip",
        "nodes": "uniform",
        "sample_count": 51,
    }
    params.update(overrides)
    return Topology(**params)  # type: ignore[arg-type]


class FakeCppSolver:
    def metadata(self) -> dict[str, str]:
        return {"route": "PF/psin/uniform/Ip"}

    def metadata_json(self) -> str:
        return json.dumps(self.metadata())

    def set_case_json(self, payload: str) -> None:
        self.payload = payload

    def warmup(self, count: int) -> None:
        self.warmup_count = count

    def solve_json(self) -> str:
        return json.dumps({"success": True})

    def solve_direct(self) -> tuple[bool]:
        return (True,)

    def residual_var_into(self, x: Any, out: Any) -> None:
        out[:] = x

    @property
    def last_elapsed_ms(self) -> float:
        return 0.0


def test_thread_owned_kernel_solver_rejects_cross_thread_use() -> None:
    solver = ThreadOwnedKernelSolver(FakeCppSolver())
    errors: list[BaseException] = []

    def use_from_other_thread() -> None:
        try:
            solver.metadata()
        except BaseException as exc:  # noqa: BLE001 - test captures exact exception below
            errors.append(exc)

    thread = threading.Thread(target=use_from_other_thread)
    thread.start()
    thread.join()

    assert errors
    assert isinstance(errors[0], SolverThreadError)


def test_veqlib_solver_build_dry_run_uses_registry(tmp_path: Path) -> None:
    registry = KernelRegistry(cache_root=tmp_path)
    solver = VEQlibSolver(make_topology(), registry=registry)

    artifact = solver.build(dry_run=True)

    assert artifact.root_dir == tmp_path / "v1.0.0" / "fastmath" / artifact.artifact_id
    assert artifact.metadata["artifact"]["module_name"].endswith(".veqlib_ext")
    assert artifact.metadata["artifact"]["status"] == "planned"


def test_registry_loads_existing_nanobind_artifact_when_debug_build_exists(tmp_path: Path) -> None:
    candidates = sorted(Path("veqlib/build/debug").glob("veqlib_ext*.so"))
    if not candidates:
        pytest.skip("veqlib debug nanobind extension has not been built")

    artifact = KernelRegistry(cache_root=tmp_path).get_or_build(make_topology(), dry_run=True)
    shutil.copy2(candidates[0], artifact.shared_library_path)

    module = _load_artifact_module(artifact)
    solver = module.KernelSolver()
    metadata = solver.metadata()

    assert metadata["route"] == "PF/psin/uniform/Ip"
    assert metadata["x_size"] == 18


def test_veqlib_solver_solve_reuses_prebuilt_artifact_when_debug_build_exists(
    tmp_path: Path,
) -> None:
    candidates = sorted(Path("veqlib/build/debug").glob("veqlib_ext*.so"))
    if not candidates:
        pytest.skip("veqlib debug nanobind extension has not been built")

    topology = make_topology()
    registry = KernelRegistry(cache_root=tmp_path)
    artifact = registry.get_or_build(topology, dry_run=True)
    shutil.copy2(candidates[0], artifact.shared_library_path)
    metadata = json.loads(artifact.metadata_path.read_text())
    metadata["artifact"]["status"] = "built"
    metadata["artifact"]["shared_library_sha256"] = hashlib.sha256(
        artifact.shared_library_path.read_bytes()
    ).hexdigest()
    artifact.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    solver = VEQlibSolver(topology, registry=registry)
    result = solver.solve_direct()

    assert len(result) == 15
    assert result[1] is True
    assert solver.metadata()["route"] == "PF/psin/uniform/Ip"
