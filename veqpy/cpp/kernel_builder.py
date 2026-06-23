from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from veqpy.topology import Topology

GENERATOR_VERSION = "veqpy.cpp.kernel_builder.v1"
ARTIFACT_SCHEMA = "veqpy.kernel_artifact.v1"
SOURCE_DIGEST_SCHEMA = "veqlib.source_digest.v1"


class KernelBuildError(RuntimeError):
    """Raised when a VEQlib kernel artifact cannot be planned or built."""


@dataclass(frozen=True, slots=True)
class KernelArtifact:
    """Resolved on-disk VEQlib kernel artifact."""

    topology: Topology
    artifact_id: str
    root_dir: Path
    cmake_build_dir: Path
    metadata_path: Path
    topology_path: Path
    build_path: Path
    kernel_py_path: Path
    shared_library_path: Path
    metadata: dict[str, Any]
    reused: bool
    built: bool


def default_kernel_cache_root() -> Path:
    """Return the VEQPy kernel cache root without creating it."""

    override = os.environ.get("VEQPY_KERNEL_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "veqpy" / "kernels"


def build_kernel(
    topology: Topology,
    *,
    cache_root: Path | None = None,
    source_dir: Path | None = None,
    cxx: str = "clang++",
    force: bool = False,
    dry_run: bool = False,
) -> KernelArtifact:
    """Resolve, optionally build, and record a VEQlib kernel artifact for ``topology``.

    ``dry_run=True`` writes the same planning metadata and CMake arguments but does not invoke
    CMake. It is the fast validation path used before the nanobind production API is finalized.
    """

    topology.validate_supported_for_veqlib_mvp()
    source_dir = _default_source_dir() if source_dir is None else source_dir.resolve()
    if not source_dir.exists():
        raise KernelBuildError(f"VEQlib source directory does not exist: {source_dir}")

    build_identity = _build_identity(topology, source_dir=source_dir, cxx=cxx)
    artifact_id = _compute_artifact_id(topology, build_identity)
    root = (cache_root or default_kernel_cache_root()).expanduser()
    root_dir = root / topology.version / topology.build / artifact_id
    lock_path = root / topology.version / topology.build / f"{artifact_id}.lock"
    root_dir.parent.mkdir(parents=True, exist_ok=True)

    with _exclusive_lock(lock_path):
        root_dir.mkdir(parents=True, exist_ok=True)
        paths = _artifact_paths(root_dir)
        reusable = _artifact_is_reusable(paths["metadata_path"], paths["shared_library_path"])
        if not force and reusable:
            metadata = _read_json(paths["metadata_path"])
            return KernelArtifact(
                topology=topology,
                artifact_id=artifact_id,
                root_dir=root_dir,
                cmake_build_dir=paths["cmake_build_dir"],
                metadata_path=paths["metadata_path"],
                topology_path=paths["topology_path"],
                build_path=paths["build_path"],
                kernel_py_path=paths["kernel_py_path"],
                shared_library_path=paths["shared_library_path"],
                metadata=metadata,
                reused=True,
                built=False,
            )

        started = time.perf_counter()
        cmake_args = _cmake_configure_args(topology, source_dir, paths["cmake_build_dir"], cxx)
        build_command = [
            "cmake",
            "--build",
            str(paths["cmake_build_dir"]),
            "--target",
            "veqlib_ext",
        ]
        metadata = _metadata_payload(
            topology=topology,
            artifact_id=artifact_id,
            source_dir=source_dir,
            build_identity=build_identity,
            cmake_args=cmake_args,
            build_command=build_command,
            dry_run=dry_run,
        )
        _write_json(paths["topology_path"], topology.to_canonical_dict())
        _write_json(paths["build_path"], metadata["build"])
        _write_kernel_py(paths["kernel_py_path"], metadata)

        built = False
        if dry_run:
            metadata["build"]["elapsed_ms"] = _elapsed_ms(started)
            metadata["artifact"]["status"] = "planned"
        else:
            _run_logged(cmake_args, paths["configure_log_path"], cwd=source_dir)
            _run_logged(build_command, paths["build_log_path"], cwd=source_dir)
            _copy_extension(paths["cmake_build_dir"], paths["shared_library_path"])
            metadata["build"]["elapsed_ms"] = _elapsed_ms(started)
            metadata["artifact"]["status"] = "built"
            metadata["artifact"]["shared_library_sha256"] = _file_sha256(
                paths["shared_library_path"]
            )
            built = True

        _write_json(paths["metadata_path"], metadata)
        return KernelArtifact(
            topology=topology,
            artifact_id=artifact_id,
            root_dir=root_dir,
            cmake_build_dir=paths["cmake_build_dir"],
            metadata_path=paths["metadata_path"],
            topology_path=paths["topology_path"],
            build_path=paths["build_path"],
            kernel_py_path=paths["kernel_py_path"],
            shared_library_path=paths["shared_library_path"],
            metadata=metadata,
            reused=False,
            built=built,
        )


def _artifact_paths(root_dir: Path) -> dict[str, Path]:
    return {
        "cmake_build_dir": root_dir / "cmake-build",
        "metadata_path": root_dir / "metadata.json",
        "topology_path": root_dir / "topology.json",
        "build_path": root_dir / "build.json",
        "kernel_py_path": root_dir / "kernel.py",
        "shared_library_path": root_dir / "veqlib.so",
        "configure_log_path": root_dir / "configure.log",
        "build_log_path": root_dir / "build.log",
    }


def _artifact_is_reusable(metadata_path: Path, shared_library_path: Path) -> bool:
    if not metadata_path.exists() or not shared_library_path.exists():
        return False
    metadata = _read_json(metadata_path)
    if metadata.get("schema") != ARTIFACT_SCHEMA:
        return False
    if metadata.get("artifact", {}).get("status") != "built":
        return False
    expected = metadata.get("artifact", {}).get("shared_library_sha256")
    return isinstance(expected, str) and _file_sha256(shared_library_path) == expected


def _metadata_payload(
    *,
    topology: Topology,
    artifact_id: str,
    source_dir: Path,
    build_identity: dict[str, Any],
    cmake_args: list[str],
    build_command: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    module_name = f"veqk_{artifact_id[:20]}"
    return {
        "schema": ARTIFACT_SCHEMA,
        "generator": GENERATOR_VERSION,
        "artifact": {
            "artifact_id": artifact_id,
            "status": "planned" if dry_run else "building",
            "module_name": module_name,
            "shared_library": "veqlib.so",
            "shared_library_sha256": None,
        },
        "topology": topology.to_canonical_dict(),
        "build_identity": build_identity,
        "build": {
            "build": topology.build,
            "dry_run": dry_run,
            "source_dir": str(source_dir),
            "cmake_configure": cmake_args,
            "cmake_build": build_command,
            "elapsed_ms": None,
        },
    }


def _write_kernel_py(path: Path, metadata: dict[str, Any]) -> None:
    module_name = metadata["artifact"]["module_name"]
    text = f'''from __future__ import annotations

import importlib.util
from pathlib import Path

ARTIFACT_ID = {metadata["artifact"]["artifact_id"]!r}
MODULE_NAME = {module_name!r}
SHARED_LIBRARY = Path(__file__).with_name("veqlib.so")


def load():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, SHARED_LIBRARY)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load VEQlib kernel artifact {{ARTIFACT_ID}}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
'''
    path.write_text(text)


def _cmake_configure_args(
    topology: Topology, source_dir: Path, build_dir: Path, cxx: str
) -> list[str]:
    build_type = "Debug" if topology.build == "debug" else "Release"
    fp_mode = "RELAXED" if topology.build == "fastmath" else "STRICT"
    kmax_limit = max(2, topology.K_max or 2)
    return [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DCMAKE_CXX_COMPILER={cxx}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DENABLE_ENZYME=OFF",
        "-DVEQLIB_ENABLE_PYTHON_BINDINGS=ON",
        "-DVEQLIB_ENABLE_NATIVE_OPTIMIZATIONS=ON",
        f"-DVEQLIB_FP_MODE={fp_mode}",
        "-DVEQLIB_ENABLE_THIN_LTO=ON",
        f"-DVEQ_NR={topology.Nr}",
        f"-DVEQ_NT={topology.Nt}",
        f"-DVEQ_H_PROFILE_COUNT={topology.h_count}",
        f"-DVEQ_V_PROFILE_COUNT={topology.v_count}",
        f"-DVEQ_KAPPA_PROFILE_COUNT={topology.kappa_count}",
        f"-DVEQ_PSIN_PROFILE_COUNT={topology.psin_count}",
        f"-DVEQ_F_PROFILE_COUNT={topology.F_count}",
        f"-DVEQ_COS_PROFILE_COUNTS={_cmake_list(topology.c_counts)}",
        f"-DVEQ_SIN_PROFILE_COUNTS={_cmake_list(topology.s_counts)}",
        f"-DVEQ_BOUNDARY_M_MAX={topology.M_max}",
        f"-DVEQ_PROFILE_KMAX_LIMIT={kmax_limit}",
    ]


def _build_identity(topology: Topology, *, source_dir: Path, cxx: str) -> dict[str, Any]:
    return {
        "schema": "veqpy.kernel_build_identity.v1",
        "generator": GENERATOR_VERSION,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag,
            "abi_flags": getattr(sys, "abiflags", ""),
        },
        "tools": {
            "cmake": _command_version(["cmake", "--version"]),
            "cxx": cxx,
            "cxx_version": _command_version([cxx, "--version"]),
            "nanobind": _package_version("nanobind"),
        },
        "topology_build": topology.build,
        "veqlib_source_digest": _source_digest(source_dir),
    }


def _compute_artifact_id(topology: Topology, build_identity: dict[str, Any]) -> str:
    payload = {
        "schema": ARTIFACT_SCHEMA,
        "topology": topology.to_canonical_dict(),
        "build_identity": build_identity,
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    digest = hashlib.sha256(data).digest()
    return base64.b32encode(digest).decode("ascii").lower().rstrip("=")[:32]


def _source_digest(source_dir: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = [path for path in source_dir.rglob("*") if _is_source_digest_file(path)]
    for path in sorted(files, key=lambda item: item.relative_to(source_dir).as_posix()):
        relative = path.relative_to(source_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "schema": SOURCE_DIGEST_SCHEMA,
        "algorithm": "sha256",
        "file_count": len(files),
        "sha256": digest.hexdigest(),
    }


def _is_source_digest_file(path: Path) -> bool:
    if not path.is_file():
        return False
    ignored = {"build", "__pycache__", "experiments"}
    if any(part in ignored for part in path.parts):
        return False
    return path.suffix in {".h", ".cpp", ".in", ".txt"} or path.name in {
        "CMakeLists.txt",
        "CMakePresets.json",
        "README.md",
    }


def _copy_extension(build_dir: Path, destination: Path) -> None:
    candidates = sorted(build_dir.glob("veqlib_ext*.so"))
    if not candidates:
        raise KernelBuildError(f"CMake build did not produce veqlib_ext*.so in {build_dir}")
    shutil.copy2(candidates[0], destination)


def _run_logged(command: list[str], log_path: Path, *, cwd: Path) -> None:
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(
        f"# started {started}\n# cwd {cwd}\n# command {' '.join(command)}\n\n{completed.stdout}"
    )
    if completed.returncode != 0:
        raise KernelBuildError(
            f"command failed with exit code {completed.returncode}; see {log_path}: "
            + " ".join(command)
        )


@contextlib.contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _command_version(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.splitlines()[0] if completed.stdout else None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cmake_list(values: tuple[int, ...]) -> str:
    return ";".join(str(value) for value in values) if values else "0"


def _default_source_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "veqlib"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise KernelBuildError(f"expected JSON object in {path}")
    return data


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
