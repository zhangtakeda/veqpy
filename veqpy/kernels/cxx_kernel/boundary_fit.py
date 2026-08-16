"""
Module: veqpy.kernels.cxx_kernel.boundary_fit

Role:
- Build and load the standalone native boundary fitters.

Notes:
- The fitter is topology-independent and accepts runtime R/Z scatter length plus
  runtime c/s orders.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import platform
import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from veqpy.kernels.boundary_fit import normalize_boundary_fit_method

from .builder import (
    DEFAULT_CXX_COMPILER,
    PrepareError,
    _command_version,
    _default_build_parallel_jobs,
    _elapsed_ms,
    _exclusive_lock,
    _file_sha256,
    _package_version,
    _read_json,
    _run_logged,
    _write_json,
    default_kernel_cache_root,
)

BOUNDARY_FIT_NATIVE_SCHEMA = "veqpy.boundary_fit_native.v1"
BOUNDARY_FIT_NATIVE_IDENTITY_SCHEMA = "veqpy.boundary_fit_native_identity.v1"
BOUNDARY_FIT_MODULE_NAME = "veqpy_boundary_fit_ext"
BOUNDARY_FIT_BUILD = "release-relaxed"
BOUNDARY_FIT_CMAKE_BUILD_TYPE = "Release"


def fit_boundary_params_cxx(
    R_boundary: Any,
    Z_boundary: Any,
    *,
    c_order: int,
    s_order: int,
    maxtol: float = 1.0e-2,
    method: str | None = "gnqr",
) -> dict[str, float | np.ndarray]:
    """Fit RZ boundary samples with the native boundary fitter."""

    R = np.ascontiguousarray(R_boundary, dtype=np.float64)
    Z = np.ascontiguousarray(Z_boundary, dtype=np.float64)
    if R.ndim != 1:
        raise ValueError(f"R_boundary must be 1D, got {R.shape}")
    if Z.ndim != 1:
        raise ValueError(f"Z_boundary must be 1D, got {Z.shape}")
    maxtol = float(maxtol)
    if maxtol <= 0.0:
        raise ValueError(f"maxtol must be positive, got {maxtol!r}")

    method = normalize_boundary_fit_method(method)

    native = _boundary_fit_module()
    if method == "qr":
        payload = native.fit_boundary_qr(R, Z, int(c_order), int(s_order))
    elif method == "gnqr":
        payload = native.fit_boundary_weighted_gnqr(R, Z, int(c_order), int(s_order))
    elif method == "least-square":
        payload = native.fit_boundary_least_square(R, Z, int(c_order), int(s_order))
    else:
        raise AssertionError(f"unhandled boundary fit method {method!r}")
    result = {
        "R0": float(payload["R0"]),
        "Z0": float(payload["Z0"]),
        "a": float(payload["a"]),
        "ka": float(payload["ka"]),
        "c_offsets": np.asarray(payload["c_offsets"], dtype=np.float64),
        "s_offsets": np.asarray(payload["s_offsets"], dtype=np.float64),
        "rms": float(payload["rms"]),
        "max_curve_error": float(payload["max_curve_error"]),
        "c_order": int(payload["c_order"]),
        "s_order": int(payload["s_order"]),
        "method": method,
    }
    if result["rms"] >= maxtol:
        warnings.warn(
            (
                f"Boundary fit RMS {float(result['rms']):.6e} exceeds maxtol "
                f"{maxtol:.6e} for c/s orders={c_order}/{s_order}"
            ),
            stacklevel=2,
        )
    return result


@lru_cache(maxsize=1)
def _boundary_fit_module() -> ModuleType:
    artifact = _get_or_build_boundary_fit_native()
    return _load_boundary_fit_module(
        Path(str(artifact["shared_library_path"])),
        artifact_id=str(artifact["artifact_id"]),
    )


def _get_or_build_boundary_fit_native() -> dict[str, Any]:
    cxx = DEFAULT_CXX_COMPILER
    core_dir = _core_dir()
    identity = _boundary_fit_identity(core_dir=core_dir, cxx=cxx)
    artifact_id = _compute_boundary_fit_id(identity)
    root_dir = default_kernel_cache_root() / BOUNDARY_FIT_BUILD / "_boundary_fit" / artifact_id
    build_dir = root_dir / "cmake-build"
    shared_library_path = root_dir / f"{BOUNDARY_FIT_MODULE_NAME}.so"
    metadata_path = root_dir / "metadata.json"
    lock_path = root_dir.with_suffix(".lock")

    root_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(lock_path):
        if _boundary_fit_artifact_is_reusable(metadata_path, shared_library_path, identity):
            metadata = _read_json(metadata_path)
            return {**metadata, "reused": True}

        started = time.perf_counter()
        cmake_path = root_dir / "CMakeLists.txt"
        configure_log_path = root_dir / "configure.log"
        build_log_path = root_dir / "build.log"
        cmake_path.write_text(_boundary_fit_cmake_project(), encoding="utf-8")
        configure = [
            "cmake",
            "-S",
            str(root_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_BUILD_TYPE={BOUNDARY_FIT_CMAKE_BUILD_TYPE}",
            f"-DCMAKE_CXX_COMPILER={cxx}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DVEQPY_CXX_CORE_DIR={core_dir}",
            f"-DVEQPY_BOUNDARY_FIT_SOURCE={core_dir / 'boundary_fit_bindings.cpp'}",
        ]
        build_command = [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            BOUNDARY_FIT_MODULE_NAME,
            "--parallel",
            str(_default_build_parallel_jobs()),
        ]
        _run_logged(configure, configure_log_path, cwd=root_dir)
        _run_logged(build_command, build_log_path, cwd=root_dir)
        _copy_boundary_fit_extension(build_dir, shared_library_path)
        metadata = {
            "schema": BOUNDARY_FIT_NATIVE_SCHEMA,
            "artifact_id": artifact_id,
            "status": "built",
            "identity": identity,
            "shared_library_path": str(shared_library_path),
            "shared_library_sha256": _file_sha256(shared_library_path),
            "configure": configure,
            "build_command": build_command,
            "elapsed_ms": _elapsed_ms(started),
            "reused": False,
        }
        _write_json(metadata_path, metadata)
        return metadata


def _boundary_fit_identity(*, core_dir: Path, cxx: str) -> dict[str, Any]:
    return {
        "schema": BOUNDARY_FIT_NATIVE_IDENTITY_SCHEMA,
        "module": BOUNDARY_FIT_MODULE_NAME,
        "build": BOUNDARY_FIT_BUILD,
        "cmake_build_type": BOUNDARY_FIT_CMAKE_BUILD_TYPE,
        "source": _boundary_fit_source_digest(core_dir),
        "cmake_project_sha256": hashlib.sha256(
            _boundary_fit_cmake_project().encode("utf-8")
        ).hexdigest(),
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
            "gcem": "cmake-find-package",
        },
    }


def _boundary_fit_source_digest(core_dir: Path) -> dict[str, Any]:
    retained = (
        "boundary_fit.h",
        "boundary_fit_bindings.cpp",
        "veq_numeric.h",
        "tensor.h",
    )
    digest = hashlib.sha256()
    for name in retained:
        path = core_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "schema": "veqpy.boundary_fit_native_source_digest.v1",
        "algorithm": "sha256",
        "files": retained,
        "sha256": digest.hexdigest(),
    }


def _compute_boundary_fit_id(identity: dict[str, Any]) -> str:
    data = json.dumps(identity, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    digest = hashlib.sha256(data).digest()
    return base64.b32encode(digest).decode("ascii").lower().rstrip("=")[:32]


def _boundary_fit_artifact_is_reusable(
    metadata_path: Path,
    shared_library_path: Path,
    identity: dict[str, Any],
) -> bool:
    if not metadata_path.exists() or not shared_library_path.exists():
        return False
    metadata = _read_json(metadata_path)
    if metadata.get("schema") != BOUNDARY_FIT_NATIVE_SCHEMA:
        return False
    if metadata.get("status") != "built":
        return False
    if metadata.get("identity") != identity:
        return False
    expected = metadata.get("shared_library_sha256")
    return isinstance(expected, str) and _file_sha256(shared_library_path) == expected


def _copy_boundary_fit_extension(build_dir: Path, destination: Path) -> None:
    candidates = sorted(build_dir.glob(f"{BOUNDARY_FIT_MODULE_NAME}*.so"))
    if not candidates:
        raise PrepareError(
            f"CMake build did not produce {BOUNDARY_FIT_MODULE_NAME}*.so in {build_dir}"
        )
    destination.write_bytes(candidates[0].read_bytes())


def _load_boundary_fit_module(path: Path, *, artifact_id: str) -> ModuleType:
    if not path.exists():
        raise ImportError(f"Cxx boundary fitter shared library is missing: {path}")
    module_name = f"veqpy._native_boundary_fit.{artifact_id}.{BOUNDARY_FIT_MODULE_NAME}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _core_dir() -> Path:
    return Path(__file__).resolve().parent / "core"


def _boundary_fit_cmake_project() -> str:
    return """cmake_minimum_required(VERSION 3.24)
project(veqpy_boundary_fit_native LANGUAGES CXX)

set(VEQPY_CXX_GCEM_ROOT "$ENV{HOME}/opt/gcem-install" CACHE PATH "GCEM install prefix")
if(VEQPY_CXX_GCEM_ROOT)
    list(PREPEND CMAKE_PREFIX_PATH "${VEQPY_CXX_GCEM_ROOT}")
endif()

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    RESULT_VARIABLE VEQPY_NANOBIND_CMAKE_RESULT
    OUTPUT_VARIABLE VEQPY_NANOBIND_CMAKE_DIR
    ERROR_VARIABLE VEQPY_NANOBIND_CMAKE_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)
if(NOT VEQPY_NANOBIND_CMAKE_RESULT EQUAL 0)
    message(FATAL_ERROR "nanobind --cmake_dir failed: ${VEQPY_NANOBIND_CMAKE_ERROR}")
endif()
set(nanobind_DIR "${VEQPY_NANOBIND_CMAKE_DIR}" CACHE PATH "nanobind CMake package directory")
find_package(nanobind CONFIG REQUIRED)
find_package(gcem CONFIG REQUIRED)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "VEQPy boundary fitter requires clang++")
endif()
if(NOT CMAKE_CXX_COMPILER_VERSION MATCHES "^22[.]")
    message(FATAL_ERROR "VEQPy boundary fitter requires LLVM/Clang 22")
endif()

nanobind_add_module(
    veqpy_boundary_fit_ext
    NB_SUPPRESS_WARNINGS
    "${VEQPY_BOUNDARY_FIT_SOURCE}"
)
target_include_directories(veqpy_boundary_fit_ext PRIVATE "${VEQPY_CXX_CORE_DIR}")
target_link_libraries(veqpy_boundary_fit_ext PRIVATE gcem)
set_target_properties(
    veqpy_boundary_fit_ext
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)
target_compile_options(
    veqpy_boundary_fit_ext
    PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-mtune=native>
        $<$<CONFIG:Release>:-mprefer-vector-width=256>
        $<$<CONFIG:Release>:-fstrict-aliasing>
        $<$<CONFIG:Release>:-fomit-frame-pointer>
        $<$<CONFIG:Release>:-funroll-loops>
        $<$<CONFIG:Release>:-fvectorize>
        $<$<CONFIG:Release>:-fslp-vectorize>
        $<$<CONFIG:Release>:-ffunction-sections>
        $<$<CONFIG:Release>:-fdata-sections>
        $<$<CONFIG:Release>:-ffast-math>
        $<$<CONFIG:Release>:-ffp-contract=fast>
        $<$<CONFIG:Release>:-funsafe-math-optimizations>
        $<$<CONFIG:Release>:-fno-math-errno>
        $<$<CONFIG:Release>:-fno-trapping-math>
        $<$<CONFIG:Release>:-fno-signed-zeros>
        $<$<CONFIG:Release>:-freciprocal-math>
        $<$<CONFIG:Release>:-ffinite-math-only>
        $<$<CONFIG:Release>:-fapprox-func>
        $<$<CONFIG:Release>:-flto=thin>
)
if(APPLE)
    target_link_options(
        veqpy_boundary_fit_ext
        PRIVATE
            $<$<CONFIG:Release>:-Wl,-dead_strip>
            $<$<CONFIG:Release>:-flto=thin>
    )
else()
    target_link_options(
        veqpy_boundary_fit_ext
        PRIVATE
            $<$<CONFIG:Release>:-Wl,-O3>
            $<$<CONFIG:Release>:-Wl,--gc-sections>
            $<$<CONFIG:Release>:-flto=thin>
            $<$<CONFIG:Release>:-fuse-ld=lld>
    )
endif()
"""
