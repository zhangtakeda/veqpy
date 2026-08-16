"""Standalone VEQPy command-line diagnostics and demo runner."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .api import build
from .demo_case import make_demo_inputs


def main(argv: list[str] | None = None) -> int:
    """Run one stable VEQPy package check."""

    parser = argparse.ArgumentParser(prog="python -m veqpy")
    parser.add_argument("--version", action="store_true", help="print the package version")
    parser.add_argument("--check", action="store_true", help="run the Numba smoke check")
    parser.add_argument("--links", action="store_true", help="print project links")
    parser.add_argument(
        "--demo",
        choices=("numba", "cxx", "cxx-strict", "cxx-relaxed", "cxx-enzyme"),
        help="run the minimal Module demo",
    )
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.links:
        print("Homepage: https://zhangtakeda.github.io")
        print("Repository: https://github.com/zhangtakeda/veqpy")
        return 0
    if args.check:
        return _run_demo("numba", check_only=True)
    if args.demo:
        return _run_demo(args.demo, check_only=False)
    parser.print_help()
    return 0


def _demo_topology() -> dict[str, object]:
    """Return the small dict topology used by all CLI backends."""

    return {
        "h_count": 2,
        "v_count": 0,
        "kappa_count": 2,
        "psin_count": 3,
        "F_count": 0,
        "c_counts": (),
        "s_counts": (2, 2),
        "Nr": 8,
        "Nt": 12,
        "route": "PF",
        "coordinate": "psin",
        "constraint": "ip",
    }


def _run_demo(backend: str, *, check_only: bool) -> int:
    """Run a minimal independent Module smoke check."""

    module = None
    try:
        module = build(
            topology=_demo_topology(),
            solver={"max_evaluations": 800},
            backend=backend,
            materialize=not check_only,
            verbose=True,
            report=False,
        )
        boundary, source, targets = make_demo_inputs()
        result = module.solve(
            boundary=boundary,
            source=source,
            targets=targets,
            materialize=not check_only,
            verbose=True,
            report=False,
        )
    except Exception as error:  # pragma: no cover - environment-specific Cxx diagnostics
        print(
            f"VEQPy {backend} backend unavailable: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2 if backend.startswith("cxx") else 1
    finally:
        if module is not None:
            module.close()
    print(f"backend={backend} solved={result.solved} residual={result.residual_norm:.3e}")
    return 0 if result.accepted else 1


__all__ = ["main"]
