"""Command-line diagnostics for the standalone VEQPy Module."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .demo_case import make_demo_plasma
from .kernels import KernelConfig, KernelTopology
from .module import VEQ


def main(argv: list[str] | None = None) -> int:
    """Run one of the stable VEQPy package checks."""

    parser = argparse.ArgumentParser(prog="python -m veqpy")
    parser.add_argument("--version", action="store_true", help="print the package version")
    parser.add_argument("--check", action="store_true", help="run the Numba ABI smoke check")
    parser.add_argument("--links", action="store_true", help="print project links")
    parser.add_argument("--demo", choices=("numba", "cxx"), help="run the minimal Module demo")
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


def _run_demo(backend: str, *, check_only: bool) -> int:
    topology = KernelTopology(
        h_count=2,
        v_count=0,
        kappa_count=2,
        psin_count=3,
        F_count=0,
        c_counts=(),
        s_counts=(2, 2),
        Nr=8,
        Nt=12,
        route="PF",
        coordinate="psin",
        nodes="uniform",
        constraint="ip",
        sample_count=8,
    )
    try:
        module = VEQ(
            topology=topology,
            backend=backend,
            config=KernelConfig(max_evaluations=800),
        )
        result = module.run(plasma=make_demo_plasma(), materialize=not check_only)
    except Exception as error:  # pragma: no cover - environment-specific Cxx diagnostics
        print(f"VEQPy {backend} backend unavailable: {type(error).__name__}: {error}", file=sys.stderr)
        return 2 if backend == "cxx" else 1
    finally:
        if "module" in locals():
            module.close()
    print(f"backend={backend} solved={result.solved} residual={result.residual_norm:.3e}")
    return 0 if result.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
