from __future__ import annotations

import veqlib_ext

POWELL = 1
LEVENBERG_MARQUARDT = 2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_enzyme_jacobian_fast_path(
    result: tuple[object, ...], x_size: int, *, max_njev: int
) -> None:
    nfev = int(result[3])
    njev = int(result[4])
    jacobian_component_evaluations = int(result[6])
    _require(0 < njev <= max_njev, f"unexpected Enzyme Jacobian evaluation count: {njev}")
    _require(
        nfev <= max_njev * x_size,
        f"Enzyme Jacobian path appears to have fallen back: {nfev=}",
    )
    _require(
        jacobian_component_evaluations == x_size * njev,
        "Jacobian component evaluations should match dense Enzyme Jacobian calls",
    )


def main() -> int:
    kernel = veqlib_ext.KernelSolver(POWELL)
    meta = kernel.metadata()
    x_size = int(meta["x_size"])

    powell_result = kernel.solve_direct()
    lm_result = veqlib_ext.KernelSolver(LEVENBERG_MARQUARDT).solve_direct()

    _require(meta["route"] == "PF/psin/uniform/Ip", f"unexpected route: {meta['route']!r}")
    _require(len(powell_result) == 15, f"unexpected solve tuple length: {len(powell_result)}")
    _require(bool(powell_result[1]), f"Powell solve failed: info={powell_result[2]}")
    _require(bool(lm_result[1]), f"LM solve failed: info={lm_result[2]}")
    _require(powell_result[11].shape == (x_size,), "packed solution view has wrong shape")
    _require(powell_result[14].shape == (2,), "alpha view has wrong shape")
    _require(not powell_result[11].flags.writeable, "packed solution view should be read-only")

    if "Enzyme" in meta["solver"]["jacobian"]:
        _check_enzyme_jacobian_fast_path(powell_result, x_size, max_njev=2)
        _check_enzyme_jacobian_fast_path(lm_result, x_size, max_njev=x_size)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
