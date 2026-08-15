from __future__ import annotations

from pathlib import Path

import numpy as np

from veqpy import VEQ, Geqdsk, KernelTopology
from veqpy.demo_case import make_demo_plasma
from veqpy.io import export_geqdsk


def _topology() -> KernelTopology:
    return KernelTopology(
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


def test_base_equilibrium_geqdsk_export_roundtrip(tmp_path: Path) -> None:
    module = VEQ(topology=_topology())
    try:
        record = module.run(plasma=make_demo_plasma())
        assert record.equilibrium is not None
        geqdsk = export_geqdsk(
            record.equilibrium,
            R_range=(2.0, 4.0),
            Z_range=(-1.5, 1.5),
            NR=32,
            NZ=32,
            header="VEQPy 2.x test",
        )
    finally:
        module.close()
    assert geqdsk.boundary.shape == (65, 2)
    assert np.array_equal(geqdsk.boundary[0], geqdsk.boundary[-1])
    path = tmp_path / "roundtrip.geqdsk"
    geqdsk.write(path)
    restored = Geqdsk(path)
    restored.check()
    assert (restored.NR, restored.NZ) == (32, 32)
    assert restored.psi.shape == (32, 32)
    assert np.all(np.isfinite(restored.P_psi))


def test_bundled_geqdsk_is_a_real_versioned_fixture() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "SOLOVEV.geqdsk"
    geqdsk = Geqdsk(path)
    geqdsk.check()
    assert geqdsk.NR == geqdsk.NZ
    assert geqdsk.boundary.shape[0] > 4
    assert np.isfinite(geqdsk.Bt0)
