from __future__ import annotations

from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from veqpy.model import Geqdsk

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize("filename", ["SOLOVEV.geqdsk", "CHEASE.geqdsk", "EFIT.geqdsk"])
def test_geqdsk_read_write_roundtrip(filename: str, tmp_path: Path) -> None:
    source = Geqdsk(DATA_DIR / filename)
    source.check()

    outpath = tmp_path / filename
    source.write(outpath)
    restored = Geqdsk(outpath)
    restored.check()

    assert restored.NR == source.NR
    assert restored.NZ == source.NZ
    assert restored.header == source.header
    assert_allclose(restored.F, source.F, rtol=1e-8, atol=1e-8)
    assert_allclose(restored.P, source.P, rtol=1e-8, atol=1e-8)
    assert_allclose(restored.FF_psi, source.FF_psi, rtol=1e-8, atol=1e-8)
    assert_allclose(restored.P_psi, source.P_psi, rtol=1e-8, atol=1e-8)
    assert_allclose(restored.q, source.q, rtol=1e-8, atol=1e-8)
    assert_allclose(restored.psi, source.psi, rtol=1e-8, atol=1e-8)
    assert_allclose(restored.boundary, source.boundary, rtol=1e-8, atol=1e-8)
    assert_allclose(restored.limiter, source.limiter, rtol=1e-8, atol=1e-8)


def test_geqdsk_check_rejects_shape_mismatch() -> None:
    geqdsk = Geqdsk(NR=2, NZ=2, F=[1.0], P=[1.0], FF_psi=[1.0], P_psi=[1.0], q=[1.0])
    with pytest.raises(ValueError, match="F must have length"):
        geqdsk.check()
