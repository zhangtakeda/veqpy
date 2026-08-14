from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from veqpy.model import Equilibrium, Geometry, Grid, Profile


def _geometry(
    *,
    Nr: int = 12,
    Nt: int = 12,
    L_max: int = 2,
    M_max: int = 2,
    K_max: int | None = None,
) -> Geometry:
    return Geometry(
        Nr=Nr,
        Nt=Nt,
        radial_rule="legendre",
        radial_calculus="spectral",
        K_max=K_max,
        R0=3.0,
        Z0=0.2,
        a=1.0,
        kappa_lcfs=1.6,
        c_lcfs=np.zeros(M_max + 1),
        s_lcfs=np.zeros(M_max),
        h_coeffs=np.zeros(L_max + 1),
        v_coeffs=np.zeros(L_max + 1),
        kappa_coeffs=np.zeros(L_max + 1),
        c_coeffs=np.zeros((M_max + 1, L_max + 1)),
        s_coeffs=np.zeros((M_max, L_max + 1)),
    )


def test_geometry_derives_layout_and_default_grid() -> None:
    geometry = _geometry(Nr=12, Nt=10, L_max=3, M_max=2)

    assert geometry.L_max == 3
    assert geometry.M_max == 2
    assert geometry.r.shape == geometry.radial_weights.shape == (12,)
    assert geometry.theta.shape == (10,)
    assert geometry.accumulator.shape == geometry.differentiator.shape == (12, 12)
    assert_allclose(np.sum(geometry.radial_weights), 1.0, atol=1.0e-14)
    assert not geometry.r.flags.writeable
    assert not geometry.differentiator.flags.writeable


def test_geometry_dense_coefficients_materialize_elliptic_surfaces() -> None:
    geometry = _geometry()
    r = geometry.r[:, None]
    theta = geometry.theta[None, :]

    assert_allclose(geometry.h, 0.0)
    assert_allclose(geometry.v, 0.0)
    assert_allclose(geometry.kappa, geometry.kappa_lcfs)
    assert_allclose(geometry.R, geometry.R0 + geometry.a * r * np.cos(theta))
    assert_allclose(
        geometry.Z,
        geometry.Z0 - geometry.a * geometry.kappa_lcfs * r * np.sin(theta),
    )
    assert_allclose(geometry.R_lcfs, geometry.R0 + geometry.a * np.cos(geometry.theta))
    assert_allclose(
        geometry.Z_lcfs,
        geometry.Z0 - geometry.a * geometry.kappa_lcfs * np.sin(geometry.theta),
    )


def test_geometry_k_max_caps_fourier_radial_power() -> None:
    uncapped = _geometry(K_max=None)
    capped = _geometry(K_max=1)
    c_lcfs = np.array([0.0, 0.0, 0.25])
    uncapped.c_lcfs = c_lcfs
    capped.c_lcfs = c_lcfs

    assert_allclose(uncapped.c[2], 0.25 * uncapped.r**2)
    assert_allclose(capped.c[2], 0.25 * capped.r)


def test_geometry_canonicalizes_calculus_alias_and_accepts_nonbinding_k_max() -> None:
    geometry = _geometry()
    geometry.radial_calculus = "compact"

    assert geometry.radial_calculus == "cfd33"
    geometry.K_max = geometry.M_max
    geometry.check()
    assert_allclose(geometry.c_fields, _geometry(K_max=None).c_fields)


def test_geometry_matches_legacy_equilibrium_shape_materialization() -> None:
    geometry = _geometry(Nr=12, Nt=12, L_max=2, M_max=2, K_max=1)
    geometry.h_coeffs = np.array([0.05, -0.01, 0.002])
    geometry.v_coeffs = np.array([0.02, 0.005, -0.001])
    geometry.kappa_coeffs = np.array([0.04, -0.006, 0.001])
    geometry.c_lcfs = np.array([0.01, 0.03, -0.02])
    geometry.s_lcfs = np.array([0.025, -0.015])
    geometry.c_coeffs = np.array(
        [[0.005, -0.002, 0.001], [0.003, 0.001, -0.0005], [0.002, -0.001, 0.0002]]
    )
    geometry.s_coeffs = np.array(
        [[-0.002, 0.001, 0.0004], [0.0015, -0.0007, 0.0001]]
    )
    geometry.check()

    grid = Grid(
        Nr=geometry.Nr,
        Nt=geometry.Nt,
        L_max=geometry.L_max,
        M_max=geometry.M_max,
        K_max=geometry.K_max,
        quadrature_scheme=geometry.radial_rule,
        calculus_scheme=geometry.radial_calculus,
    )
    powers = grid.K_values
    shape_profiles = {
        "h": Profile(offset=0.0, coeff=geometry.h_coeffs),
        "v": Profile(offset=0.0, coeff=geometry.v_coeffs),
        "k": Profile(offset=geometry.kappa_lcfs, coeff=geometry.kappa_coeffs),
    }
    for order in range(geometry.M_max + 1):
        shape_profiles[f"c{order}"] = Profile(
            offset=float(geometry.c_lcfs[order]),
            power=int(powers[order]),
            coeff=geometry.c_coeffs[order],
        )
    for order in range(1, geometry.M_max + 1):
        shape_profiles[f"s{order}"] = Profile(
            offset=float(geometry.s_lcfs[order - 1]),
            power=int(powers[order]),
            coeff=geometry.s_coeffs[order - 1],
        )
    zeros = np.zeros(geometry.Nr, dtype=np.float64)
    legacy = Equilibrium(
        R0=geometry.R0,
        Z0=geometry.Z0,
        B0=3.0,
        a=geometry.a,
        grid=grid,
        shape_profiles=shape_profiles,
        FFn_psin=zeros,
        Pn_psin=zeros,
        psin=zeros,
        psin_r=zeros,
        psin_rr=zeros,
    )

    assert_allclose(geometry.h_fields, legacy._h_fields)
    assert_allclose(geometry.v_fields, legacy._v_fields)
    assert_allclose(geometry.kappa_fields, legacy._kappa_fields)
    assert_allclose(geometry.R, legacy.R)
    assert_allclose(geometry.Z, legacy.Z)


def test_geometry_owns_roots_and_freezes_after_validation() -> None:
    source = np.zeros((3, 3), dtype=np.float64)
    geometry = _geometry()
    geometry.c_coeffs = source
    source[0, 0] = 9.0

    assert geometry.c_coeffs[0, 0] == 0.0
    geometry.freeze()
    assert geometry.is_frozen
    assert not geometry.c_coeffs.flags.writeable
    with pytest.raises(AttributeError, match="is frozen"):
        geometry.R0 = 4.0
    with pytest.raises(ValueError, match="read-only"):
        geometry.c_coeffs[0, 0] = 1.0


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ({"s_coeffs": np.zeros((3, 3))}, "c_coeffs must store c0..cM"),
        ({"c_lcfs": np.zeros(2)}, "c_lcfs must have shape"),
        ({"h_coeffs": np.zeros(2)}, "h_coeffs must have shape"),
    ),
)
def test_geometry_rejects_inconsistent_dense_layout(
    replacement: dict[str, np.ndarray],
    message: str,
) -> None:
    geometry = _geometry()
    for name, value in replacement.items():
        setattr(geometry, name, value)
    with pytest.raises(ValueError, match=message):
        geometry.check()


def test_geometry_serializes_only_authoritative_roots(tmp_path: Path) -> None:
    geometry = _geometry()
    _ = geometry.R
    path = tmp_path / "geometry.json"

    geometry.write(path)
    payload = json.loads(path.read_text())["Geometry"]
    restored = Geometry.load(path)

    assert set(payload) == set(Geometry.serial_attributes())
    assert "r" not in payload
    assert "radial_weights" not in payload
    assert "differentiator" not in payload
    assert restored.L_max == geometry.L_max
    assert restored.M_max == geometry.M_max
    assert_allclose(restored.R, geometry.R)
