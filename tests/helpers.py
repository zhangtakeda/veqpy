from __future__ import annotations

import numpy as np

from veqpy.model import Boundary, Grid, Profile
from veqpy.operator import Operator, OperatorCase

MU0 = 4.0e-7 * np.pi


def profiles(coeffs: dict[str, int | list[float] | np.ndarray | None]) -> dict[str, Profile]:
    result: dict[str, Profile] = {}
    for name, coeff in coeffs.items():
        if isinstance(coeff, int):
            result[name] = Profile(coeff=np.zeros(coeff, dtype=np.float64))
        else:
            result[name] = Profile(coeff=coeff)
    return result


def tiny_boundary() -> Boundary:
    return Boundary(
        a=0.5,
        R0=1.0,
        Z0=0.0,
        B0=3.0,
        ka=1.7,
        s_offsets=np.array([0.0, np.arcsin(0.2)], dtype=np.float64),
    )


def pf_reference_profiles(psin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75
    alpha_p, alpha_f = 5.0, 3.32
    exp_ap, exp_af = np.exp(alpha_p), np.exp(alpha_f)
    den_p = 1.0 + exp_ap * (alpha_p - 1.0)
    den_f = 1.0 + exp_af * (alpha_f - 1.0)
    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    return current_input.astype(np.float64), heat_input.astype(np.float64)


def tiny_pf_case() -> OperatorCase:
    psin = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    ffn_psin, pn_psin = pf_reference_profiles(psin)
    return OperatorCase(
        route="PF",
        coordinate="psin",
        nodes="uniform",
        profiles=profiles({
            "psin": 3,
            "h": [0.0, 0.0],
            "k": [0.0, 0.0],
            "s1": [0.0, 0.0],
        }),
        boundary=tiny_boundary(),
        heat_input=pn_psin / MU0,
        current_input=ffn_psin,
        Ip=3.0e6,
    )


def tiny_grid() -> Grid:
    return Grid(Nr=8, Nt=8, L_max=3, M_max=2, K_max=1, quadrature_scheme="legendre")


def tiny_operator() -> Operator:
    return Operator(tiny_grid(), tiny_pf_case())
