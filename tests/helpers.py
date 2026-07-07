from __future__ import annotations

import numpy as np

MU0 = 4.0e-7 * np.pi


def pf_reference_profiles(psin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75
    alpha_p, alpha_f = 5.0, 3.32
    exp_ap, exp_af = np.exp(alpha_p), np.exp(alpha_f)
    den_p = 1.0 + exp_ap * (alpha_p - 1.0)
    den_f = 1.0 + exp_af * (alpha_f - 1.0)
    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    return current_input.astype(np.float64), heat_input.astype(np.float64)
