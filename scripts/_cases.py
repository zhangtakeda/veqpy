"""Manuscript case definitions shared by figure scripts."""

from __future__ import annotations

from _common import data_path
from _plotting import LINESTYLE_C

MU0 = 4.0e-7 * 3.141592653589793

CASE_KEYS = ("solovev", "chease", "efit")
CASE_LABELS = {
    "solovev": "D-shape",
    "chease": "H-mode",
    "efit": "X-point",
}
REFERENCE_LABELS = {case_key: "GEQDSK" for case_key in CASE_KEYS}
CASE_COLORS = {
    "solovev": "#1f77b4",
    "chease": "#ff7f0e",
    "efit": "#2ca02c",
}
CASE_LINESTYLES = {
    "solovev": "-",
    "chease": "--",
    "efit": LINESTYLE_C,
}
CASE_LINE_COLORS = {
    "solovev": ("#101010", "#777777", "#74a9cf", "#1f77b4", "#08306b"),
    "chease": ("#101010", "#777777", "#fdb863", "#ff7f0e", "#7f2704"),
    "efit": ("#101010", "#777777", "#74c476", "#2ca02c", "#00441b"),
}
CONFIG_LABELS = ("Low", "Medium", "High", "Ref")
REDUCED_CONFIG_LABELS = ("Low", "Medium", "High")
CONFIG_LINE_COLORS = {
    "Low": "#777777",
    "Medium": "#999999",
    "High": "#555555",
    "Ref": "#111111",
}
LOW_LINESTYLE = (0, (5, 1.6, 1.2, 1.6, 1.2, 1.6))
LEVEL_LINESTYLES = {
    "Ref": ":",
    "Low": LOW_LINESTYLE,
    "Medium": "--",
    "High": "-",
}

CASE_REFERENCE_GFILES = {
    "solovev": data_path("SOLOVEV.geqdsk"),
    "chease": data_path("CHEASE.geqdsk"),
    "efit": data_path("EFIT.geqdsk"),
}
CASE_REFERENCE_EQUILIBRIUM_JSONS = {
    "solovev": data_path("solovev-equilibrium.json"),
    "chease": data_path("chease-equilibrium.json"),
    "efit": data_path("efit-equilibrium.json"),
}
REFERENCE_EQUILIBRIUM_MANIFEST_PATH = data_path("reference_equilibria.json")
CASE_REFERENCE_PROFILE_LENGTHS = {
    "demo(psin)": {
        "psin": [0.0] * 6,
        "h": [0.0] * 3,
        "k": [0.0] * 6,
        "s1": [0.0] * 3,
    },
    "demo(rho)": {
        "h": [0.0] * 3,
        "k": [0.0] * 6,
        "s1": [0.0] * 3,
    },
    "solovev": {
        "psin": 10,
        "h": 10,
        "k": 10,
        "s1": 10,
        "s2": 5,
        "s3": 5,
        "s4": 5,
        "s5": 5,
        "s6": 5,
        "s7": 5,
        "s8": 5,
    },
    "chease": {
        "psin": 10,
        "h": 10,
        "k": 10,
        "v": 10,
        "c0": 10,
        "c1": 5,
        "c2": 5,
        "c3": 5,
        "c4": 5,
        "c5": 5,
        "c6": 5,
        "c7": 5,
        "s1": 10,
        "s2": 5,
        "s3": 5,
        "s4": 5,
        "s5": 5,
        "s6": 5,
        "s7": 5,
        "s8": 5,
    },
    "efit": {
        "psin": 10,
        "h": 10,
        "k": 10,
        "v": 10,
        "c0": 10,
        "c1": 5,
        "c2": 5,
        "c3": 5,
        "c4": 5,
        "c5": 5,
        "c6": 5,
        "c7": 5,
        "s1": 10,
        "s2": 5,
        "s3": 5,
        "s4": 5,
        "s5": 5,
        "s6": 5,
        "s7": 5,
        "s8": 5,
    },
}
REDUCED_EQUILIBRIUM_JSON_TEMPLATE = data_path("pareto_reduced_{case_key}_{config_label}.json")
REDUCED_EQUILIBRIUM_MANIFEST_PATH = data_path("pareto_reduced_equilibria.json")
DEFAULT_JSON_STEM = data_path("pareto")

REFERENCE_LAYOUT_NR = 32
REFERENCE_LAYOUT_NT = 32
REFERENCE_SOLVER_MAXFEV = 2000
SOLVER_INITIAL_POLICY = "auto"
TEST_SOURCE_SAMPLE_COUNT = 51
BOUNDARY_MAXTOL = 1.0
CASE_SOLVER_METHODS = {
    "solovev": "hybr",
    "efit": "hybr",
    "chease": "hybr",
}
CASE_BOUNDARY_FIT_M = {
    "solovev": 10,
    "chease": 10,
    "efit": 10,
}
CASE_BOUNDARY_FIT_N = {
    "solovev": 10,
    "chease": 10,
    "efit": 10,
}

DEMO_GRID = {
    "Nr": 64,
    "Nt": 64,
    "quadrature_scheme": "legendre",
}
DEMO_SNAPSHOT_GRID = {
    "Nr": 128,
    "Nt": 256,
    "quadrature_scheme": "uniform",
}
DEMO_BOUNDARY = {
    "a": 1.05 / 1.85,
    "R0": 1.05,
    "Z0": 0.0,
    "B0": 3.0,
    "ka": 2.2,
    "s_offsets": (0.0, 0.5235987755982989),
}
DEMO_SOLVER_CONFIG = {
    "method": "hybr",
    "enable_verbose": False,
}
DEMO_PROFILE_COEFFS = {
    "psin": [0.0] * 5,
    "h": [0.0] * 3,
    "k": [0.0] * 5,
    "s1": [0.0] * 3,
}
DEMO_SOURCE_SAMPLE_COUNT = 128
DEMO_ROUTE = "PF"
DEMO_COORDINATE = "psin"
DEMO_NODES = "uniform"
DEMO_IP = 3.0e6
