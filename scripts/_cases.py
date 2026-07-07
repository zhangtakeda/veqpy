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
CASE_SOLVER_METHODS = {
    "solovev": "hybr",
    "efit": "hybr",
    "chease": "hybr",
}

CASE_BOUNDARY_PARAMETERS = {
    "solovev": {
        "a": 1.999991815361528,
        "R0": 6.199980064550139,
        "Z0": -4.0265979758802695e-05,
        "ka": 1.6999963725270892,
        "fit_rms": 0.0003896377390478151,
        "fit_max_curve_error": 0.010740887012215919,
        "c_offsets": (
            -5.2020273089148361e-06,
            7.1653084700073520e-05,
            9.0046463771700606e-06,
            -9.7789815345071059e-06,
            -2.4974793964430963e-05,
            6.9834184194999052e-05,
            7.2729999976377730e-06,
            -1.0542591490004266e-05,
            -2.4922126347913156e-05,
            6.9894200998597234e-05,
            -6.1688729054668674e-06,
        ),
        "s_offsets": (
            0.0,
            3.3418610987342229e-01,
            1.1666700286207601e-03,
            -2.0450964699343364e-03,
            -1.5833379968858402e-04,
            -3.8596504114596517e-05,
            -4.1901619591027170e-05,
            1.4968468916007242e-05,
            2.6075948416621554e-05,
            -4.8303378655747770e-05,
            -3.4317843409700719e-05,
        ),
    },
    "chease": {
        "a": 0.6504127010781183,
        "R0": 0.9999628382309164,
        "Z0": 0.00016201215320952212,
        "ka": 1.8353512314259297,
        "fit_rms": 0.000690058644014939,
        "fit_max_curve_error": 0.012183767841949586,
        "c_offsets": (
            -0.10093500947178713,
            0.09953753013381786,
            0.00263797964542851,
            0.0002364648141952,
            -0.00187058163436749,
            -0.00015749468335108,
            0.00250455340197678,
            -0.00021010577314975,
            -0.00138185135101046,
            0.00046348834914139,
            0.0010925753359023,
        ),
        "s_offsets": (
            0.0,
            0.39741190586126168,
            0.30000064059401577,
            -0.19752465029931532,
            0.00012375584334904224,
            -0.0028918496083633863,
            0.00055728321961535845,
            0.0020885844545475403,
            -0.00065686603212885782,
            -0.0016236488464970841,
            0.0010090121427006112,
        ),
    },
    "efit": {
        "a": 0.6171676117603371,
        "R0": 1.6613762798644713,
        "Z0": -0.08363305046404519,
        "ka": 1.7821260070974403,
        "fit_rms": 0.00028107711744079425,
        "fit_max_curve_error": 0.001729559104966416,
        "c_offsets": (
            0.07852828254669175,
            0.06312715933508059,
            -0.07905163910660493,
            -0.01769724397809446,
            0.02910897204124359,
            0.02433733959900054,
            -0.0055909769586503,
            -0.00879177546686944,
            0.00550482646737737,
            0.00504000048961428,
            0.00162409293766496,
        ),
        "s_offsets": (
            0.0,
            0.6133689729358927,
            0.04214392213264707,
            -0.12629878943384087,
            0.01888688896953439,
            0.02390779724943099,
            0.02772395371056901,
            -0.00865560279811922,
            -0.00225943350361505,
            0.00283741776166376,
            0.004692674167182,
        ),
    },
}
CASE_BOUNDARY_C_ORDER = {
    case_key: len(params["c_offsets"]) - 1
    for case_key, params in CASE_BOUNDARY_PARAMETERS.items()
}
CASE_BOUNDARY_S_ORDER = {
    case_key: len(params["s_offsets"]) - 1
    for case_key, params in CASE_BOUNDARY_PARAMETERS.items()
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
