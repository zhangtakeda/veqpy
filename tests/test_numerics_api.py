from __future__ import annotations

from veqpy.numerics import normalize_source_interpolation_kind


def test_source_interpolation_kind_normalizes_canonical_tokens() -> None:
    assert normalize_source_interpolation_kind("barycentric") == "barycentric"
    assert normalize_source_interpolation_kind("NOT_A_KNOT") == "not-a-knot"
    assert normalize_source_interpolation_kind("Linear") == "linear"
    assert normalize_source_interpolation_kind("quadratic") == "quadratic"
    assert normalize_source_interpolation_kind("CUBIC") == "cubic"
