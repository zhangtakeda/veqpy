# Baseline Blocker Report: Solver Auto Policy Threshold

## Summary

Phase 0A investigated the pre-existing baseline failure in:

```text
tests/test_solver_api.py::test_solver_auto_policy_selects_refined_only_above_curve_strain_threshold
```

This blocker occurred before any JAX/backend work started. The fix is limited to
the existing solver API test fixture and does not change solver implementation or
JAX/backend code.

## Environment

- Worktree: `/Users/yang/mie_veq_work/veqpy_jax_phase0_8`
- Branch: `feature/jax-backend-autonomous-phase0-8`
- Base commit: `c4a54fb16524386f516d1099c3166467099fa107`
- Python used: `3.13.12`
- `pyproject.toml` `requires-python`: `>=3.12`
- Python classifiers: `3.12`, `3.13`
- Python 3.13.12 support: supported by project metadata
- Python 3.14 support: not advertised by classifiers; treat as unsupported or
  experimental until project metadata says otherwise
- JAX installed: `False`

The failure reproduced deterministically under the supported Python 3.13.12
environment, so it is not classified as an unsupported-Python-only failure.

## Reproduction

Single-test reproduction:

```bash
.venv/bin/python -m pytest -q \
  tests/test_solver_api.py::test_solver_auto_policy_selects_refined_only_above_curve_strain_threshold \
  -vv --tb=long
```

Whole solver API file:

```bash
.venv/bin/python -m pytest -q tests/test_solver_api.py -vv --tb=short
```

Determinism check:

```bash
for i in 1 2 3 4 5; do
  .venv/bin/python -m pytest -q \
    tests/test_solver_api.py::test_solver_auto_policy_selects_refined_only_above_curve_strain_threshold || break
done
```

The failure was deterministic. The assertion failed at
`tests/test_solver_api.py:363`, where `large_projection_auto` was expected to
match the zero initial state, but auto selected the geometric-refined state.

Observed values:

```text
large_projection_auto = [0.19183495, 0., -0., 0.35472297, 0., 0., 0., 0., 0.]
large_projection_zeros = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
```

## Intended Contract

The test contract is:

- below `_AUTO_CURVE_STRAIN_THRESHOLD`, `initial_policy="auto"` should use the
  zero initial state;
- at or above `_AUTO_CURVE_STRAIN_THRESHOLD`, `initial_policy="auto"` should use
  the geometric-refined initial state;
- both sine and cosine boundary offsets may contribute to the curve-strain
  decision.

This contract is consistent with the implementation in
`veqpy/solver/solver.py`, where `auto` selects geometric-refined when:

```text
_boundary_curve_strain(boundary) >= _AUTO_CURVE_STRAIN_THRESHOLD
```

## Root Cause Classification

Classification: **E. Test construction bug**.

The test fixture intended to represent a below-threshold cosine-offset case, but
the selected value was actually above the current curve-strain threshold.

Measured strain values:

```text
threshold                       0.2
moderate                        0.06406805992590138
old large projection c=0.6      0.2563713149357272
fixed large projection c=0.4    0.17363315800764298
high_s                          0.34496530340851517
high_c                          0.3325337640880544
ellipse                         0.00018838515623806847
```

Because `c_offsets=[0.0, 0.6]` produces strain `0.256371...`, the production
implementation correctly selected the refined initial state. The surrounding
`high_c` case expects a cosine offset above threshold to select refined, so this
is not an implementation bug in the c-offset path.

## Fix Applied

The below-threshold cosine fixture was changed from `c=0.6` to `c=0.4`, whose
curve strain is `0.173633... < 0.2`.

The test now also asserts the intended below/above-threshold conditions directly
using `_boundary_curve_strain(...)` and `_AUTO_CURVE_STRAIN_THRESHOLD`. This keeps
the behavior assertion strict while preventing the synthetic fixture from
silently drifting across the threshold again.

Files changed:

```text
tests/test_solver_api.py
docs/baseline_blocker_solver_auto_policy_report.md
```

Solver behavior changed: **No**.

JAX/backend work started: **No**.

## Validation

Phase 0A targeted validation:

```bash
.venv/bin/python -m pytest -q \
  tests/test_solver_api.py::test_solver_auto_policy_selects_refined_only_above_curve_strain_threshold \
  -vv --tb=long
```

Result:

```text
1 passed
```

The full Phase 0 baseline validation is recorded in
`docs/jax_backend_autonomous_report.md` after this fix.
