# JAX Backend Autonomous Report

## Status

**Final status: PASS for the first numerical JAX milestone.**

Mandatory Numba/default-path gates passed. A separate JAX-enabled CPU virtual
environment was created, and PF/rho/grid JAX stage parity, fused residual parity,
and SciPy solver smoke now pass. JAX remains optional and lazy-imported; no
JAX-native nonlinear solver is implemented.

The initial repository baseline failed before any JAX/backend code behavior
changes were made. Per the Phase 0 gate, implementation did not proceed to Phase
1 until the pre-existing solver auto-policy blocker was triaged and fixed in
Phase 0A.

## Base And Branch

- Requested base: `develop @ c4a54f`
- Actual base commit: `c4a54fb16524386f516d1099c3166467099fa107`
- Branch: `feature/jax-backend-autonomous-phase0-8`
- Worktree: `/Users/yang/mie_veq_work/veqpy_jax_phase0_8`

The original `/Users/yang/mie_veq_work/veqpy` worktree had unrelated local
changes from prior work, so this task used a separate clean worktree.

## Environment

- Worktree-local `.venv`: created with Python 3.13.12 after the initial
  sibling-venv preflight reproduced the same baseline failure.
- `python` command: not present on this machine
- Python used for final preflight: `.venv/bin/python`
- Python version: `3.13.12`
- JAX installed in original `.venv`: `False`
- JAX-enabled parity venv: `.venv-jax`
- `.venv-jax` Python version: `3.13.12`
- `.venv-jax` JAX version: `0.10.1`
- `.venv-jax` JAX devices: `[CpuDevice(id=0)]`
- `.venv-jax` platform: `cpu`
- `.venv-jax` x64: `True`

## Initial Decisions

- JAX optional dependency policy: JAX must remain optional and lazy-imported.
- Missing JAX error type: planned as `MissingOptionalBackendError`.
- First supported JAX route: `PF/rho/grid`.
- Unsupported public method behavior: raise `UnsupportedBackendFeature`, never
  read stale Numba workspaces.
- JAX-native nonlinear solver: explicitly out of scope.

## Phase 0 Baseline Commands

```bash
git rev-parse HEAD
git status --short
.venv/bin/python -m compileall -q veqpy tests
.venv/bin/ruff check veqpy tests
.venv/bin/python -m pytest -q tests/test_public_api.py tests/test_operator_public_api.py
.venv/bin/python -m pytest -q tests/test_engine_source_registry.py tests/test_packed_layout_api.py
.venv/bin/python -m pytest -q tests/test_solver_api.py
```

## Phase 0 Results

- `git rev-parse HEAD`: `c4a54fb16524386f516d1099c3166467099fa107`
- Initial `git status --short`: clean
- `compileall`: passed
- `ruff check veqpy tests`: passed
- `tests/test_public_api.py tests/test_operator_public_api.py`: 18 passed
- `tests/test_engine_source_registry.py tests/test_packed_layout_api.py`: 8 passed
- `tests/test_solver_api.py`: 12 passed, 1 failed

Failing test:

```text
tests/test_solver_api.py::test_solver_auto_policy_selects_refined_only_above_curve_strain_threshold
```

Failure summary:

```text
large_projection_auto != large_projection_zeros

large_projection_auto:
[ 0.19183495, 0., -0., 0.35472297, 0., 0., 0., 0., 0. ]

large_projection_zeros:
[0., 0., 0., 0., 0., 0., 0., 0., 0.]
```

The same baseline failure was first observed with the sibling worktree venv
Python 3.14.5, then reproduced after creating a fresh Python 3.13.12 `.venv`
inside this worktree. The final raw output is recorded in:

```text
/Users/yang/Desktop/项目/瞬原/.codex-harness/veqpy-jax-backend-phase0-8/logs/phase0_baseline_py313.log
```

## Phase 0A Baseline Blocker Resolved

Root cause: **test construction bug**.

The failing fixture intended to represent a below-threshold cosine-offset case,
but `c_offsets=[0.0, 0.6]` produced curve strain `0.256371...`, which is above
the production threshold `0.2`. The solver therefore correctly selected the
geometric-refined initial state. Surrounding coverage already expects cosine
offsets above threshold to select the refined path.

Fix:

- changed the below-threshold cosine fixture to `c_offsets=[0.0, 0.4]`, whose
  curve strain is `0.173633...`;
- added explicit below/above threshold assertions to the solver API test so the
  fixture cannot silently drift across the threshold.

Files changed:

```text
tests/test_solver_api.py
docs/baseline_blocker_solver_auto_policy_report.md
docs/jax_backend_autonomous_report.md
```

Solver behavior changed: **No**.

JAX work started before baseline fix: **No**.

Phase 0A validation:

```bash
.venv/bin/python -m pytest -q \
  tests/test_solver_api.py::test_solver_auto_policy_selects_refined_only_above_curve_strain_threshold \
  -vv --tb=long
.venv/bin/python -m pytest -q tests/test_solver_api.py -vv --tb=short
.venv/bin/python -m compileall -q veqpy tests
.venv/bin/ruff check veqpy tests
.venv/bin/python -m pytest -q tests/test_public_api.py tests/test_operator_public_api.py
.venv/bin/python -m pytest -q tests/test_engine_source_registry.py tests/test_packed_layout_api.py
.venv/bin/python -m pytest -q tests/test_solver_api.py
.venv/bin/python -m pytest -q \
  tests/test_solver_api.py::test_solver_auto_policy_selects_refined_only_above_curve_strain_threshold \
  -vv
```

Results:

```text
targeted solver auto-policy test: passed
tests/test_solver_api.py: 13 passed
compileall: passed
ruff: passed
tests/test_public_api.py tests/test_operator_public_api.py: 18 passed
tests/test_engine_source_registry.py tests/test_packed_layout_api.py: 8 passed
tests/test_solver_api.py: 13 passed
```

Decision: `BASELINE_FIXED_CONTINUING_TO_PHASE_1`.

## Files Changed Through Phase 0A

- `docs/jax_backend_autonomous_report.md`
- `docs/baseline_blocker_solver_auto_policy_report.md`
- `tests/test_solver_api.py`

No backend implementation files were changed through Phase 0A.

## Phase 1 Numba-Default Backend Selection Shell

Phase 1 introduced backend vocabulary without changing the active default
runtime.

Files changed:

```text
veqpy/engine/backend.py
veqpy/layout/binding.py
veqpy/operator/operator.py
tests/test_backend_selection.py
tests/test_model_backend_free.py
```

Implemented behavior:

- `Operator(..., backend="numba")` is accepted and matches default behavior.
- `Operator(..., backend="jax")` is recognized as a valid backend spelling but
  raises `UnsupportedBackendFeature` before an unsupported JAX runtime is used.
- Invalid backend names raise `InvalidBackendError`.
- Backend selection belongs to `Operator`, not `Problem`.
- `Problem`, `Grid`, `Profile`, `Boundary`, `Geqdsk`, and `Equilibrium` remain
  backend-free public model objects.
- `veqpy`, `veqpy.model`, `veqpy.operator`, and `veqpy.solver` imports do not
  import JAX.

Validation:

```bash
.venv/bin/python -m compileall -q veqpy tests
.venv/bin/ruff check veqpy tests
.venv/bin/python -m pytest -q tests/test_public_api.py tests/test_operator_public_api.py
.venv/bin/python -m pytest -q tests/test_engine_source_registry.py tests/test_packed_layout_api.py tests/test_solver_api.py
.venv/bin/python -m pytest -q tests/test_backend_selection.py tests/test_model_backend_free.py
```

Results:

```text
compileall: passed
ruff: passed
tests/test_public_api.py tests/test_operator_public_api.py: 18 passed
tests/test_engine_source_registry.py tests/test_packed_layout_api.py tests/test_solver_api.py: 21 passed
tests/test_backend_selection.py tests/test_model_backend_free.py: 11 passed
```

Decision: Phase 1 accepted; continue to Phase 2.

## Phase 2 Source Route And Numba ABI Split

Phase 2 separated backend-neutral source semantics from the concrete Numba fused
ABI while preserving compatibility imports.

Files changed:

```text
veqpy/engine/backend_abi.py
veqpy/engine/numba_abi.py
veqpy/operator/build_plan.py
veqpy/operator/source_execution.py
veqpy/operator/source_plan.py
veqpy/operator/source_routes.py
tests/test_source_plan_backend_neutral.py
```

Implemented behavior:

- `veqpy/operator/source_routes.py` owns route key normalization and
  backend-neutral route metadata.
- `veqpy/operator/source_execution.py` owns active psin/F source ownership and
  workspace-needs metadata.
- `veqpy/engine/numba_abi.py` owns concrete Numba ABI bundles and the
  `NumbaSourceBindingPlan` adapter.
- `veqpy/engine/backend_abi.py` remains as a compatibility shim re-exporting
  the Numba ABI names.
- `SourcePlan` no longer stores a concrete Numba kernel callable in its dataclass
  fields. Compatibility properties retrieve Numba callables and integer codes
  through `NumbaSourceBindingPlan`.
- Existing `veqpy.engine.validate_route` compatibility remains unchanged.

Validation:

```bash
.venv/bin/python -m compileall -q veqpy tests
.venv/bin/ruff check veqpy tests
.venv/bin/python -m pytest -q tests/test_engine_source_registry.py
.venv/bin/python -m pytest -q tests/test_source_plan_backend_neutral.py
.venv/bin/python -m pytest -q tests/test_workspace_field_contracts.py
.venv/bin/python -m pytest -q tests/test_packed_layout_api.py
.venv/bin/python -m pytest -q tests/test_solver_api.py
```

Results:

```text
compileall: passed
ruff: passed
tests/test_engine_source_registry.py: 2 passed
tests/test_source_plan_backend_neutral.py: 5 passed
tests/test_workspace_field_contracts.py: 9 passed
tests/test_packed_layout_api.py: 6 passed
tests/test_solver_api.py: 13 passed
```

Decision: Phase 2 accepted; continue to Phase 3.

## Phase 3 Static/Dynamic Manifest Gate

Phase 3 created the JAX lowering manifest before numerical JAX kernels.

Files changed:

```text
docs/jax_static_dynamic_manifest.md
tests/test_jax_static_manifest.py
```

Implemented behavior:

- Classified `OperatorBuildPlan` fields into static metadata, dynamic device
  leaves, host-only bridge state, and unsupported/not-yet-lowered state.
- Classified workspace-derived lowering arrays.
- Defined route/capability cache-key boundaries.
- Defined dynamic per-call `x` and residual output behavior.
- Defined backend option cache-key behavior.
- Added a public Operator method behavior matrix for the initial JAX backend.
- Added a manifest gate test proving large arrays are not marked static.

Validation:

```bash
.venv/bin/python -m compileall -q veqpy tests
.venv/bin/ruff check veqpy tests
.venv/bin/python -m pytest -q tests/test_jax_static_manifest.py
```

Results:

```text
compileall: passed
ruff: passed
tests/test_jax_static_manifest.py: 4 passed
```

Decision: Phase 3 accepted; continue to Phase 4.

## Phase 4 Optional JAX Runtime Shell

Phase 4 added the optional JAX runtime boundary without implementing residual
parity.

Files changed:

```text
pyproject.toml
veqpy/engine/jax/__init__.py
veqpy/engine/jax/config.py
veqpy/engine/jax/memory.py
veqpy/engine/jax/state.py
veqpy/layout/backend_binding.py
veqpy/layout/binding.py
veqpy/layout/jax_binding.py
veqpy/layout/numba_binding.py
veqpy/operator/operator.py
tests/jax_helpers.py
tests/test_jax_device_state.py
tests/test_jax_memory_options.py
tests/test_jax_optional_dependency.py
tests/test_jax_public_method_contract.py
tests/test_jax_unsupported_routes.py
```

Implemented behavior:

- JAX remains optional via the `jax` extra and is not a required dependency.
- JAX is imported only through the JAX backend path.
- Unsupported JAX routes fail before optional JAX import.
- Supported route shell is `PF/rho/grid`.
- With JAX missing, supported route construction raises
  `MissingOptionalBackendError`.
- At Phase 4, the JAX layout shell was constructible without enabling a partial
  numerical residual path; Phase 5-8 later enabled PF/rho/grid residual parity.
- `alpha1`/`alpha2` do not read stale Numba workspace state under backend `jax`.
- `binding.py` remains as a compatibility wrapper while backend dispatch lives
  in `backend_binding.py`.

Validation:

```bash
.venv/bin/python -m compileall -q veqpy tests
.venv/bin/ruff check veqpy tests
.venv/bin/python -m pytest -q tests/test_backend_selection.py tests/test_model_backend_free.py
.venv/bin/python -m pytest -q \
  tests/test_jax_optional_dependency.py tests/test_jax_memory_options.py \
  tests/test_jax_device_state.py tests/test_jax_unsupported_routes.py \
  tests/test_jax_public_method_contract.py
.venv/bin/python -m pytest -q tests/test_public_api.py tests/test_operator_public_api.py tests/test_solver_api.py
```

Results:

```text
compileall: passed
ruff: passed
tests/test_backend_selection.py tests/test_model_backend_free.py: 11 passed
JAX shell tests: 15 passed
tests/test_public_api.py tests/test_operator_public_api.py tests/test_solver_api.py: 31 passed
```

Decision: Phase 4 accepted; continue to Phase 5.

## Phase 5-8 JAX Numerical Path Status

Files changed:

```text
veqpy/engine/jax/profile.py
veqpy/engine/jax/geometry.py
veqpy/engine/jax/source.py
veqpy/engine/jax/residual.py
veqpy/engine/jax/compile.py
veqpy/engine/jax/operator.py
tests/test_jax_stage_parity.py
tests/test_jax_residual_parity.py
tests/test_jax_solver_smoke.py
```

Implemented behavior:

- Implemented PF/rho/grid JAX profile, geometry, source, fused residual, compile,
  and host bridge path.
- `Operator(..., backend="jax").residual_var(...)` and
  `residual_var_into(...)` return/copy NumPy arrays for PF/rho/grid.
- JAX residual execution uses the existing PackedLayout metadata and residual
  block order; no alternate packed order is introduced.
- JAX residual execution is a residual-only hot path: it copies only the packed
  residual back to host for SciPy and does not publish profile, geometry,
  source, root, alpha, or snapshot state.
- `build_equilibrium(x)` and `publish_snapshot(x)` use an explicit lazy snapshot
  graph that publishes the full NumPy-backed host state exactly for the requested
  packed state.
- Did not implement a JAX-native nonlinear solver.
- Kept supported JAX route list at `PF/rho/grid` only.
- Left all unsupported routes explicit through `UnsupportedBackendFeature`.
- Added stage parity, residual parity, and SciPy solver smoke tests.

Validation:

```bash
.venv/bin/python -m compileall -q veqpy tests
.venv/bin/ruff check veqpy tests
.venv/bin/python -m pytest -q \
  tests/test_jax_residual_parity.py tests/test_jax_stage_parity.py \
  tests/test_public_api.py tests/test_operator_public_api.py \
  tests/test_backend_selection.py tests/test_model_backend_free.py \
  tests/test_jax_optional_dependency.py tests/test_jax_unsupported_routes.py \
  tests/test_jax_public_method_contract.py
.venv/bin/python -m pytest -q \
  tests/test_jax_unsupported_routes.py tests/test_jax_public_method_contract.py \
  tests/test_jax_stage_parity.py tests/test_jax_residual_parity.py \
  tests/test_jax_solver_smoke.py
```

Results:

```text
JAX env: jax 0.10.1, [CpuDevice(id=0)], x64=True, backend=cpu
compileall: passed
ruff: passed
tests/test_jax_stage_parity.py: 2 passed
tests/test_jax_residual_parity.py: 5 passed
tests/test_jax_solver_smoke.py: 2 passed
```

Numerical parity tolerances:

```text
stage fields: rtol=1e-10, atol=1e-10
packed residual: rtol=1e-8, atol=1e-8
```

## Residual-Only Hot Path And Explicit Snapshot Publication

The JAX runtime is split into two independently compiled paths:

```text
Path A: SciPy hot path
host x -> device_put -> compiled_residual -> device residual -> host NumPy residual

Path B: explicit snapshot publication
final x -> device_put -> compiled_snapshot -> host NumPy snapshot -> Equilibrium
```

Important boundary:

- the existing SciPy solver still requires one host NumPy residual per function
  evaluation, so residual calls still synchronize/copy the residual vector;
- residual calls do not publish profile, geometry, source, root, alpha, or
  equilibrium snapshot state;
- `build_equilibrium(x)` publishes a snapshot for exactly `x` unless an exact
  cached snapshot for the same runtime generation and static signature exists;
- `Solver.solve()` publishes one final JAX snapshot for `scipy_result.x`, then
  later `Solver.build_equilibrium()` reuses that exact snapshot;
- `alpha1`/`alpha2` are readable only after a valid explicit snapshot and remain
  read-only for JAX;
- staged APIs and collocation remain unsupported for JAX because fused residual
  calls do not refresh staged host workspaces;
- a future JAX-native nonlinear solver is required to remove the per-evaluation
  residual host/device transfer entirely.

## Public API Changes

- `Operator` accepts `backend="numba"` and `backend="jax"`.
- `Operator.backend` records the normalized backend name.
- `Operator.backend_options` stores optional backend options.
- Public model objects remain backend-free.
- No public JAX arrays/tracers are exposed.

## Backend Behavior Table

| Backend | Behavior |
|---|---|
| default | Same as `numba`. |
| `numba` | Existing Numba layout/runtime path. |
| `jax`, unsupported route | Raises `UnsupportedBackendFeature` before importing JAX. |
| `jax`, `PF/rho/grid`, JAX missing | Raises `MissingOptionalBackendError`. |
| `jax`, `PF/rho/grid`, JAX installed | Runs JAX-backed fused residual through NumPy-facing Operator API. |

## Supported JAX Routes

| Route | Status |
|---|---|
| `PF/rho/grid` | Stage parity, fused residual parity, and SciPy solver smoke pass on CPU JAX. |

## Unsupported JAX Routes

All routes except `PF/rho/grid` are unsupported and fail before JIT/import:

```text
PF/rho/uniform
PF/psin/grid
PF/psin/uniform
PP/*
PI/*
PJ1/*
PJ2/*
PQ/*
```

## Public Method Behavior Matrix

| Method | JAX behavior in this build |
|---|---|
| `residual_var` | Implemented for PF/rho/grid; returns residual-only `np.ndarray`; no snapshot publication. |
| `residual_var_into` | Implemented for PF/rho/grid; writes caller-owned residual-only `np.ndarray`; no snapshot publication. |
| `residual_collocation` | `UnsupportedBackendFeature`. |
| `residual_collocation_into` | `UnsupportedBackendFeature`. |
| `stage_a_profile` | `UnsupportedBackendFeature`. |
| `stage_b_geometry` | `UnsupportedBackendFeature`. |
| `stage_c_source` | `UnsupportedBackendFeature`. |
| `stage_d_residual` | `UnsupportedBackendFeature`. |
| `build_equilibrium` | Publishes or reuses an explicit snapshot for exactly `x`; returns NumPy-backed `Equilibrium`. |
| `replace_problem` | Revalidates JAX route capability and raises on unsupported route. |
| `replace_case` | Alias of `replace_problem`; same behavior. |
| `alpha1` | Read-only value from a valid published snapshot; `SnapshotNotPublishedError` before publish. |
| `alpha2` | Read-only value from a valid published snapshot; `SnapshotNotPublishedError` before publish. |

## Static/Dynamic Manifest Summary

`docs/jax_static_dynamic_manifest.md` exists and classifies:

- `OperatorBuildPlan` fields;
- workspace-derived lowering arrays;
- route/capability keys;
- dynamic `x` and residual output;
- backend options and cache-key semantics;
- host publication state;
- public Operator method behavior.

The manifest gate verifies required sections and checks that large arrays are
not marked as static metadata.

## Final Validation

```bash
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu .venv-jax/bin/python -m compileall -q veqpy tests scripts
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu .venv-jax/bin/ruff check veqpy tests scripts
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu .venv-jax/bin/python -m pytest -q tests/test_public_api.py tests/test_operator_public_api.py
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu .venv-jax/bin/python -m pytest -q tests/test_engine_source_registry.py tests/test_workspace_field_contracts.py
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu .venv-jax/bin/python -m pytest -q tests/test_packed_layout_api.py tests/test_solver_api.py
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu .venv-jax/bin/python -m pytest -q \
  tests/test_backend_selection.py tests/test_model_backend_free.py tests/test_source_plan_backend_neutral.py
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu .venv-jax/bin/python -m pytest -q \
  tests/test_jax_static_manifest.py tests/test_jax_optional_dependency.py \
  tests/test_jax_memory_options.py tests/test_jax_device_state.py
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu .venv-jax/bin/python -m pytest -q \
  tests/test_jax_unsupported_routes.py tests/test_jax_public_method_contract.py \
  tests/test_jax_stage_parity.py tests/test_jax_residual_parity.py \
  tests/test_jax_snapshot_publication.py tests/test_jax_solver_smoke.py
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu .venv-jax/bin/python -m pytest -q -m "not slow"
```

Results:

```text
compileall: passed
ruff: passed
public/operator API: 18 passed
engine source/workspace contracts: 11 passed
packed layout/solver API: 19 passed
backend/model/source-plan/static tests: 20 passed
JAX optional/memory/state/unsupported/public tests: 13 passed, 2 skipped
JAX stage/residual/snapshot/solver parity tests: 18 passed
broad non-slow suite: 125 passed, 2 skipped, 2 deselected
```

## Git Diff Summary

Tracked diff:

```text
pyproject.toml
tests/test_solver_api.py
veqpy/engine/backend_abi.py
veqpy/layout/binding.py
veqpy/operator/build_plan.py
veqpy/operator/operator.py
veqpy/operator/source_plan.py
```

Additional untracked files are the new docs, tests, backend modules, layout
modules, source-route/source-execution modules, and private JAX shell modules.
Run `git status --short` for the full list before committing.

## Known Limitations

- JAX numerical execution is supported only for `PF/rho/grid`.
- JAX public staged methods and collocation methods intentionally raise
  `UnsupportedBackendFeature`.
- JAX-native nonlinear solve is not implemented; SciPy still calls the existing
  `Operator.residual_var` / `residual_var_into` API.
- GPU optimization, psin routes, active-F routes, PJ2, and PQ are not
  implemented.
- `backend_abi.py` remains as a compatibility shim; downstream code should move
  toward `numba_abi.py` for concrete Numba bindings.

## Next Recommendations

1. Add the next route only after adding parity tests; suggested order:
   `PF/rho/uniform`, then `PP/rho/grid`.
2. Add explicit optional slow benchmarks for JAX residual wall time versus Numba.
3. Keep JAX as an optional dependency and avoid GPU-specific policy until CPU
   parity coverage is broader.
4. Decide whether staged public JAX methods should gain explicit host
   publication or remain unsupported.
