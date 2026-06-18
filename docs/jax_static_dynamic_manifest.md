# JAX Static/Dynamic Manifest

## Scope

This manifest defines the Phase 4-8 lowering boundary for the optional JAX
backend. It is intentionally written before numerical JAX residual kernels.

Initial supported target:

```text
route: PF
coordinate: rho
nodes: grid
public bridge: host NumPy x -> device residual -> host NumPy residual
solver: existing SciPy solver only
```

JAX-native nonlinear solving is out of scope.

## Categories

| Category | Meaning |
|---|---|
| static metadata | Hashable metadata that may enter the JAX compile/cache key. |
| dynamic device leaf | Numeric array copied to device state or per-call input. |
| host-only bridge | Host-side object or NumPy publication boundary. |
| unsupported/not-yet-lowered | Explicitly unsupported in the initial JAX backend. |

Large numeric arrays must not enter the static spec. Per-call `x` and residual
output are dynamic input/output.

## OperatorBuildPlan Fields

| Field | Category | Notes |
|---|---|---|
| `grid_workspace.Nr` | static metadata | Shape/control-flow-defining dimension. |
| `grid_workspace.Nt` | static metadata | Shape/control-flow-defining dimension. |
| `grid_workspace.K_max` | static metadata | Basis shape and residual block limit. |
| `grid_workspace.L_max` | static metadata | Profile radial order limit. |
| `grid_workspace.M_max` | static metadata | Fourier family shape limit. |
| `prefix_profile_names` | static metadata | Packed layout identity. |
| `shape_profile_names` | static metadata | Packed layout identity. |
| `profile_names` | static metadata | Packed layout identity. |
| `profile_index` | static metadata | Packed layout identity. |
| `c_profile_names` | static metadata | Geometry family identity. |
| `s_profile_names` | static metadata | Geometry family identity. |
| `profile_L` | static metadata | Active length/order metadata. |
| `coeff_index` | static metadata | Packed coefficient row ids. |
| `order_offsets` | static metadata | Packed coefficient order boundaries. |
| `active_profile_mask` | static metadata | Active packed profile mask. |
| `active_profile_ids` | static metadata | Active packed profile ids. |
| `x_size` | static metadata | Dynamic vector length. |
| `source_route_spec` | static metadata | Backend-neutral route metadata only. |
| `source_plan.route_key` | static metadata | Route/control-flow key. |
| `source_plan.coordinate` | static metadata | Route/control-flow key. |
| `source_plan.nodes` | static metadata | Route/control-flow key. |
| `source_plan.parameterization` | static metadata | Source sampling transform identity. |
| `source_plan.scaled_heat` | dynamic device leaf | Numeric source input. |
| `source_plan.scaled_current` | dynamic device leaf | Numeric source input. |
| `source_plan.scaled_Ip` | dynamic device leaf | Numeric scalar leaf. |
| `source_plan.beta` | dynamic device leaf | Numeric scalar leaf. |
| `source_plan.interpolation_kind` | static metadata | Algorithm selection for non-grid routes. |
| `source_execution` | static metadata | Ownership booleans and active lengths. |
| `residual_binding_layout.active_profile_names` | static metadata | Residual block identity. |
| `residual_binding_layout.active_residual_block_codes` | static metadata | Residual block codes/order. |
| `residual_binding_layout.active_residual_block_orders` | static metadata | Residual block codes/order. |
| `residual_binding_layout.active_residual_block_radial_powers` | static metadata | Residual block radial powers. |
| `profile_static_kwargs_by_name` | static metadata | Profile construction metadata. |
| `profile_offset_specs` | static metadata | Boundary/profile offset policy. |
| `profile_offsets` | dynamic device leaf | Numeric profile offsets. |
| `profile_scales` | dynamic device leaf | Numeric profile scales. |
| `profile_powers` | static metadata | Integer profile powers. |
| `profile_envelope_powers` | static metadata | Integer profile envelope powers. |
| `profile_amplitude_powers` | dynamic device leaf | Numeric profile amplitude powers. |

## Workspace-Derived JAX Lowering Arrays

| Array | Category | Notes |
|---|---|---|
| `grid_workspace.radial_fields` | dynamic device leaf | Includes rho powers and radial tables. |
| `grid_workspace.poloidal_fields` | dynamic device leaf | Includes theta and trigonometric tables. |
| `grid_workspace.weights` | dynamic device leaf | Quadrature weights. |
| `grid_workspace.differentiator` | dynamic device leaf | Radial differentiator matrix. |
| `grid_workspace.accumulator` | dynamic device leaf | Radial accumulator matrix. |
| `profile_workspace.profile_fields` | dynamic device leaf | Runtime profile values and derivatives. |
| `profile_workspace.profile_rp_fields` | dynamic device leaf | Runtime profile/rho-power fields. |
| `profile_workspace.profile_env_fields` | dynamic device leaf | Runtime envelope fields. |
| `profile_workspace.active_offsets` | dynamic device leaf | Active profile offsets. |
| `profile_workspace.active_scales` | dynamic device leaf | Active profile scales. |
| `profile_workspace.active_amplitude_powers` | dynamic device leaf | Active amplitude powers. |
| `profile_workspace.active_coeff_index_rows` | static metadata | Packed coefficient row ids. |
| `profile_workspace.active_lengths` | static metadata | Active profile lengths. |
| `profile_workspace.c_family_fields` | dynamic device leaf | Geometry family field slab. |
| `profile_workspace.s_family_fields` | dynamic device leaf | Geometry family field slab. |
| `profile_workspace.c_family_base_fields` | dynamic device leaf | Geometry base family field slab. |
| `profile_workspace.s_family_base_fields` | dynamic device leaf | Geometry base family field slab. |
| `geometry_workspace.surface_fields` | dynamic device leaf | Surface geometry scratch/state. |
| `geometry_workspace.radial_fields` | dynamic device leaf | Radial geometry scratch/state. |
| `source_workspace.array_scratch` | dynamic device leaf | Source scratch rows. |
| `source_workspace.matrix_scratch` | dynamic device leaf | Source matrix scratch. |
| `source_workspace.heat_spline_coeff` | dynamic device leaf | Interpolation/remap table. |
| `source_workspace.current_spline_coeff` | dynamic device leaf | Interpolation/remap table. |
| `source_workspace.barycentric_weights` | dynamic device leaf | Interpolation/remap table. |
| `source_workspace.materialized_heat_input` | dynamic device leaf | Source-owned materialization. |
| `source_workspace.materialized_current_input` | dynamic device leaf | Source-owned materialization. |
| `source_workspace.target_root_fields` | dynamic device leaf | Target root fields for source-owned psin routes. |
| `source_workspace.alpha_state` | host-only bridge | Host publication unless explicitly lowered later. |
| `residual_workspace.root_fields` | dynamic device leaf | Root field state for residual publication. |
| `residual_workspace.surface_fields` | dynamic device leaf | Surface residual scratch/state. |
| `residual_workspace.pack_scratch` | dynamic device leaf | Residual packing scratch. |
| `residual_workspace.pack_scratch_rows` | dynamic device leaf | Residual packing scratch rows. |

## Route And Capability Keys

| Item | Category | Notes |
|---|---|---|
| route key `route/coordinate/nodes` | static metadata | First supported route is `PF/rho/grid`. |
| coordinate name | static metadata | Determines source coordinate branch. |
| node layout | static metadata | Determines interpolation branch. |
| source parameterization | static metadata | Determines source axis transform. |
| supported route capability flag | static metadata | Unsupported routes fail before JIT. |
| active psin ownership flag | static metadata | Determines source/root ownership. |
| active F ownership flag | static metadata | Determines PJ2-only ownership. |
| source query workspace need | static metadata | Controls source scratch lowering. |
| target root field need | static metadata | Controls root field publication. |

## Dynamic Inputs And Outputs

| Item | Category | Notes |
|---|---|---|
| packed state `x` | dynamic device leaf | Per-call dynamic input. |
| residual output | dynamic device leaf | Per-call dynamic output copied back to NumPy for SciPy. |
| snapshot output PyTree | dynamic device leaf | Produced only by explicit snapshot publication. |
| caller-provided `out` in `residual_var_into` | host-only bridge | Caller-owned NumPy array, copy completes before return. |

## Backend Options

| Option | Category | Notes |
|---|---|---|
| `backend` | static metadata | Selects backend dispatch. |
| `JaxBackendOptions.platform` | static metadata | May change device/runtime selection. |
| `JaxBackendOptions.enable_x64` | static metadata | Changes numeric dtype policy. |
| `JaxBackendOptions.preallocate` | host-only bridge | Pre-import/runtime configuration. |
| `JaxBackendOptions.mem_fraction` | host-only bridge | Pre-import/runtime configuration. |
| `JaxBackendOptions.allocator` | host-only bridge | Pre-import/runtime configuration. |
| `JaxBackendOptions.donate_x` | static metadata | May alter compiled call signature. |
| `JaxBackendOptions.profile_memory` | host-only bridge | Debug/profiling only; not numerical cache key. |

Debug/profiling options do not enter the numerical cache key unless they change
numeric behavior or compiled function signature.

## Host Publication And Unsupported State

| State Or Method Family | Category | Notes |
|---|---|---|
| fused variational residual | dynamic device leaf | Implemented first for `PF/rho/grid`; returns residual only. |
| residual-only publication | host-only bridge | Return NumPy residual only; no full-state publication. |
| explicit snapshot publication | host-only bridge | Separate lazy snapshot graph used by `publish_snapshot` and `build_equilibrium`. |
| root fields publication | host-only bridge | Published only by the explicit snapshot path. |
| alpha state publication | host-only bridge | Published only by the explicit snapshot path; read-only afterwards. |
| collocation fields | unsupported/not-yet-lowered | Unsupported until parity exists. |
| snapshot fields | host-only bridge | Host NumPy snapshot cache keyed by exact `x`, generation, and static signature. |
| staged profile/geometry/source/residual APIs | unsupported/not-yet-lowered | Unsupported until staged publication exists. |

## Public Operator Method Matrix

| Method | Initial JAX behavior |
|---|---|
| `residual_var` | Supported for PF/rho/grid as residual-only host bridge; otherwise `UnsupportedBackendFeature`. |
| `residual_var_into` | Supported for PF/rho/grid as residual-only host bridge; otherwise `UnsupportedBackendFeature`. |
| `residual_collocation` | `UnsupportedBackendFeature`. |
| `residual_collocation_into` | `UnsupportedBackendFeature`. |
| `stage_a_profile` | `UnsupportedBackendFeature`. |
| `stage_b_geometry` | `UnsupportedBackendFeature`. |
| `stage_c_source` | `UnsupportedBackendFeature`. |
| `stage_d_residual` | `UnsupportedBackendFeature`. |
| `build_equilibrium` | Publishes/reuses explicit snapshot for exactly `x`; never reads stale Numba workspace. |
| `replace_problem` | Revalidates backend capability/static signature. |
| `replace_case` | Alias of `replace_problem`; same backend revalidation. |
| `alpha1` | Read-only reflected value after valid explicit snapshot; otherwise `SnapshotNotPublishedError`. |
| `alpha2` | Read-only reflected value after valid explicit snapshot; otherwise `SnapshotNotPublishedError`. |

## Unsupported Or Not Yet Lowered

| Feature | Category | Notes |
|---|---|---|
| `PF/rho/uniform` | unsupported/not-yet-lowered | Phase 7 candidate after grid route parity. |
| `PP` routes | unsupported/not-yet-lowered | Add sequentially with parity tests. |
| `PI` routes | unsupported/not-yet-lowered | Add sequentially with parity tests. |
| `PJ1` routes | unsupported/not-yet-lowered | Add sequentially with parity tests. |
| `PJ2` routes | unsupported/not-yet-lowered | Requires active F parity. |
| `PQ` routes | unsupported/not-yet-lowered | Later route; no silent support. |
| psin-coordinate routes | unsupported/not-yet-lowered | Require interpolation/remap parity first. |
| JAX-native nonlinear solver | unsupported/not-yet-lowered | Explicit non-goal. |
