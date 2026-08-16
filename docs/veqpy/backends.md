# Backends

The supported backend tokens are `numba`, `cxx`, `cxx-strict`, `cxx-relaxed`,
and `cxx-enzyme`. `numba` is the default and always uses strict floating-point
math. `cxx` is the short spelling for the Release relaxed artifact. The other
three Cxx tokens select independent Release artifacts.

All five paths consume the same runtime explicit source contract. A source
grid can contain any strictly increasing normalized nodes accepted by the
capacity policy, including two-point and nonuniform inputs. `rho` is a normal
coordinate choice, not a Cxx exclusion. Cxx keeps its native workspace bound
at 1024 while the Python buffer retains the active runtime count and capacity
epoch.

Build-only options are `artifact_dir`, `cpu_affinity`, and `rebuild`; they are
not solver or per-call options. The build policy is fixed before preparation.

```bash
.venv/bin/python -m veqpy --demo numba
.venv/bin/python -m veqpy --demo cxx-strict
.venv/bin/python -m veqpy --demo cxx-relaxed
.venv/bin/python -m veqpy --demo cxx-enzyme
```

The native toolchain is optional for importing VEQPy. A missing compiler or
native dependency is reported when a Cxx artifact is prepared.

## Cached native loading

The first successful Cxx preparation records an immutable artifact using the
shared `fusionprime_base.native` pointer and loader. Subsequent cached
preparation resolves that pointer and directly imports the extension. It does
not run compiler/CMake probes, calculate source or binary hashes, write a
last-used timestamp, or repeat preparation when the first solve starts.

A present but incompatible pointer, missing library, or failed dynamic load is
an explicit error; it never falls through to an implicit rebuild. Use the
build-only `rebuild=True` option after changing native sources in an editable
checkout. Package version, Python ABI, platform, source root, topology, backend
recipe, and native runtime schema participate in pointer selection.

## Enzyme backend

`cxx-enzyme` is the fifth public backend. It is a separate Release relaxed
artifact: the primal semantics match `cxx-relaxed`, while Enzyme supplies both
the nonlinear solver derivatives and the public raw Kernel residual
JVP/Jacobian. A new build fails when Enzyme is absent or incompatible and never
falls back to finite differences or another backend. A cached immutable
artifact remains loadable without the compiler plugin.

The macOS arm64 reference toolchain is Homebrew LLVM/Clang `22.1.8` plus
Enzyme `0.0.290`:

```bash
brew install enzyme
$(brew --prefix llvm@22)/bin/clang++ --version
```

The compiler and `ClangEnzyme-22.dylib` must have the same LLVM major. These
are system build dependencies and are not managed by Python packaging. All
VEQPy Cxx environments use LLVM major 22; Ubuntu 24.04 CI installs `clang-22`
and `lld-22` from apt.llvm.org. Enzyme builds on every platform require a
ClangEnzyme plugin and headers built for that same LLVM major; no platform gets
a fallback derivative provider or an alternate compiler contract.

The complete artifact and representative `psin`, `r`, `rho`, PJ2, and PJ3
source closures pass on the macOS reference toolchain. Dense derivatives use
scalar forward columns: Enzyme `0.0.290` still crashes LLVM 22 type analysis
when the full residual is requested through the vector-width intrinsic. The
supported artifact identity records the compiler and plugin versions, plugin
and header hashes, paths, and build flags. Enzyme provides residual derivatives
with respect to packed Kernel unknowns; a complete Module solve-map derivative
still requires implicit input linearization.
