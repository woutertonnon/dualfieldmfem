# Phase 0 Inventory: Config and Output Contract

This document captures the current JSON/config and output contract used by the
solver executables. It is the baseline for the split described in
`docs/split-testbench-plan.md`.

## Scope

- C++ executables in `apps/*.cpp`
- Config loader and typed config classes in `include/io.h`
- Existing generated configs under `data/config/*/*.json`

## Executable -> Config Class Mapping

- `apps/dualfieldnavierstokes_nitsche.cpp` -> `DualFieldConfig`
- `apps/hcurl_dualfieldnavierstokes_nitsche.cpp` -> `DualFieldConfig`
- `apps/hdiv_dualfieldnavierstokes_nitsche.cpp` -> `DualFieldConfig`
- `apps/hcurl_singlefieldnavierstokes_nitsche.cpp` -> `DualFieldConfig`
- `apps/hdiv_singlefieldnavierstokes_nitsche.cpp` -> `DualFieldConfig`
- `apps/Stokes.cpp` -> `NitscheStokesConfig`

## Typed Config Keys in C++

### DualFieldConfig

Defined in `include/io.h:133`.

Required string keys (no default):
- `mesh` (`include/io.h:139`)
- `outputfile` (`include/io.h:140`)
- `solver` (`include/io.h:141`)

Scalar keys with defaults:
- `dt` default `0.02` (`include/io.h:142`)
- `T` default `1.` (`include/io.h:143`)
- `refinements` default `10` (`include/io.h:144`)
- `order` default `1` (`include/io.h:145`)
- `visualisation` default `0` (`include/io.h:146`)
- `tol` default `1e-8` (`include/io.h:147`)
- `viscosity` default `0.` (`include/io.h:148`)
- `printlevel` default `0` (`include/io.h:149`)

Optional exact-solution presence flag:
- `exact_data_u` used for `has_exact_u_solution` (`include/io.h:150`)

Runtime-loaded function bodies expected by the generated shared library:
- `force_data`
- `initial_data_u`
- `initial_data_w`
- `exact_data_u`
- `exact_data_w`
- `boundary_data_u`
  (`include/io.h:152`)

Optional boundary attributes:
- `lid_attributes` detected via `has_lid_attributes()` (`include/io.h:170`)

### NitscheStokesConfig

Defined in `include/io.h:207`.

Required string keys (no default):
- `mesh` (`include/io.h:213`)
- `outputfile` (`include/io.h:214`)
- `solver` (`include/io.h:215`)

Scalar keys with defaults:
- `refinements` default `10` (`include/io.h:216`)
- `order` default `1` (`include/io.h:217`)
- `visualisation` default `0` (`include/io.h:218`)
- `tol` default `1e-8` (`include/io.h:219`)
- `mass` default `0.` (`include/io.h:220`)
- `viscosity` default `0.` (`include/io.h:221`)
- `printlevel` default `0` (`include/io.h:222`)

Optional exact-solution presence flag:
- `exact_data_u` used for `has_exact_u_solution` (`include/io.h:223`)

Runtime-loaded function bodies expected by generated shared library:
- `force_data`
- `exact_data_u`
  (`include/io.h:225`)

## Per-Executable Key Usage

### Dual-field Navier--Stokes executables

Used in:
- `apps/dualfieldnavierstokes_nitsche.cpp`
- `apps/hcurl_dualfieldnavierstokes_nitsche.cpp`
- `apps/hdiv_dualfieldnavierstokes_nitsche.cpp`

Read scalar/config keys:
- `viscosity`, `refinements`, `order`, `visualisation`, `tol`, `dt`, `T`,
  `mesh`, `outputfile`

Read function keys:
- `force_data`, `boundary_data_u`, `initial_data_u`

Optional:
- `lid_attributes` through `has_lid_attributes()` and `get_lid_marker(...)`

### Single-field Navier--Stokes executables

Used in:
- `apps/hcurl_singlefieldnavierstokes_nitsche.cpp`
- `apps/hdiv_singlefieldnavierstokes_nitsche.cpp`

Read scalar/config keys:
- `viscosity`, `refinements`, `order`, `visualisation`, `printlevel`, `tol`,
  `mesh`, `outputfile`, `solver`, `dt`, `T`

Read function keys:
- `force_data`, `boundary_data_u`, `initial_data_u`

Optional:
- `lid_attributes` through `has_lid_attributes()` and `get_lid_marker(...)`

### Stokes executable

Used in:
- `apps/Stokes.cpp`

Read scalar/config keys:
- `mass`, `viscosity`, `refinements`, `order`, `visualisation`, `printlevel`,
  `tol`, `mesh`, `outputfile`, `solver`

Read function keys:
- `force_data`, `exact_data_u`

## Existing JSON Dataset Snapshot

Observed generated config directories and file counts (from `data/config`):

- `3DTaylorGreen` (7)
- `ConstantField` (1)
- `LidDrivenCavity3D` (1)
- `LidDrivenCavity3DExact` (1)
- `LidDrivenCavity3DExactCw1000` (1)
- `LidDrivenCavity3DExactDualHcurlCw1000` (1)
- `LidDrivenCavity3DExactParallel` (1)
- `LidDrivenCavity3Dnoconvection` (1)
- `LidDrivenCavity3Dnormaljumppenalty` (1)
- `NavierStokesTest` (9)
- `RigidRotation` (1)
- `RigidRotationDualField` (1)
- `SingleFieldVorticityPrevTimestep` (7)

Note: this snapshot contains Navier--Stokes-style configs only. A generated
Stokes-style config set with `mass` was not present in `data/config` at scan
time, even though `NitscheStokesConfig` requires/uses `mass`.

## Union of Keys Found in Existing JSON Files

- `T`
- `boundary_data_u`
- `dt`
- `exact_data_u`
- `exact_data_w`
- `force_data`
- `initial_data_u`
- `initial_data_w`
- `lid_attributes`
- `mesh`
- `order`
- `outputfile`
- `printlevel`
- `refinements`
- `solver`
- `tol`
- `viscosity`
- `visualisation`

## Output Contract (Current)

- CSV outputs are written to `./out/data/<outputfile>_vars.csv`
  (`include/io.h:264`).
- `outputfile` itself is typically `<benchmark>/<case_name>` from Python config
  generation.
- Plotting/data-collection scripts expect files under
  `out/data/<benchmark>/<benchmark>_conv_order*_ref*_vars.csv`.

## Phase 0 Conclusions

- The split can treat `DualFieldConfig` and `NitscheStokesConfig` as two
  adapter-level schemas sharing a small core of keys (`mesh`, `outputfile`,
  `solver`, `order`, `refinements`, logging/visualisation controls).
- Function-body keys (`force_data`, `initial_data_u`, etc.) are a hard runtime
  contract due to the generated shared library in `SimulationConfig`.
- Backward compatibility should preserve the legacy flat JSON shape until C++
  executables migrate to a new canonical schema adapter layer.
