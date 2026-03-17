# Split Testbench Schema v1

This document defines a canonical, solver-agnostic case schema (`v1`) and the
explicit translation rules to the current legacy flat JSON format consumed by
the existing C++ executables.

It is intended to be used together with:
- `docs/split-testbench-plan.md`
- `docs/split-testbench-phase0-inventory.md`

## Design Goals

- Keep the canonical schema solver-agnostic and explicit.
- Preserve backward compatibility with current executables.
- Make adapter translation deterministic and testable.
- Support local and cluster execution metadata without polluting solver payload.

## Canonical Schema (v1)

Top-level required fields:

- `schema_version` (int): must be `1`.
- `case_id` (string): unique case identifier.
- `solver` (object): solver identity and payload.
- `discretization` (object): order/refinement settings.
- `time` (object): time controls.
- `runtime` (object): generic runtime controls.
- `output` (object): output naming/paths.

Top-level optional fields:

- `runner` (object): execution backend metadata (local/ssh/slurm).
- `metadata` (object): tags/provenance notes.

## Canonical Field Specification

### `schema_version`

- Type: `int`
- Allowed: `1`

### `case_id`

- Type: `string`
- Example: `LidDrivenCavity3D_conv_order1_ref2`

### `solver`

- Type: `object`
- Required:
  - `family` (string): e.g. `mfem_navier_stokes`, `mfem_stokes`
  - `variant` (string): e.g. `hcurl_dualfield_nitsche`, `hdiv_singlefield_nitsche`
  - `linear_solver` (string): e.g. `GMRES`
  - `parameters` (object): scalar solver parameters
  - `functions` (object): runtime function bodies as C snippets
- Optional:
  - `boundary` (object): boundary-condition metadata

Expected keys inside `solver.parameters` for current adapters:
- Navier--Stokes adapter:
  - `viscosity` (float)
- Stokes adapter:
  - `mass` (float)
  - `viscosity` (float)

Expected keys inside `solver.functions` for current adapters:
- Navier--Stokes style:
  - `force_data`
  - `initial_data_u`
  - `initial_data_w`
  - `boundary_data_u`
  - optional exact fields: `exact_data_u`, `exact_data_w`
- Stokes style:
  - `force_data`
  - optional exact field: `exact_data_u`

Expected keys inside `solver.boundary` for current adapters:
- optional `lid_attributes` (array[int])

### `discretization`

- Type: `object`
- Required:
  - `order` (int)
  - `refinements` (int)

### `time`

- Type: `object`
- Required:
  - `dt` (float)
  - `T` (float)

### `runtime`

- Type: `object`
- Required:
  - `tol` (float)
  - `visualisation` (int)
  - `printlevel` (int)

### `output`

- Type: `object`
- Required:
  - `mesh` (string): path to mesh file
  - `outputfile` (string): legacy-compatible output stem

### `runner` (optional)

- Type: `object`
- Optional keys (suggested):
  - `backend` (`local` | `ssh_slurm`)
  - `hostname`
  - `username`
  - `remote_root`
  - `slurm` object (`time`, `mem_per_cpu`, `cpus_per_task`)

### `metadata` (optional)

- Type: `object`
- Free-form tags, git commit hash, notes, etc.

## Canonical Example: Navier--Stokes Case

```json
{
  "schema_version": 1,
  "case_id": "ConstantField_conv_order1_ref0",
  "solver": {
    "family": "mfem_navier_stokes",
    "variant": "hcurl_singlefield_nitsche",
    "linear_solver": "GMRES",
    "parameters": {
      "viscosity": 0.001
    },
    "functions": {
      "force_data": "out[0]=0;out[1]=0;out[2]=0;",
      "initial_data_u": "out[0]=0;out[1]=0;out[2]=0;",
      "initial_data_w": "out[0]=0;out[1]=0;out[2]=0;",
      "boundary_data_u": "out[0]=1;out[1]=0;out[2]=0;",
      "exact_data_u": "out[0]=1;out[1]=0;out[2]=0;",
      "exact_data_w": "out[0]=0;out[1]=0;out[2]=0;"
    },
    "boundary": {
      "lid_attributes": [16]
    }
  },
  "discretization": {
    "order": 1,
    "refinements": 0
  },
  "time": {
    "dt": 0.01,
    "T": 100.0
  },
  "runtime": {
    "tol": 1e-7,
    "visualisation": 1,
    "printlevel": 2
  },
  "output": {
    "mesh": "./geo/mesh/ConstantField.msh",
    "outputfile": "ConstantField/ConstantField_conv_order1_ref0"
  }
}
```

## Canonical Example: Stokes Case

```json
{
  "schema_version": 1,
  "case_id": "StokesTest_conv_order2_ref3",
  "solver": {
    "family": "mfem_stokes",
    "variant": "hcurl_nitsche",
    "linear_solver": "GMRES",
    "parameters": {
      "mass": 1.0,
      "viscosity": 0.02
    },
    "functions": {
      "force_data": "out[0]=0;out[1]=0;out[2]=0;",
      "exact_data_u": "out[0]=...;out[1]=...;out[2]=...;"
    }
  },
  "discretization": {
    "order": 2,
    "refinements": 3
  },
  "time": {
    "dt": 0.0,
    "T": 0.0
  },
  "runtime": {
    "tol": 1e-5,
    "visualisation": 0,
    "printlevel": 1
  },
  "output": {
    "mesh": "./extern/mfem/data/ref-cube.mesh",
    "outputfile": "StokesTest/StokesTest_conv_order2_ref3"
  }
}
```

Note: `time` is kept in canonical schema even for steady problems for shape
uniformity. Adapter can ignore it when translating to a solver that does not
use `dt`/`T`.

## Legacy Translation Map (Canonical -> Current Flat JSON)

The adapter must translate canonical schema to current executable contract.

### Common mappings (all current executables)

- `output.mesh` -> `mesh`
- `solver.linear_solver` -> `solver`
- `runtime.visualisation` -> `visualisation`
- `runtime.printlevel` -> `printlevel`
- `runtime.tol` -> `tol`
- `discretization.order` -> `order`
- `discretization.refinements` -> `refinements`
- `output.outputfile` -> `outputfile`

### Navier--Stokes mappings (`DualFieldConfig` family)

- `time.dt` -> `dt`
- `time.T` -> `T`
- `solver.parameters.viscosity` -> `viscosity`
- `solver.functions.force_data` -> `force_data`
- `solver.functions.initial_data_u` -> `initial_data_u`
- `solver.functions.initial_data_w` -> `initial_data_w`
- `solver.functions.boundary_data_u` -> `boundary_data_u`
- optional `solver.functions.exact_data_u` -> `exact_data_u`
- optional `solver.functions.exact_data_w` -> `exact_data_w`
- optional `solver.boundary.lid_attributes` -> `lid_attributes`

### Stokes mappings (`NitscheStokesConfig` family)

- `solver.parameters.mass` -> `mass`
- `solver.parameters.viscosity` -> `viscosity`
- `solver.functions.force_data` -> `force_data`
- optional `solver.functions.exact_data_u` -> `exact_data_u`

No `dt`/`T` fields are emitted for legacy Stokes config unless executable
contract is extended.

## Legacy Defaults and Validation Rules

Defaults should be applied in adapter (or validator) before translation, not
implicitly relied upon in C++.

Recommended defaults (matching current C++ behavior):
- `runtime.visualisation = 0`
- `runtime.printlevel = 0`
- `runtime.tol = 1e-8`
- `discretization.order = 1`
- `discretization.refinements = 10`
- Navier--Stokes: `time.dt = 0.02`, `time.T = 1.0`
- `solver.parameters.viscosity = 0.0` (if not provided)
- Stokes: `solver.parameters.mass = 0.0` (if not provided)

Required validation checks:
- `schema_version == 1`
- required sections exist
- required function keys exist for selected adapter family
- numeric fields are finite and non-negative where applicable
- `lid_attributes` values are positive integers

## Suggested Implementation API

```python
def validate_case_v1(case: dict) -> None: ...

def to_legacy_flat(case: dict) -> dict: ...

def write_legacy_config(case: dict, path: Path) -> Path: ...
```

Adapter-specific wrappers:

```python
class MfemNavierStokesAdapter:
    def build_case(self, ...): ...
    def to_legacy(self, case: dict) -> dict: ...

class MfemStokesAdapter:
    def build_case(self, ...): ...
    def to_legacy(self, case: dict) -> dict: ...
```

## Contract Tests (Required)

For each benchmark family, add tests that:

1. Build canonical case (`v1`).
2. Translate to legacy flat JSON.
3. Compare translated JSON with golden legacy JSON (normalized keys/order).

Minimum golden sets:
- one dual-field Navier--Stokes case,
- one single-field Navier--Stokes case,
- one Stokes case with `mass`.
