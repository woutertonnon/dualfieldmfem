# Split Testbench Plan

## Objective

Extract the testbench/config generation/parsing pipeline from the current
Navier--Stokes-focused codebase so it can support multiple solvers, while
preserving existing behavior for current MFEM Navier--Stokes and Stokes runs.

## Current State (Observed Coupling)

- `scripts/SimIO.py` currently combines several concerns in one module:
  - symbolic PDE/manufactured-solution definitions,
  - config generation,
  - local execution,
  - Euler SSH/SLURM orchestration,
  - output collection and plotting.
- `scripts/benchmarks.py` is tightly coupled to Navier--Stokes assumptions
  (fields, executable defaults, output columns).
- `include/io.h` combines:
  - generic JSON/runtime function-loader logic,
  - solver-specific typed configs (`DualFieldConfig`, `NitscheStokesConfig`),
  - solver-specific CSV loggers.
- JSON contracts are implicit and spread across Python and C++.

## Target Architecture

### Layer A: Core Testbench (Solver-Agnostic)

Responsibilities:
- sweep expansion and case matrix generation,
- canonical config schema + validation,
- runner backends (local + SSH/SLURM),
- generic result discovery and plotting hooks.

### Layer B: Solver Adapters (Solver-Specific)

Responsibilities:
- map canonical schema into solver-native config,
- define executable command invocation,
- parse solver outputs into canonical metrics,
- provide optional benchmark recipe helpers,
- provide optional symbolic/manufactured-data builders.

## Proposed Project Layout (Monorepo First)

Introduce modules in this repository first, then split repos later if desired.

```
simbench_core/
  schema/
  io/
  sweep.py
  runner/
    local.py
    ssh_slurm.py
  postprocess/
    base.py

simbench_adapters/
  mfem_ns/
    adapter.py
    config_builder.py
    parser.py
    benchmarks.py
    symbolics.py
```

Compatibility wrappers remain in `scripts/SimIO.py` during migration.

## Adapter Contract (Initial)

Each solver adapter should implement:

- `build_case_config(case: dict) -> dict`
  - Build solver-native config dictionary from a canonical case spec.
- `write_config(case: dict, out_path: Path) -> Path`
  - Write the actual config file for one case.
- `command_for(config_path: Path) -> list[str]`
  - Return executable command and arguments.
- `result_glob(case: dict) -> str`
  - Return output-file pattern for this case.
- `parse_results(files: list[Path]) -> object`
  - Parse results into canonical structure for plotting/reporting.
- `default_output_columns() -> list[str]`
  - Declare expected error/diagnostic columns.

## Config Strategy

### Canonical Schema v1

Use a versioned schema with explicit namespaces:

- `schema_version`
- `case_id`
- `runner`
- `sweep`
- `output`
- `solver`

### Backward Compatibility

For existing C++ binaries, adapter emits current legacy flat JSON shape,
including keys such as:

- `mesh`, `solver`, `visualisation`, `printlevel`,
- `viscosity`, `mass` (if applicable),
- `force_data`, `initial_data_u`, `initial_data_w`,
- `boundary_data_u`, `exact_data_u`, `exact_data_w`,
- `outputfile`, `dt`, `T`, `refinements`, `order`, `tol`,
- `lid_attributes` (if present).

This allows migration without changing solver executables immediately.

## C++ Refactor Strategy

Split `include/io.h` into clearer components:

1. Generic runtime config utilities:
   - JSON loading/access,
   - generated shared-library function loading (`dlopen`/`dlsym`).
2. Solver-specific typed config classes:
   - `DualFieldConfig`,
   - `NitscheStokesConfig`.
3. Solver-specific loggers:
   - `DualFieldCSVLogger`,
   - `HCurlDualFieldCSVLogger`,
   - `HDivDualFieldCSVLogger`,
   - `NitscheStokesCSVLogger`.

No behavior changes in this stage; this is only responsibility separation.

## Migration Phases

### Phase 0: Contract Inventory (No Refactor)

- Enumerate config keys consumed by each executable in `apps/*.cpp`.
- Capture sample generated configs from `data/config/*` as golden files.
- Document output naming and CSV column conventions currently used.

Exit criteria:
- key inventory doc completed,
- at least one golden config per benchmark family captured.

### Phase 1: Extract Python Core Utilities

- Move generic sweep/config writing logic out of `scripts/SimIO.py`.
- Move local and remote runner logic into `simbench_core.runner`.
- Keep wrappers in `scripts/SimIO.py` that delegate to new modules.

Exit criteria:
- existing benchmark scripts still run unchanged via wrappers.

### Phase 2: Implement MFEM-NS Adapter

- Create `simbench_adapters/mfem_ns`.
- Port Navier--Stokes/Stokes config builder logic into adapter.
- Ensure adapter can emit legacy JSON shape used by current binaries.

Exit criteria:
- generated JSON parity with previous pipeline for selected benchmarks.

### Phase 3: Migrate Benchmark Recipes

- Move benchmark orchestration from `scripts/benchmarks.py` to adapter package.
- Keep optional shim imports in `scripts/benchmarks.py` for compatibility.

Exit criteria:
- old entry points still work,
- new adapter-first entry points available.

### Phase 4: Postprocessing Abstraction

- Split data collection/plotting into core framework + adapter parsers.
- Preserve existing output directories under `out/plots/<name>/...`.

Exit criteria:
- convergence plots and conservation plots match current outputs.

### Phase 5: C++ Header Decomposition

- Extract generic config/runtime loader into separate header/source pair.
- Keep typed solver configs and CSV loggers solver-side.
- Update includes in `apps/*.cpp` with no semantic changes.

Exit criteria:
- all current executables build/run with unchanged behavior.

### Phase 6: Packaging and Optional Repository Split

- Add Python packaging metadata for core and adapter modules.
- Optionally move `simbench_core` to dedicated repository when stable.

Exit criteria:
- core can be installed/used independently,
- adapter package depends on core via explicit version.

### Phase 7: Deprecation Cleanup

- Mark old direct APIs deprecated.
- Remove wrappers after one transition cycle.

Exit criteria:
- all internal callers use new APIs,
- deprecated paths removed cleanly.

## Validation and Regression Gates

Required before each major phase completion:

- Config parity checks:
  - compare legacy and adapter-generated JSON (normalized compare).
- Local end-to-end benchmark smoke tests:
  - one Navier--Stokes case,
  - one lid-driven cavity case,
  - one Stokes manufactured case.
- Output compatibility checks:
  - expected CSV columns still present for plotting.
- Plot smoke tests:
  - expected files generated under `out/plots/...`.

## Risks and Mitigations

- Hidden config-key dependencies in executables.
  - Mitigation: explicit key inventory + contract tests before refactor.
- Coupling in remote execution flow (Euler assumptions, paths).
  - Mitigation: backend abstraction with explicit backend-specific config.
- Regressions from splitting monolithic Python module.
  - Mitigation: compatibility wrappers + phased extraction.
- Premature over-generalization.
  - Mitigation: ship one adapter first (`mfem_ns`) before adding others.

## Deliverables

- `docs/split-testbench-plan.md` (this file).
- architecture decision notes for adapter interface and schema.
- `simbench_core` module with tests.
- `simbench_adapters/mfem_ns` adapter module with migrated benchmarks.
- compatibility wrappers in old script paths.
- validation scripts for config parity and smoke runs.

## Suggested Execution Order

1. Complete Phase 0 contract inventory.
2. Implement core extraction (Phase 1) with wrappers.
3. Implement MFEM-NS adapter and parity tests (Phase 2).
4. Migrate benchmarks (Phase 3) and postprocess abstraction (Phase 4).
5. Refactor C++ config/logging headers (Phase 5).
6. Package and optionally split repositories (Phase 6).
7. Remove deprecated wrappers after transition window (Phase 7).
