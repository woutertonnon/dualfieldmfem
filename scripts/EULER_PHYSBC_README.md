# Navier–Stokes with physical (no-slip) boundary conditions — Euler workflow

Convergence studies for the thesis chapter, run on ETH Euler. Four experiments,
each at orders 1 and 2 and a sweep of viscosities ε:

| Experiment (registry prefix) | Domain | ε set | Notes |
|---|---|---|---|
| `MMS2D_tangential`, `MMS2D_noslip` | unit square | 1e-2, 1e-3, 1e-4 | stream-function MMS; tangential vs genuine no-slip |
| `EthierSteinman3D` | (−1,1)³ | 1e-2, 1e-3, 1e-4 | exact 3D solution |
| `StokesSecond` | (0,3)×(0,1) | 1e-2, 5e-3, 2e-3 | oscillating-wall layer; resolvable ε |
| `Womersley` | (0,3)×(0,1) | 1e-2, 5e-3, 2e-3 | pulsatile channel, two wall layers |

Each (experiment, ε, order) is one SLURM job running the full h/τ sweep
(τ ∝ h). Per refinement level a CSV is written with the final-time
L²(Ω) velocity error (column `u1_err_L2`).

## 1. One-time setup on Euler

```bash
git pull                                   # get this branch onto Euler
# Python env for config/mesh generation + post-processing (NOT the solver):
python -m venv ~/physbc-venv
source ~/physbc-venv/bin/activate
pip install -r scripts/requirements_euler.txt
# Build the solver (or let the first job build it with BUILD_BINARY=1):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target \
  semilagrangian_navierstokes_nitsche_mpi \
  semilagrangian_navierstokes_nitsche_order2_mpi
```

The C++ compiler is a **runtime** dependency: `io.h` JIT-compiles the
force/boundary/initial-data snippets into `./tmp/*.so` during the run, so the
same toolchain module must be loaded in the batch job, not only at build time.

## 2. Submit everything

```bash
export MODULES="module load stack/2024-06 gcc/12.2.0 cmake suitesparse boost openmpi python"
export VENV=~/physbc-venv
source "$VENV/bin/activate"
bash scripts/submit_physbc_euler.sh                       # all 30 cases
# or filter by prefix:
bash scripts/submit_physbc_euler.sh StokesSecond Womersley
```

`MODULES`/`VENV` propagate to every job via `--export=ALL`. Override sbatch
resources with `SBATCH_ARGS`, e.g. `SBATCH_ARGS="--time=12:00:00 --ntasks=4"`.

### Single case (debug)
```bash
sbatch --export=ALL,BENCH=MMS2D_tangential_eps1em3_order1 \
  scripts/slurm_physbc_convergence_euler.sh
```
The runner script (`slurm_physbc_convergence_euler.sh`) picks the MPI binary and
solver matching the order in `BENCH`, generates the config+mesh sweep, runs each
level, and prints a convergence summary. Order-1 → GMRES, order-2 → MINRES; the
order-2 build includes the Nitsche-penalty consistency fix required for clean
2nd-order convergence.

## 3. Collect & plot

CSVs land in `out/data/<BENCH>/<BENCH>_conv_order*_ref*_vars.csv`. Then:

```bash
python scripts/plot_physbc_convergence.py        # all families, overlay orders 1&2 vs h
```
Writes `out/plots/physbc/<family>_order12_convergence.png` (log–log error vs h,
one curve per ε, O(hᵖ) reference lines).

Optional solution-field snapshots (per ε, same mesh; shows the layer sharpens
without instability):
```bash
python scripts/plot_physbc_solution_field.py --family StokesSecond --order 1 --level 2
```

## Notes / knobs
- `REFINES=<n>` (env, read by the runner) caps the refinement levels — handy for
  quick tests; omit it for the full class-default sweep on Euler.
- Boundary-layer ε must keep the coarsest level near h/δ ≲ 3 (δ=√(2ε/ω)); at
  h/δ ≳ 5 the under-resolved layer destabilises. The committed ε sets respect this.
- The exact inviscid limit ε=0 is intentionally excluded (FGMRES breaks down when
  the viscous/Nitsche block degenerates to mass-only — a separate open issue).
