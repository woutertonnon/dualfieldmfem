#!/usr/bin/env bash
#SBATCH --job-name=physbc-conv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096M
#SBATCH --output=logs/physbc_conv_%j.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Convergence study for the semi-Lagrangian + Nitsche Navier-Stokes solver with
# PHYSICAL (no-slip) boundary conditions, on ETHZ Euler.
#
# Thesis chapter "Navier-Stokes with physical boundary conditions". One job runs
# the full h/tau sweep (tau coupled to h) for ONE registered case at one fixed
# viscosity epsilon and one polynomial order; the CSV at each refinement level
# carries the final-time L2(Omega) velocity error (column u1_err_L2). Run several
# jobs (one per epsilon) to assemble the epsilon-robustness picture, e.g.
#
#   for eps in 1em2 1em3 1em4 0; do
#     sbatch --export=ALL,BENCH=MMS2D_tangential_eps${eps}_order1 \
#       scripts/slurm_physbc_convergence_euler.sh
#   done
#
# The registry keys (BENCH) are produced by _build_physbc_benchmark_map() in
# scripts/semilagrangian_benchmarks.py. List them with:
#   python3 -c 'import sys; sys.path.insert(0,"scripts"); \
#     import semilagrangian_benchmarks as s; \
#     print("\n".join(sorted(s._build_physbc_benchmark_map())))'
# Examples:
#   MMS2D_tangential_eps1em3_order1   MMS2D_noslip_eps1em3_order2
#   EthierSteinman3D_eps1em2_order1   StokesSecond_eps1em3_order1
#   Womersley_eps1em4_order2
#
# Parallelism: the linear solve is replicated per rank; only the semi-Lagrangian
# advection is distributed over MPI ranks (partition-invariant, bit-exact) and
# OpenMP-threaded within a rank. For these convergence meshes the default of a
# single rank with cpus-per-task OpenMP threads is plenty; bump --ntasks only
# for the finer 3D Ethier-Steinman levels.
#
# Required:
#   BENCH        a registry key (see above)
#
# Optional environment overrides (sane defaults):
#   EXE          solver binary. Default: the MPI binary matching the order
#                inferred from BENCH (..._order2 -> *_order2_mpi, else *_mpi).
#   SOLVER       linear solver string (default GMRES; order-2 uses MINRES).
#   TOL          linear solver tolerance (default 1e-8).
#   PYTHON       python interpreter that has gmsh, sympy, pandas, matplotlib,
#                paramiko (used only to GENERATE configs + meshes). Default python3.
#   VENV         path to a virtualenv to `source <VENV>/bin/activate` first.
#   MODULES      command(s) to set up the toolchain, e.g.
#                MODULES="module load stack/2024-06 gcc/12.2.0 cmake suitesparse boost openmpi python"
#                IMPORTANT: the C++ compiler is a RUNTIME dependency — io.h
#                JIT-compiles the force/boundary/initial-data snippets into
#                ./tmp/*.so during the run, so the toolchain must be loaded here.
#   BUILD_BINARY set to 1 to (re)build the target before running.
#   OMP_*        OMP_SCHEDULE_VALUE / OMP_PROC_BIND_VALUE / OMP_PLACES_VALUE / OMP_DYNAMIC_VALUE
# ---------------------------------------------------------------------------

# Resolve repo root (Slurm may run a spool copy of this script).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT_DIR"

: "${BENCH:?set BENCH=<registry key, e.g. MMS2D_tangential_eps1em3_order1>}"

# Optional toolchain / module setup (needed at RUNTIME too: JIT compile).
if [[ -n "${MODULES:-}" ]]; then
    echo "[info] Setting up environment: $MODULES"
    eval "$MODULES"
fi
if [[ -n "${VENV:-}" ]]; then
    echo "[info] Activating virtualenv: $VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

PYTHON="${PYTHON:-python3}"
SOLVER_DEFAULT="GMRES"
if [[ "$BENCH" == *order2* ]]; then
    SOLVER_DEFAULT="MINRES"
fi
SOLVER="${SOLVER:-$SOLVER_DEFAULT}"
TOL="${TOL:-1e-8}"

# Default executable: the MPI binary matching the order in BENCH.
if [[ "$BENCH" == *order2* ]]; then
    EXE="${EXE:-./build/semilagrangian_navierstokes_nitsche_order2_mpi}"
else
    EXE="${EXE:-./build/semilagrangian_navierstokes_nitsche_mpi}"
fi

SCHEDULE_POLICY="${OMP_SCHEDULE_VALUE:-dynamic,64}"
PROC_BIND_POLICY="${OMP_PROC_BIND_VALUE:-close}"
PLACES_POLICY="${OMP_PLACES_VALUE:-cores}"
DYNAMIC_POLICY="${OMP_DYNAMIC_VALUE:-false}"

BUILD_BINARY="${BUILD_BINARY:-0}"
RANKS="${SLURM_NTASKS:-1}"
THREADS="${SLURM_CPUS_PER_TASK:-4}"
BUILD_JOBS="${SLURM_CPUS_ON_NODE:-$(nproc)}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "[error] python interpreter '$PYTHON' not found (load a python module via MODULES or set PYTHON/VENV)"
    exit 1
fi

mkdir -p tmp logs
export TMPDIR="${TMPDIR:-$ROOT_DIR/tmp}"

# Optional (re)build of the MPI solver binary.
TARGET="$(basename "$EXE")"
if [[ "$BUILD_BINARY" == "1" || ! -x "$EXE" ]]; then
    echo "[info] Building $TARGET ..."
    cmake -S . -B build >/dev/null
    cmake --build build --target "$TARGET" -j"${BUILD_JOBS}"
fi
if [[ ! -x "$EXE" ]]; then
    echo "[error] Executable not found or not executable: $EXE"
    exit 1
fi

# ---------------------------------------------------------------------------
# 1) Generate the config sweep (and the mesh) for this case. This writes
#    data/config/<NAME>/<NAME>_conv_order<o>_ref<r>.json for every refinement
#    level, and the gmsh mesh under geo/mesh/. NAME == BENCH.
# ---------------------------------------------------------------------------
echo "[info] Generating config sweep for BENCH=$BENCH (solver=$SOLVER tol=$TOL) ..."
NAME="$(BENCH="$BENCH" SOLVER="$SOLVER" TOL="$TOL" "$PYTHON" - <<'PY'
import os, sys
sys.path.insert(0, "scripts")
# Route library chatter (mesh-gen "Wrote ..." prints etc.) to stderr so that
# only the resolved case name lands on stdout for capture.
real_stdout = sys.stdout
sys.stdout = sys.stderr
import semilagrangian_benchmarks as slb
bm = slb._build_physbc_benchmark_map()
key = os.environ["BENCH"]
if key not in bm:
    sys.stderr.write(f"unknown BENCH '{key}'. Available:\n  " +
                     "\n  ".join(sorted(bm)) + "\n")
    sys.exit(2)
b = bm[key](solver=os.environ["SOLVER"])
b.tol = float(os.environ["TOL"])
# Optional REFINES override: cap the number of h-refinement levels (r=0..N-1).
# Useful for quick local runs; the full sweep (the class default) is meant for
# the cluster.
_ref = os.environ.get("REFINES")
if _ref:
    _n = int(_ref)
    b.refinements = lambda order_: range(0, _n)
b.SimulationHelper.generate_config_files(b.T, b.dts, b.refinements, b.orders, tol=b.tol)
sys.stdout = real_stdout
print(b.name)
PY
)"
echo "[info] Case name: $NAME"

CONFIG_DIR="data/config/$NAME"
shopt -s nullglob
CONFIGS=("$CONFIG_DIR"/*.json)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "[error] No configs generated in $CONFIG_DIR"
    exit 1
fi
# The app throws if the CSV output directory is missing; ParaView auto-creates.
mkdir -p "out/data/$NAME"

echo "[info] Executable:   $EXE"
echo "[info] MPI ranks:    $RANKS   OpenMP threads/rank: $THREADS"
echo "[info] ${#CONFIGS[@]} refinement level(s) to run:"
printf '         %s\n' "${CONFIGS[@]}"

omp_env=(
    OMP_NUM_THREADS="$THREADS"
    OMP_DYNAMIC="$DYNAMIC_POLICY"
    OMP_PROC_BIND="$PROC_BIND_POLICY"
    OMP_PLACES="$PLACES_POLICY"
    OMP_SCHEDULE="$SCHEDULE_POLICY"
)

run_one() {
    local cfg="$1"
    echo "[info] === running $cfg ==="
    if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
        srun --ntasks="$RANKS" --cpus-per-task="$THREADS" --cpu-bind=cores \
            env "${omp_env[@]}" "$EXE" -c "$cfg"
    elif command -v mpirun >/dev/null 2>&1 && [[ "$RANKS" -gt 1 ]]; then
        mpirun -np "$RANKS" env "${omp_env[@]}" "$EXE" -c "$cfg"
    else
        env "${omp_env[@]}" "$EXE" -c "$cfg"
    fi
}

rc_all=0
for cfg in "${CONFIGS[@]}"; do
    set +e
    run_one "$cfg"
    rc=$?
    set -e
    [[ $rc -ne 0 ]] && rc_all=$rc && echo "[warn] $cfg exited with $rc"
done

# ---------------------------------------------------------------------------
# 2) Summarize: final-time L2 velocity error per refinement level + rates.
# ---------------------------------------------------------------------------
echo "[info] === convergence summary (final-time u1_err_L2) ==="
NAME="$NAME" "$PYTHON" - <<'PY' || true
import os, glob, math, csv
name = os.environ["NAME"]
rows = []
for f in sorted(glob.glob(f"out/data/{name}/{name}_conv_order*_ref*_vars.csv")):
    try:
        with open(f) as fh:
            r = list(csv.DictReader(fh))
        if not r:
            continue
        col = next((c for c in r[-1] if "err" in c.lower()), None)
        err = float(r[-1][col]) if col else float("nan")
        ref = int(f.split("_ref")[1].split("_")[0])
        rows.append((ref, err, len(r), f))
    except Exception as e:
        print(f"  [skip] {f}: {e}")
rows.sort()
prev = None
for ref, err, n, f in rows:
    rate = ""
    if prev and prev[1] > 0 and err > 0:
        rate = f"  rate={math.log(prev[1]/err)/math.log(2):+.2f}"
    print(f"  ref{ref}: nsteps={n:4d}  L2_err={err:.6e}{rate}")
    prev = (ref, err)
PY

echo "[info] CSVs under out/data/$NAME/"
echo "[info] overall exit code: $rc_all"
exit $rc_all
