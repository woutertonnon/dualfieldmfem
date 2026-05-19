#!/usr/bin/env bash
#SBATCH --job-name=cyl3d-o2
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=128
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=1024M
#SBATCH --output=logs/cyl3d_o2_%j.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Full run of the ORDER-2 semi-Lagrangian Navier-Stokes solver on the 3D
# Schaefer-Turek flow-around-cylinder benchmark, on ETHZ Euler, using the
# MPI binary (semilagrangian_navierstokes_nitsche_order2_mpi).
#
# Parallelism model (hybrid MPI + OpenMP):
#   * The semi-Lagrangian ADVECTION (the ~100 s/step bottleneck at ND2 in 3D)
#     is distributed over MPI ranks by an element partition with per-ND2-DOF
#     ownership; the result is partition-invariant (np-independent, bit-exact).
#   * The StokesMG / FGMRES linear solve is REPLICATED on every rank (serial
#     mesh, no ParMesh) — it does NOT scale with ranks and each rank holds the
#     full UMFPACK coarse factorization.  => per-rank memory ~ a serial run;
#     plan --mem-per-cpu so that (cpus-per-task * mem-per-cpu) per rank covers
#     the 3D order-2 UMFPACK factors (GB-scale; the default 8*4G=32G/rank is a
#     safe envelope for refinements=0).
#   * Within a rank, the advection element loop is OpenMP-threaded
#     (cpus-per-task threads).
#
# So: --ntasks = MPI ranks (scales the advection across nodes),
#     --cpus-per-task = OpenMP threads per rank.
#
# Usage on Euler (submit from the repo root):
#   sbatch scripts/slurm_flow_around_cylinder_3d_order2_euler.sh
#
# Common overrides at submission time (sized for one job):
#   sbatch --nodes=4 --ntasks=16 --cpus-per-task=8 --mem-per-cpu=4G \
#     --time=48:00:00 \
#     --export=ALL,REFINEMENTS=0,"MODULES=module load <stack/gcc/cmake/suitesparse/boost/python>" \
#     scripts/slurm_flow_around_cylinder_3d_order2_euler.sh
#
# Environment overrides (all optional, sane defaults):
#   EXE          solver binary
#                (default ./build/semilagrangian_navierstokes_nitsche_order2_mpi)
#   BASE_CONFIG  3D order-1 config to clone (order is forced to 2)
#   REFINEMENTS  uniform h-refinements before the p-refine to order 2
#                (default 0; 1 is MUCH heavier in 3D — ~8x tets — and ~8x
#                 the per-rank UMFPACK memory)
#   T, DT        end time / timestep (defaults from BASE_CONFIG if unset)
#   VIS          visualisation cadence in cycles (default 20)
#   MODULES      a command to set up the toolchain, e.g.
#                MODULES="module load stack/2024-06 gcc/12.2.0 cmake suitesparse boost openmpi python"
#                IMPORTANT: the C++ compiler is a RUNTIME dependency — io.h
#                JIT-compiles the force/boundary/initial-data snippets into
#                ./tmp/*.so during the run, so the same toolchain must be
#                loaded in this batch job, not only at build time.
#   BUILD_BINARY set to 1 to (re)build the target before running
#   OMP_SCHEDULE_VALUE / OMP_PROC_BIND_VALUE / OMP_PLACES_VALUE / OMP_DYNAMIC_VALUE
# ---------------------------------------------------------------------------

# Resolve repo root (Slurm may run a spool copy of this script).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT_DIR"

# Optional environment / module setup (Euler-specific; user supplies the exact
# module names — they cannot be hardcoded portably). Needed at RUNTIME too
# (JIT compile) and must include an MPI module matching the build.
if [[ -n "${MODULES:-}" ]]; then
    echo "[info] Setting up environment: $MODULES"
    eval "$MODULES"
fi

EXE="${EXE:-./build/semilagrangian_navierstokes_nitsche_order2_mpi}"
BASE_CONFIG="${BASE_CONFIG:-data/config/FlowAroundCylinderSemiLag/FlowAroundCylinderSemiLag_order1_ref1.json}"

REFINEMENTS="${REFINEMENTS:-0}"
T_OVERRIDE="${T:-}"
DT_OVERRIDE="${DT:-}"
VIS="${VIS:-20}"

SCHEDULE_POLICY="${OMP_SCHEDULE_VALUE:-dynamic,64}"
PROC_BIND_POLICY="${OMP_PROC_BIND_VALUE:-close}"
PLACES_POLICY="${OMP_PLACES_VALUE:-cores}"
DYNAMIC_POLICY="${OMP_DYNAMIC_VALUE:-false}"

BUILD_BINARY="${BUILD_BINARY:-0}"

# MPI ranks and OpenMP threads/rank come from the Slurm allocation.
RANKS="${SLURM_NTASKS:-2}"
THREADS="${SLURM_CPUS_PER_TASK:-4}"
BUILD_JOBS="${SLURM_CPUS_ON_NODE:-$(nproc)}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[error] python3 not found in PATH (load a python module via MODULES)"
    exit 1
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
    # data/config/ is git-ignored (data/config/.gitignore = '*'), so the base
    # config never reaches a fresh Euler checkout via git.  The committed mesh
    # IS present, so synthesize the canonical 3D Schaefer-Turek order-1 config
    # here, making this script self-contained.
    echo "[info] Base config '$BASE_CONFIG' absent (git-ignored) — generating default."
    mkdir -p "$(dirname "$BASE_CONFIG")"
    cat > "$BASE_CONFIG" <<'JSON'
{
    "mesh": "./geo/mesh/flow_around_cylinder_3d.msh",
    "solver": "MINRES",
    "visualisation": 20,
    "printlevel": 1,
    "outputfile": "FlowAroundCylinderSemiLag/FlowAroundCylinderSemiLag_order1_ref1",
    "order": 1,
    "refinements": 1,
    "tol": 1e-05,
    "dt": 0.01,
    "trace_order": 2,
    "settls_iterations": 1,
    "vertex_velocity_mode": "edge_dihedral",
    "T": 10,
    "viscosity": 0.001,
    "lid_attributes": [2, 3, 4],
    "force_data": "out[0] = 0; out[1] = 0; out[2] = 0;",
    "initial_data_u": "out[0] = 16.0*2.25*x[1]*x[2]*(0.41-x[1])*(0.41-x[2])/(0.41*0.41*0.41*0.41); out[1] = 0; out[2] = 0;",
    "boundary_data_u": "out[0] = 16.0*2.25*x[1]*x[2]*(0.41-x[1])*(0.41-x[2])/(0.41*0.41*0.41*0.41); out[1] = 0; out[2] = 0;"
}
JSON
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "[error] Base config could not be created: $BASE_CONFIG"
    exit 1
fi
if [[ ! -f "geo/mesh/flow_around_cylinder_3d.msh" ]]; then
    echo "[error] 3D mesh geo/mesh/flow_around_cylinder_3d.msh missing."
    echo "        It is git-tracked — ensure the Euler checkout is up to date"
    echo "        (git pull), or regenerate via geo/gmsh/flow_around_cylinder_3d.py."
    exit 1
fi

# Optional (re)build. Requires MFEM_USE_SUITESPARSE=ON (UMFPACK MG coarse
# solve) and an MPI toolchain.
if [[ "$BUILD_BINARY" == "1" || ! -x "$EXE" ]]; then
    echo "[info] Building semilagrangian_navierstokes_nitsche_order2_mpi ..."
    cmake -S . -B build >/dev/null
    cmake --build build --target semilagrangian_navierstokes_nitsche_order2_mpi \
        -j"${BUILD_JOBS}"
fi
if [[ ! -x "$EXE" ]]; then
    echo "[error] Executable not found or not executable: $EXE"
    exit 1
fi

JOB_TAG="${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_REL="FlowAroundCylinderSemiLag/cyl3d_order2_${JOB_TAG}"

# The app does NOT mkdir the CSV directory (throws if missing); ParaView dir
# auto-creates.  ./tmp must exist in cwd: io.h JIT-writes config_library*.so
# there relative to the working directory (pid-unique names → no rank clash).
mkdir -p "out/data/$(dirname "$OUTPUT_REL")" out/paraview tmp logs \
         tmp/job_configs
export TMPDIR="${TMPDIR:-$ROOT_DIR/tmp}"

RUNTIME_CFG="tmp/job_configs/cyl3d_order2_${JOB_TAG}.json"
BASE_CONFIG="$BASE_CONFIG" OUT_CFG="$RUNTIME_CFG" OUTPUT_REL="$OUTPUT_REL" \
REFINEMENTS="$REFINEMENTS" T_OVERRIDE="$T_OVERRIDE" \
DT_OVERRIDE="$DT_OVERRIDE" VIS="$VIS" \
python3 - <<'PY'
import json, os
cfg = json.loads(open(os.environ["BASE_CONFIG"]).read())
cfg["order"] = 2
cfg["refinements"] = int(os.environ["REFINEMENTS"])
cfg["outputfile"] = os.environ["OUTPUT_REL"]
cfg["visualisation"] = int(os.environ["VIS"])
if os.environ.get("T_OVERRIDE"):
    cfg["T"] = float(os.environ["T_OVERRIDE"])
if os.environ.get("DT_OVERRIDE"):
    cfg["dt"] = float(os.environ["DT_OVERRIDE"])
open(os.environ["OUT_CFG"], "w").write(json.dumps(cfg, indent=4) + "\n")
print(f"[info] config: order=2 refinements={cfg['refinements']} "
      f"dt={cfg['dt']} T={cfg['T']} mesh={cfg['mesh']}")
PY

echo "[info] Root:           $ROOT_DIR"
echo "[info] Executable:     $EXE"
echo "[info] Runtime config: $RUNTIME_CFG"
echo "[info] MPI ranks:      $RANKS    OpenMP threads/rank: $THREADS"
echo "[info] OMP schedule:   $SCHEDULE_POLICY  bind:$PROC_BIND_POLICY  places:$PLACES_POLICY"
echo "[info] CSV  -> out/data/${OUTPUT_REL}_vars.csv"
echo "[info] PVD  -> out/paraview/${OUTPUT_REL}/$(basename "$OUTPUT_REL").pvd"

# Per-rank OpenMP environment.
omp_env=(
    OMP_NUM_THREADS="$THREADS"
    OMP_DYNAMIC="$DYNAMIC_POLICY"
    OMP_PROC_BIND="$PROC_BIND_POLICY"
    OMP_PLACES="$PLACES_POLICY"
    OMP_SCHEDULE="$SCHEDULE_POLICY"
)

set +e
if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --ntasks="$RANKS" --cpus-per-task="$THREADS" --cpu-bind=cores \
        env "${omp_env[@]}" "$EXE" -c "$RUNTIME_CFG"
elif command -v mpirun >/dev/null 2>&1; then
    # Local / non-Slurm fallback.
    mpirun -np "$RANKS" env "${omp_env[@]}" "$EXE" -c "$RUNTIME_CFG"
else
    echo "[error] neither srun nor mpirun available to launch the MPI binary"
    exit 1
fi
rc=$?
set -e

echo "[info] solver exit code: $rc"
echo "[info] CSV: out/data/${OUTPUT_REL}_vars.csv"
echo "[info] PVD: out/paraview/${OUTPUT_REL}/$(basename "$OUTPUT_REL").pvd"
exit $rc
