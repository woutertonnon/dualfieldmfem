#!/usr/bin/env bash
#SBATCH --job-name=flowcube-semilag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=4096

set -euo pipefail

# Run the flow-around-cube semi-Lagrangian case on Euler in one Slurm job.
# Supports both the serial and MPI executables (hybrid MPI+OpenMP).
#
# Submit (serial, default):
#   sbatch scripts/slurm_flow_around_cube_semilag_euler.sh
#
# Submit (MPI, 2 nodes x 8 threads):
#   sbatch --nodes=2 --ntasks-per-node=1 --cpus-per-task=8 \
#     --export=ALL,USE_MPI=1 \
#     scripts/slurm_flow_around_cube_semilag_euler.sh
#
# Submit (MPI, 4 ranks on 1 node x 4 threads each):
#   sbatch --nodes=1 --ntasks-per-node=4 --cpus-per-task=4 \
#     --export=ALL,USE_MPI=1 \
#     scripts/slurm_flow_around_cube_semilag_euler.sh
#
# Environment overrides:
#   USE_MPI              set 1 to use the MPI executable (default: 0 = serial)
#   EXE                  solver executable path (auto-selected based on USE_MPI)
#   BASE_CONFIG          base JSON config path
#   THREADS              OpenMP threads per rank (default: SLURM_CPUS_PER_TASK)
#   SCHEDULE_POLICY      OMP_SCHEDULE (default: dynamic,64)
#   PROC_BIND_POLICY     OMP_PROC_BIND (default: close)
#   PLACES_POLICY        OMP_PLACES (default: cores)
#   DYNAMIC_POLICY       OMP_DYNAMIC (default: false)
#   OUTPUTFILE_OVERRIDE  override JSON outputfile string (optional)
#   BUILD_BINARY         set 1 to rebuild target before run (auto-enabled if EXE is missing)

# In Slurm batch mode, BASH_SOURCE may point to a spool copy of the script.
# Prefer submission directory (repo root when submitted from there).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT_DIR"

USE_MPI="${USE_MPI:-0}"
if [[ "$USE_MPI" == "1" ]]; then
    EXE="${EXE:-./build/semilagrangian_navierstokes_nitsche_mpi}"
    BUILD_TARGET="semilagrangian_navierstokes_nitsche_mpi"
else
    EXE="${EXE:-./build/semilagrangian_navierstokes_nitsche}"
    BUILD_TARGET="semilagrangian_navierstokes_nitsche"
fi
BASE_CONFIG="${BASE_CONFIG:-data/config/FlowAroundCubeSemiLag/FlowAroundCubeSemiLag_outer100_cube000_order1_ref0.json}"

THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-dynamic,64}"
PROC_BIND_POLICY="${PROC_BIND_POLICY:-close}"
PLACES_POLICY="${PLACES_POLICY:-cores}"
DYNAMIC_POLICY="${DYNAMIC_POLICY:-false}"

BUILD_BINARY="${BUILD_BINARY:-0}"

NRANKS="${SLURM_NTASKS:-1}"
NNODES="${SLURM_NNODES:-1}"

if [[ ! -x "$EXE" ]]; then
    echo "[warn] Executable not found: $EXE"
    echo "[warn] Enabling BUILD_BINARY=1 to build it inside the job"
    BUILD_BINARY=1
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "[error] Base config not found: $BASE_CONFIG"
    exit 1
fi
if ! [[ "$THREADS" =~ ^[0-9]+$ ]] || (( THREADS < 1 )); then
    echo "[error] THREADS must be a positive integer; got '$THREADS'"
    exit 1
fi

if [[ "$BUILD_BINARY" == "1" ]]; then
    if [[ ! -f "build/CMakeCache.txt" ]]; then
        echo "[info] Configuring CMake build directory..."
        cmake -S . -B build
    fi
    echo "[info] Building solver binary..."
    cmake --build build --target "$BUILD_TARGET" -j"${SLURM_CPUS_PER_TASK:-8}"
fi

if [[ ! -x "$EXE" ]]; then
    echo "[error] Executable still not found or not executable after build step: $EXE"
    echo "[error] Override with --export=ALL,EXE=<path-to-binary> if needed"
    exit 1
fi

JOB_TAG="${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}"
RUNTIME_CFG_DIR="$ROOT_DIR/tmp/flow_around_cube_runs"
mkdir -p "$RUNTIME_CFG_DIR"
RUNTIME_CFG="$RUNTIME_CFG_DIR/FlowAroundCubeSemiLag_${JOB_TAG}.json"

if [[ -n "${OUTPUTFILE_OVERRIDE:-}" ]]; then
    out_rel="$OUTPUTFILE_OVERRIDE"
else
    out_rel="FlowAroundCubeSemiLag/FlowAroundCubeSemiLag_outer100_cube000_order1_ref0_${JOB_TAG}"
fi

mkdir -p "$ROOT_DIR/out/data/$(dirname "$out_rel")"

escaped_out="$(printf '%s' "$out_rel" | sed -e 's/[\\&|]/\\&/g')"
cp "$BASE_CONFIG" "$RUNTIME_CFG"
if grep -qE '^[[:space:]]*"outputfile"[[:space:]]*:' "$RUNTIME_CFG"; then
    sed -i -E "s|^([[:space:]]*\"outputfile\"[[:space:]]*:[[:space:]]*\")[^\"]*(\"[[:space:]]*,[[:space:]]*)$|\\1${escaped_out}\\2|" "$RUNTIME_CFG"
else
    echo "[error] Could not find outputfile key in $RUNTIME_CFG"
    exit 1
fi

LOG_DIR="$ROOT_DIR/out/data/FlowAroundCubeSemiLag/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_${JOB_TAG}.log"

echo "[info] Root:            $ROOT_DIR"
echo "[info] Executable:      $EXE"
echo "[info] Mode:            $(if [[ "$USE_MPI" == "1" ]]; then echo "MPI+OpenMP (hybrid)"; else echo "OpenMP (serial)"; fi)"
echo "[info] Base config:     $BASE_CONFIG"
echo "[info] Runtime config:  $RUNTIME_CFG"
if [[ "$USE_MPI" == "1" ]]; then
    echo "[info] Nodes:           $NNODES"
    echo "[info] MPI ranks:       $NRANKS"
fi
echo "[info] Threads/rank:    $THREADS"
echo "[info] OMP schedule:    $SCHEDULE_POLICY"
echo "[info] OMP places:      $PLACES_POLICY"
echo "[info] OMP proc bind:   $PROC_BIND_POLICY"
echo "[info] Output prefix:   out/data/${out_rel}"
echo "[info] Log file:        $LOG_FILE"

if [[ "$USE_MPI" == "1" ]]; then
    # Hybrid MPI+OpenMP launch via srun
    launcher=(srun
        --ntasks="$NRANKS"
        --cpus-per-task="$THREADS"
        --cpu-bind=cores)
else
    launcher=()
    if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
        launcher=(srun --cpu-bind=cores)
    fi
fi

set +e
env \
    OMP_NUM_THREADS="$THREADS" \
    OMP_DYNAMIC="$DYNAMIC_POLICY" \
    OMP_PROC_BIND="$PROC_BIND_POLICY" \
    OMP_PLACES="$PLACES_POLICY" \
    OMP_SCHEDULE="$SCHEDULE_POLICY" \
    "${launcher[@]}" "$EXE" -c "$RUNTIME_CFG" 2>&1 | tee "$LOG_FILE"
rc=${PIPESTATUS[0]}
set -e

if (( rc != 0 )); then
    echo "[error] Solver failed with exit code $rc"
    exit "$rc"
fi

echo "[info] Done. CSV: out/data/${out_rel}_vars.csv"
