#!/usr/bin/env bash
#SBATCH --job-name=flowcube-semilag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=4096

set -euo pipefail

# Run the flow-around-cube semi-Lagrangian case on Euler in one Slurm job.
#
# Submit:
#   sbatch scripts/slurm_flow_around_cube_semilag_euler.sh
#
# Useful overrides:
#   sbatch --cpus-per-task=16 --time=24:00:00 \
#     --export=ALL,THREADS=16,SCHEDULE_POLICY=dynamic,64 \
#     scripts/slurm_flow_around_cube_semilag_euler.sh
#
# Environment overrides:
#   EXE                  solver executable path
#   BASE_CONFIG          base JSON config path
#   THREADS              OpenMP thread count (default: SLURM_CPUS_PER_TASK)
#   SCHEDULE_POLICY      OMP_SCHEDULE (default: dynamic,64)
#   PROC_BIND_POLICY     OMP_PROC_BIND (default: close)
#   PLACES_POLICY        OMP_PLACES (default: cores)
#   DYNAMIC_POLICY       OMP_DYNAMIC (default: false)
#   OUTPUTFILE_OVERRIDE  override JSON outputfile string (optional)
#   BUILD_BINARY         set 1 to rebuild target before run

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXE="${EXE:-./build/semilagrangian_navierstokes_nitsche}"
BASE_CONFIG="${BASE_CONFIG:-data/config/FlowAroundCubeSemiLag/FlowAroundCubeSemiLag_outer100_cube000_order1_ref0.json}"

THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-dynamic,64}"
PROC_BIND_POLICY="${PROC_BIND_POLICY:-close}"
PLACES_POLICY="${PLACES_POLICY:-cores}"
DYNAMIC_POLICY="${DYNAMIC_POLICY:-false}"

BUILD_BINARY="${BUILD_BINARY:-0}"

if [[ ! -x "$EXE" ]]; then
    echo "[error] Executable not found or not executable: $EXE"
    exit 1
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
    echo "[info] Rebuilding solver binary..."
    cmake --build build --target semilagrangian_navierstokes_nitsche -j"${SLURM_CPUS_PER_TASK:-8}"
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
echo "[info] Base config:     $BASE_CONFIG"
echo "[info] Runtime config:  $RUNTIME_CFG"
echo "[info] Threads:         $THREADS"
echo "[info] OMP schedule:    $SCHEDULE_POLICY"
echo "[info] OMP places:      $PLACES_POLICY"
echo "[info] OMP proc bind:   $PROC_BIND_POLICY"
echo "[info] Output prefix:   out/data/${out_rel}"
echo "[info] Log file:        $LOG_FILE"

launcher=()
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
    launcher=(srun --cpu-bind=cores)
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
