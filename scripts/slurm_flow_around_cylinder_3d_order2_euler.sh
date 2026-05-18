#!/usr/bin/env bash
#SBATCH --job-name=cyl3d-o2
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=128
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=1024M
#SBATCH --output=logs/cyl3d_o2_%j.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Single full run of the ORDER-2 semi-Lagrangian Navier-Stokes solver on the
# 3D Schaefer-Turek flow-around-cylinder benchmark, on ETHZ Euler.
#
# The order-2 app is PURE OpenMP, SINGLE NODE (no MPI).  "Many cores" means one
# fat Euler node with many threads; it does NOT scale across nodes.  The
# semi-Lagrangian advection (edge loop, ~100 s/step at ND2 in 3D on this mesh)
# is the dominant cost and is the OpenMP-parallel phase; the MG/FGMRES solve is
# ~5 s/step with bounded iterations (~60).
#
# Usage on Euler (submit from the repo root):
#   sbatch scripts/slurm_flow_around_cylinder_3d_order2_euler.sh
#
# Common overrides at submission time:
#   sbatch --cpus-per-task=64 --time=48:00:00 \
#     --export=ALL,REFINEMENTS=0,T=10,DT=0.01 \
#     scripts/slurm_flow_around_cylinder_3d_order2_euler.sh
#
# Environment overrides (all optional, sane defaults):
#   EXE          solver binary (default ./build/semilagrangian_navierstokes_nitsche_order2)
#   BASE_CONFIG  3D order-1 config to clone (order is forced to 2)
#   REFINEMENTS  uniform h-refinements before the p-refine to order 2
#                (default 0; 1 is MUCH heavier in 3D — ~8x tets)
#   T, DT        end time / timestep (defaults from BASE_CONFIG if unset)
#   VIS          visualisation cadence in cycles (default 20)
#   MODULES      a command to set up the toolchain, e.g.
#                MODULES="module load stack/2024-06 gcc/12.2.0 cmake suitesparse boost"
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
# module names — they cannot be hardcoded portably). Needed at RUNTIME too.
if [[ -n "${MODULES:-}" ]]; then
    echo "[info] Setting up environment: $MODULES"
    eval "$MODULES"
fi

EXE="${EXE:-./build/semilagrangian_navierstokes_nitsche_order2}"
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

THREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[error] python3 not found in PATH (load a python module via MODULES)"
    exit 1
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "[error] Base config not found: $BASE_CONFIG"
    exit 1
fi

# Optional (re)build. Requires MFEM_USE_SUITESPARSE=ON (UMFPACK MG coarse solve).
if [[ "$BUILD_BINARY" == "1" || ! -x "$EXE" ]]; then
    echo "[info] Building semilagrangian_navierstokes_nitsche_order2 ..."
    cmake -S . -B build >/dev/null
    cmake --build build --target semilagrangian_navierstokes_nitsche_order2 \
        -j"${THREADS}"
fi
if [[ ! -x "$EXE" ]]; then
    echo "[error] Executable not found or not executable: $EXE"
    exit 1
fi

JOB_TAG="${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_REL="FlowAroundCylinderSemiLag/cyl3d_order2_${JOB_TAG}"

# The app does NOT mkdir the CSV directory (throws if missing); ParaView dir
# auto-creates.  ./tmp must exist in cwd: io.h JIT-writes config_library*.so
# there relative to the working directory.
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
echo "[info] Threads:        $THREADS"
echo "[info] OMP schedule:   $SCHEDULE_POLICY  bind:$PROC_BIND_POLICY  places:$PLACES_POLICY"
echo "[info] CSV  -> out/data/${OUTPUT_REL}_vars.csv"
echo "[info] PVD  -> out/paraview/${OUTPUT_REL}/$(basename "$OUTPUT_REL").pvd"

run_env=(
    env
    OMP_NUM_THREADS="$THREADS"
    OMP_DYNAMIC="$DYNAMIC_POLICY"
    OMP_PROC_BIND="$PROC_BIND_POLICY"
    OMP_PLACES="$PLACES_POLICY"
    OMP_SCHEDULE="$SCHEDULE_POLICY"
    "$EXE" -c "$RUNTIME_CFG"
)

set +e
if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --ntasks=1 --cpus-per-task="$THREADS" --cpu-bind=cores "${run_env[@]}"
else
    "${run_env[@]}"
fi
rc=$?
set -e

echo "[info] solver exit code: $rc"
echo "[info] CSV: out/data/${OUTPUT_REL}_vars.csv"
echo "[info] PVD: out/paraview/${OUTPUT_REL}/$(basename "$OUTPUT_REL").pvd"
exit $rc
