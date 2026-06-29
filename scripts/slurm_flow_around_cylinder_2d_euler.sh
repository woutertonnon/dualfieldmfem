#!/usr/bin/env bash
#SBATCH --job-name=cyl2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096M
#SBATCH --output=logs/cyl2d_%j.log
set -euo pipefail

# ---------------------------------------------------------------------------
# 2D Schaefer-Turek flow-around-cylinder benchmark (order-2 semi-Lagrangian
# Nitsche solver, MPI binary) on ETHZ Euler.  Domain (0,2.2)x(0,0.41), cylinder
# D=0.1 at (0.2,0.2); parabolic inflow (tag 2), no-slip walls (4) + cylinder (5),
# consistent-Nitsche "do-nothing" pressure outflow (tag 3).  Reports c_D, c_L,
# Delta p and (unsteady) the Strouhal number via the QoI CSV.
#
# Reynolds number selects the peak inflow U_m (Ubar = 2/3 U_m, Re = Ubar*D/nu):
#   RE=20  -> U_m=0.3  (steady;   reference c_D~5.58, c_L~0.0107, dp~0.117)
#   RE=100 -> U_m=1.5  (unsteady; reference c_D,max~3.23, c_L,max~1.0, St~0.30)
#
# Usage on Euler (from repo root, venv + libGLU on LD_LIBRARY_PATH not needed
# here -- no gmsh; the committed mesh is used directly):
#   sbatch --export=ALL,RE=20  scripts/slurm_flow_around_cylinder_2d_euler.sh
#   sbatch --export=ALL,RE=100 scripts/slurm_flow_around_cylinder_2d_euler.sh
#
# Env overrides: NU, UM, T, DT, GAMMA (outflow penalty), REFINEMENTS, VIS,
#   EXE, BUILD_BINARY=1.
# ---------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if [[ -n "${MODULES:-}" ]]; then eval "$MODULES"; fi

RE="${RE:-20}"
if [[ "$RE" == "100" ]]; then
    UM_DEFAULT=1.5;  T_DEFAULT=8.0;  DT_DEFAULT=0.005
else
    UM_DEFAULT=0.3;  T_DEFAULT=10.0; DT_DEFAULT=0.05
fi
UM="${UM:-$UM_DEFAULT}"
NU="${NU:-0.001}"
T="${T:-$T_DEFAULT}"
DT="${DT:-$DT_DEFAULT}"
GAMMA="${GAMMA:-100.0}"
REFINEMENTS="${REFINEMENTS:-0}"
VIS="${VIS:-0}"
EXE="${EXE:-./build/semilagrangian_navierstokes_nitsche_order2_mpi}"

RANKS="${SLURM_NTASKS:-1}"
THREADS="${SLURM_CPUS_PER_TASK:-16}"

if [[ ! -f "geo/mesh/flow_around_cylinder_2d.msh" ]]; then
    echo "[error] mesh geo/mesh/flow_around_cylinder_2d.msh missing"
    echo "        regenerate: python geo/gmsh/flow_around_cylinder_2d.py"
    exit 1
fi

if [[ "${BUILD_BINARY:-0}" == "1" || ! -x "$EXE" ]]; then
    echo "[info] building order-2 MPI binary ..."
    cmake -S . -B build >/dev/null
    cmake --build build --target semilagrangian_navierstokes_nitsche_order2_mpi \
        -j"${SLURM_CPUS_ON_NODE:-8}"
fi

JOB_TAG="${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_REL="FlowAroundCylinder2D/cyl2d_Re${RE}_${JOB_TAG}"
mkdir -p "out/data/$(dirname "$OUTPUT_REL")" out/paraview tmp logs tmp/job_configs
export TMPDIR="${TMPDIR:-$ROOT_DIR/tmp}"

CFG="tmp/job_configs/cyl2d_Re${RE}_${JOB_TAG}.json"
UBAR=$(python3 -c "print(2.0/3.0*$UM)")

# Parabolic inflow profile (2D): u_x = 4 U_m y (0.41-y)/0.41^2, u_y = 0.
INFLOW="out[0] = 4.0*${UM}*x[1]*(0.41-x[1])/(0.41*0.41); out[1] = 0;"

cat > "$CFG" <<JSON
{
    "mesh": "./geo/mesh/flow_around_cylinder_2d.msh",
    "solver": "MINRES",
    "visualisation": ${VIS},
    "printlevel": 1,
    "outputfile": "${OUTPUT_REL}",
    "order": 2,
    "refinements": ${REFINEMENTS},
    "tol": 1e-08,
    "dt": ${DT},
    "trace_order": 2,
    "settls_iterations": 1,
    "vertex_velocity_mode": "edge_dihedral",
    "T": ${T},
    "viscosity": ${NU},
    "lid_attributes": [2],
    "outflow_attributes": [3],
    "outflow_penalty": ${GAMMA},
    "qoi_cylinder_attribute": 5,
    "qoi_Ubar": ${UBAR},
    "qoi_diameter": 0.1,
    "force_data": "out[0] = 0; out[1] = 0;",
    "initial_data_u": "${INFLOW}",
    "boundary_data_u": "${INFLOW}"
}
JSON

echo "[info] Re=${RE}  U_m=${UM}  Ubar=${UBAR}  nu=${NU}  Re_check=$(python3 -c "print($UBAR*0.1/$NU)")"
echo "[info] T=${T} dt=${DT} gamma=${GAMMA} refinements=${REFINEMENTS}"
echo "[info] config: $CFG"
echo "[info] QoI CSV -> out/data/${OUTPUT_REL}_qoi.csv"

omp_env=(OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=spread OMP_PLACES=cores)
if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --ntasks="$RANKS" --cpus-per-task="$THREADS" --cpu-bind=cores \
        env "${omp_env[@]}" "$EXE" -c "$CFG"
else
    env "${omp_env[@]}" "$EXE" -c "$CFG"
fi

echo "[done] tail of QoI CSV:"
tail -5 "out/data/${OUTPUT_REL}_qoi.csv" 2>/dev/null || true
