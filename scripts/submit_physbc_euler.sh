#!/usr/bin/env bash
# Submit ALL physical-BC convergence studies to Euler — one SLURM job per case.
#
# Each job runs the full h/tau sweep (class-default refinement levels; NOT the
# local REFINES cap) for one (experiment, viscosity, order) and writes a CSV per
# level with the final-time L2 velocity error. The set of cases is taken from
# the registry in scripts/semilagrangian_benchmarks.py, so it stays in sync.
#
# Prerequisites on the Euler LOGIN node before running this:
#   1. A Python env with the deps in scripts/requirements_euler.txt, e.g.
#        python -m venv ~/physbc-venv
#        source ~/physbc-venv/bin/activate
#        pip install -r scripts/requirements_euler.txt
#   2. The toolchain modules that match the build AND the runtime JIT compile.
#
# Configure via environment (passed through to every job via --export=ALL):
#   MODULES   module-load command, e.g.
#             export MODULES="module load stack/2024-06 gcc/12.2.0 cmake suitesparse boost openmpi python"
#   VENV      path to the Python venv, e.g.  export VENV=~/physbc-venv
#   PYTHON    python interpreter (default python3; the venv's python if VENV set)
#   BUILD_BINARY=1   to (re)build the solver inside the first jobs
#   Plus any sbatch overrides via SBATCH_ARGS (e.g. SBATCH_ARGS="--time=12:00:00").
#
# Usage:
#   export MODULES="module load ..."; export VENV=~/physbc-venv
#   source "$VENV/bin/activate"
#   bash scripts/submit_physbc_euler.sh                 # submit everything
#   bash scripts/submit_physbc_euler.sh MMS2D EthierSteinman3D   # filter by prefix
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-python3}"
SBATCH_ARGS="${SBATCH_ARGS:-}"
SLURM_SCRIPT="scripts/slurm_physbc_convergence_euler.sh"

# Pull the full list of registry keys (BENCH names) from the benchmark module.
mapfile -t ALL_KEYS < <("$PYTHON" - <<'PY'
import sys
sys.path.insert(0, "scripts")
import semilagrangian_benchmarks as slb
for k in sorted(slb._build_physbc_benchmark_map()):
    print(k)
PY
)

if [[ ${#ALL_KEYS[@]} -eq 0 ]]; then
    echo "[error] could not enumerate benchmark keys — is the Python env set up?"
    exit 1
fi

# Optional positional filters: only submit keys whose name contains any filter.
FILTERS=("$@")
match() {
    [[ ${#FILTERS[@]} -eq 0 ]] && return 0
    local k="$1"
    for f in "${FILTERS[@]}"; do [[ "$k" == *"$f"* ]] && return 0; done
    return 1
}

n=0
for k in "${ALL_KEYS[@]}"; do
    match "$k" || continue
    echo "[submit] $k"
    # shellcheck disable=SC2086
    sbatch --job-name="pb-$k" $SBATCH_ARGS \
        --export=ALL,BENCH="$k" \
        "$SLURM_SCRIPT"
    n=$((n + 1))
done
echo "[done] submitted $n job(s)."
echo "       results: out/data/<BENCH>/<BENCH>_conv_order*_ref*_vars.csv"
echo "       plot with: $PYTHON scripts/plot_physbc_convergence.py"
