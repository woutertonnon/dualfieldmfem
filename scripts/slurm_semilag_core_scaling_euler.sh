#!/usr/bin/env bash
#SBATCH --job-name=semilag-core-scale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=4096

set -euo pipefail

# One-job core-scaling sweep for semilagrangian_navierstokes_nitsche.
#
# Usage on Euler:
#   sbatch scripts/slurm_semilag_core_scaling_euler.sh
#
# Optional overrides at submission time:
#   sbatch --cpus-per-task=32 \
#     --export=ALL,THREAD_COUNTS=1,2,4,8,16,32,REPEATS=3,SCHEDULE_POLICY=dynamic,64 \
#     scripts/slurm_semilag_core_scaling_euler.sh
#
# Important environment overrides:
#   EXE                solver executable path
#   BASE_CONFIG        input JSON config to clone per run
#   THREAD_COUNTS      comma-separated thread counts (default: 1,2,4,8,16)
#   REPEATS            repetitions per thread count (default: 3)
#   SCHEDULE_POLICY     OpenMP schedule value (default: dynamic,64)
#   PROC_BIND_POLICY, PLACES_POLICY, DYNAMIC_POLICY
#   OUTPUT_ROOT_REL    subdir under out/data (default: euler_core_scaling)
#   RUN_LABEL          run label in output paths (default: TaylorGreen_ref3)
#   PRINTLEVEL_OVERRIDE, TOL_OVERRIDE, DT_OVERRIDE, T_OVERRIDE, REFINEMENTS_OVERRIDE
#

# In Slurm batch mode, BASH_SOURCE may point to a spool copy of the script.
# Prefer submission directory (repo root when submitted from there).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT_DIR"

EXE="${EXE:-./build/semilagrangian_navierstokes_nitsche}"
BASE_CONFIG="${BASE_CONFIG:-data/config/TaylorGreenSemiLag/TaylorGreenSemiLag_conv_order1_ref3.json}"

THREAD_COUNTS_CSV="${THREAD_COUNTS:-1,2,4,8,16}"
REPEATS="${REPEATS:-3}"

SCHEDULE_POLICY="${SCHEDULE_POLICY:-${OMP_SCHEDULE_VALUE:-dynamic,64}}"
PROC_BIND_POLICY="${PROC_BIND_POLICY:-${OMP_PROC_BIND_VALUE:-close}}"
PLACES_POLICY="${PLACES_POLICY:-${OMP_PLACES_VALUE:-cores}}"
DYNAMIC_POLICY="${DYNAMIC_POLICY:-${OMP_DYNAMIC_VALUE:-false}}"

OUTPUT_ROOT_REL="${OUTPUT_ROOT_REL:-euler_core_scaling}"
RUN_LABEL="${RUN_LABEL:-TaylorGreen_ref3}"

PRINTLEVEL_OVERRIDE="${PRINTLEVEL_OVERRIDE:-0}"
TOL_OVERRIDE="${TOL_OVERRIDE:-}"
DT_OVERRIDE="${DT_OVERRIDE:-}"
T_OVERRIDE="${T_OVERRIDE:-}"
REFINEMENTS_OVERRIDE="${REFINEMENTS_OVERRIDE:-}"
PROFILE_ADVECTION_BREAKDOWN="${PROFILE_ADVECTION_BREAKDOWN:-false}"
PROFILE_ADVECTION_THREAD_BALANCE="${PROFILE_ADVECTION_THREAD_BALANCE:-false}"

BUILD_BINARY="${BUILD_BINARY:-0}"

if [[ ! -x "$EXE" ]]; then
    echo "[error] Executable not found or not executable: $EXE"
    exit 1
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "[error] Base config not found: $BASE_CONFIG"
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "[error] python3 not found in PATH"
    exit 1
fi

if [[ "$BUILD_BINARY" == "1" ]]; then
    echo "[info] Rebuilding solver binary..."
    cmake --build build --target semilagrangian_navierstokes_nitsche -j"${SLURM_CPUS_PER_TASK:-8}"
fi

max_threads="${SLURM_CPUS_PER_TASK:-$(nproc)}"
declare -A seen_threads=()
declare -a THREAD_COUNTS=()
IFS=',' read -r -a requested_threads <<< "$THREAD_COUNTS_CSV"
for token in "${requested_threads[@]}"; do
    t="${token//[[:space:]]/}"
    [[ -z "$t" ]] && continue
    if ! [[ "$t" =~ ^[0-9]+$ ]]; then
        echo "[warn] Skipping invalid thread token: '$token'"
        continue
    fi
    if (( t < 1 )); then
        echo "[warn] Skipping non-positive thread count: $t"
        continue
    fi
    if (( t > max_threads )); then
        echo "[warn] Skipping thread count $t (exceeds available $max_threads)"
        continue
    fi
    if [[ -z "${seen_threads[$t]:-}" ]]; then
        seen_threads[$t]=1
        THREAD_COUNTS+=("$t")
    fi
done

if (( ${#THREAD_COUNTS[@]} == 0 )); then
    echo "[error] No valid thread counts remain after filtering"
    exit 1
fi

if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || (( REPEATS < 1 )); then
    echo "[error] REPEATS must be a positive integer; got '$REPEATS'"
    exit 1
fi

JOB_TAG="${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="${RUN_LABEL}_${JOB_TAG}"

RUN_DATA_DIR="$ROOT_DIR/out/data/${OUTPUT_ROOT_REL}/${RUN_ID}"
RUN_CFG_DIR="$RUN_DATA_DIR/configs"
mkdir -p "$RUN_CFG_DIR"

MANIFEST_CSV="$RUN_DATA_DIR/manifest.csv"
RAW_CSV="$RUN_DATA_DIR/raw_runs.csv"
SUMMARY_CSV="$RUN_DATA_DIR/summary.csv"

echo "threads,repeat,status,return_code,config_path,csv_path" > "$MANIFEST_CSV"

echo "[info] Root:            $ROOT_DIR"
echo "[info] Executable:      $EXE"
echo "[info] Base config:     $BASE_CONFIG"
echo "[info] Run directory:   $RUN_DATA_DIR"
echo "[info] Thread counts:   ${THREAD_COUNTS[*]}"
echo "[info] Repeats:         $REPEATS"
echo "[info] OMP schedule:    $SCHEDULE_POLICY"
echo "[info] OMP places:      $PLACES_POLICY"
echo "[info] OMP proc bind:   $PROC_BIND_POLICY"

make_config() {
    local out_cfg="$1"
    local output_rel="$2"

    BASE_CONFIG="$BASE_CONFIG" \
    OUT_CFG="$out_cfg" \
    OUTPUT_REL="$output_rel" \
    PRINTLEVEL_OVERRIDE="$PRINTLEVEL_OVERRIDE" \
    TOL_OVERRIDE="$TOL_OVERRIDE" \
    DT_OVERRIDE="$DT_OVERRIDE" \
    T_OVERRIDE="$T_OVERRIDE" \
    REFINEMENTS_OVERRIDE="$REFINEMENTS_OVERRIDE" \
    PROFILE_ADVECTION_BREAKDOWN="$PROFILE_ADVECTION_BREAKDOWN" \
    PROFILE_ADVECTION_THREAD_BALANCE="$PROFILE_ADVECTION_THREAD_BALANCE" \
    python3 - <<'PY'
import json
import os
from pathlib import Path


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


base = Path(os.environ["BASE_CONFIG"])
out_cfg = Path(os.environ["OUT_CFG"])
output_rel = os.environ["OUTPUT_REL"]

cfg = json.loads(base.read_text())
cfg["outputfile"] = output_rel

printlevel_override = os.environ.get("PRINTLEVEL_OVERRIDE", "").strip()
if printlevel_override:
    cfg["printlevel"] = int(printlevel_override)

tol_override = os.environ.get("TOL_OVERRIDE", "").strip()
if tol_override:
    cfg["tol"] = float(tol_override)

dt_override = os.environ.get("DT_OVERRIDE", "").strip()
if dt_override:
    cfg["dt"] = float(dt_override)

t_override = os.environ.get("T_OVERRIDE", "").strip()
if t_override:
    cfg["T"] = float(t_override)

ref_override = os.environ.get("REFINEMENTS_OVERRIDE", "").strip()
if ref_override:
    cfg["refinements"] = int(ref_override)

cfg["profile_advection_breakdown"] = parse_bool(
    os.environ.get("PROFILE_ADVECTION_BREAKDOWN", "false")
)
cfg["profile_advection_thread_balance"] = parse_bool(
    os.environ.get("PROFILE_ADVECTION_THREAD_BALANCE", "false")
)

out_cfg.write_text(json.dumps(cfg, indent=4) + "\n")
PY
}

num_failures=0
for t in "${THREAD_COUNTS[@]}"; do
    for rep in $(seq 1 "$REPEATS"); do
        run_key="t${t}_r${rep}"
        cfg_path="$RUN_CFG_DIR/${run_key}.json"
        output_rel="${OUTPUT_ROOT_REL}/${RUN_ID}/${run_key}"
        csv_path="$ROOT_DIR/out/data/${output_rel}_vars.csv"

        make_config "$cfg_path" "$output_rel"

        echo "[run] threads=$t repeat=$rep"
        set +e
        env \
            -u OMP_SCHEDULE_VALUE \
            -u OMP_PROC_BIND_VALUE \
            -u OMP_PLACES_VALUE \
            -u OMP_DYNAMIC_VALUE \
            OMP_NUM_THREADS="$t" \
            OMP_DYNAMIC="$DYNAMIC_POLICY" \
            OMP_PROC_BIND="$PROC_BIND_POLICY" \
            OMP_PLACES="$PLACES_POLICY" \
            OMP_SCHEDULE="$SCHEDULE_POLICY" \
            "$EXE" -c "$cfg_path"
        rc=$?
        set -e

        if (( rc != 0 )); then
            status="fail"
            num_failures=$((num_failures + 1))
            echo "[warn] run failed (threads=$t repeat=$rep rc=$rc)"
        elif [[ ! -f "$csv_path" ]]; then
            status="missing_csv"
            num_failures=$((num_failures + 1))
            echo "[warn] missing CSV output: $csv_path"
        else
            status="ok"
        fi

        printf "%s,%s,%s,%s,%s,%s\n" \
            "$t" "$rep" "$status" "$rc" "$cfg_path" "$csv_path" >> "$MANIFEST_CSV"
    done
done

python3 - "$MANIFEST_CSV" "$RAW_CSV" "$SUMMARY_CSV" <<'PY'
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


manifest_path = Path(sys.argv[1])
raw_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])


def parse_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def metric_sum(rows, name: str) -> float:
    if not rows or name not in rows[0]:
        return float("nan")
    return sum(parse_float(r.get(name, "nan")) for r in rows)


def metric_avg(rows, name: str) -> float:
    if not rows or name not in rows[0]:
        return float("nan")
    vals = [parse_float(r.get(name, "nan")) for r in rows]
    if not vals:
        return float("nan")
    return sum(vals) / float(len(vals))


manifest = list(csv.DictReader(manifest_path.open()))

raw_rows = []
for rec in manifest:
    row = {
        "threads": rec["threads"],
        "repeat": rec["repeat"],
        "status": rec["status"],
        "return_code": rec["return_code"],
        "config_path": rec["config_path"],
        "csv_path": rec["csv_path"],
        "num_steps": "",
        "sum_runtime_it_s": "",
        "sum_advect_s": "",
        "sum_solve_s": "",
        "avg_num_it": "",
        "final_u1_err_L2": "",
    }

    if rec["status"] == "ok" and Path(rec["csv_path"]).is_file():
        rows = list(csv.DictReader(Path(rec["csv_path"]).open()))
        if rows:
            row["num_steps"] = str(len(rows))
            row["sum_runtime_it_s"] = f"{metric_sum(rows, 'runtime_it'):.9f}"
            row["sum_advect_s"] = f"{metric_sum(rows, 'time_advect_dofs_s'):.9f}"
            row["sum_solve_s"] = f"{metric_sum(rows, 'time_solve_s'):.9f}"
            row["avg_num_it"] = f"{metric_avg(rows, 'num_it'):.9f}"
            if "u1_err_L2" in rows[-1]:
                row["final_u1_err_L2"] = f"{parse_float(rows[-1]['u1_err_L2']):.15g}"

    raw_rows.append(row)

raw_fields = [
    "threads",
    "repeat",
    "status",
    "return_code",
    "num_steps",
    "sum_runtime_it_s",
    "sum_advect_s",
    "sum_solve_s",
    "avg_num_it",
    "final_u1_err_L2",
    "config_path",
    "csv_path",
]
with raw_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=raw_fields)
    w.writeheader()
    w.writerows(raw_rows)


def med(values):
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan")
    return statistics.median(vals)


ok_by_thread = defaultdict(list)
for r in raw_rows:
    if r["status"] != "ok":
        continue
    t = int(r["threads"])
    ok_by_thread[t].append(r)

summary_rows = []
if ok_by_thread:
    base_thread = min(ok_by_thread)
    base_total = med([parse_float(r["sum_runtime_it_s"]) for r in ok_by_thread[base_thread]])
    base_adv = med([parse_float(r["sum_advect_s"]) for r in ok_by_thread[base_thread]])
    base_solve = med([parse_float(r["sum_solve_s"]) for r in ok_by_thread[base_thread]])

    for t in sorted(ok_by_thread):
        rows_t = ok_by_thread[t]
        total = med([parse_float(r["sum_runtime_it_s"]) for r in rows_t])
        adv = med([parse_float(r["sum_advect_s"]) for r in rows_t])
        solve = med([parse_float(r["sum_solve_s"]) for r in rows_t])
        avg_it = med([parse_float(r["avg_num_it"]) for r in rows_t])
        err = med([parse_float(r["final_u1_err_L2"]) for r in rows_t])

        sp_total = base_total / total if total > 0.0 and not math.isnan(total) else float("nan")
        sp_adv = base_adv / adv if adv > 0.0 and not math.isnan(adv) else float("nan")
        sp_solve = base_solve / solve if solve > 0.0 and not math.isnan(solve) else float("nan")

        eff_total = sp_total / float(t) if not math.isnan(sp_total) else float("nan")
        eff_adv = sp_adv / float(t) if not math.isnan(sp_adv) else float("nan")
        eff_solve = sp_solve / float(t) if not math.isnan(sp_solve) else float("nan")

        summary_rows.append(
            {
                "threads": t,
                "successful_runs": len(rows_t),
                "median_sum_runtime_it_s": total,
                "median_sum_advect_s": adv,
                "median_sum_solve_s": solve,
                "median_avg_num_it": avg_it,
                "median_final_u1_err_L2": err,
                "speedup_total_vs_t1": sp_total,
                "speedup_advect_vs_t1": sp_adv,
                "speedup_solve_vs_t1": sp_solve,
                "efficiency_total": eff_total,
                "efficiency_advect": eff_adv,
                "efficiency_solve": eff_solve,
            }
        )

summary_fields = [
    "threads",
    "successful_runs",
    "median_sum_runtime_it_s",
    "median_sum_advect_s",
    "median_sum_solve_s",
    "median_avg_num_it",
    "median_final_u1_err_L2",
    "speedup_total_vs_t1",
    "speedup_advect_vs_t1",
    "speedup_solve_vs_t1",
    "efficiency_total",
    "efficiency_advect",
    "efficiency_solve",
]

with summary_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary_fields)
    w.writeheader()
    for r in summary_rows:
        out = r.copy()
        for key in summary_fields:
            if key in ("threads", "successful_runs"):
                continue
            out[key] = "" if math.isnan(out[key]) else f"{out[key]:.9f}"
        w.writerow(out)

print("[info] Summary table:")
print("threads  ok_runs  median_total_s  median_advect_s  speedup_total  speedup_advect")
for r in summary_rows:
    def f(v):
        return "nan" if math.isnan(v) else f"{v:.3f}"
    print(
        f"{r['threads']:>7}  {r['successful_runs']:>7}"
        f"  {f(r['median_sum_runtime_it_s']):>14}"
        f"  {f(r['median_sum_advect_s']):>15}"
        f"  {f(r['speedup_total_vs_t1']):>13}"
        f"  {f(r['speedup_advect_vs_t1']):>14}"
    )
PY

echo "[info] Wrote: $MANIFEST_CSV"
echo "[info] Wrote: $RAW_CSV"
echo "[info] Wrote: $SUMMARY_CSV"

if (( num_failures > 0 )); then
    echo "[warn] Completed with $num_failures failed runs"
    exit 1
fi

echo "[info] Completed successfully"
