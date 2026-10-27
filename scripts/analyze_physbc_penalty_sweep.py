"""Summarize the Nitsche penalty sweep on under-resolved boundary layers.

Reads the runs registered by _build_physbc_penalty_map() in
semilagrangian_benchmarks.py (out/data/<case>_eps<e>_order<o>_Cw<P>/) and the
SLURM logs, and writes to out/plots/physbc_penalty/:

  summary.csv              one row per (case, order, eps, level, penalty)
  penalty_table.tex        relative final-time L2 errors, thesis table rows
  <case>_error_vs_Cw.{pdf,png}       relative error vs penalty, orders 1 and 2
  <case>_iterations_vs_Cw.{pdf,png}  mean FGMRES iterations per time step

The error is reported relative to the L2(Omega) norm of the exact velocity at
the final time T = 1 (one forcing period). A run counts as failed when its CSV
is missing, does not reach T, or contains a non-finite error.

Usage (from the repository root on Euler; gmsh needs libGLU, see
scripts/EULER_PHYSBC_README.md):
    export LD_LIBRARY_PATH="$GLU_LIB:$LD_LIBRARY_PATH" VENV=~/physbc-venv
    PHYSBC_MAP=penalty bash scripts/submit_physbc_euler.sh      # 84 jobs
    python scripts/analyze_physbc_penalty_sweep.py [--data out/data] [--logs logs]
"""

import argparse
import glob
import math
import os
import re

import numpy as np
import pandas as pd

OMEGA = 2.0 * math.pi
L, H = 3.0, 1.0
T_FINAL = 2.0 * math.pi / OMEGA
H0 = H / 16.0
DEFAULT_PENALTY = {1: 4e4, 2: 800.0 / 3.0}
CASES = {"StokesSecond": "Stokes' second problem", "Womersley": "Womersley flow"}

# Categorical slots 1-3 of the reference palette (validated, light surface);
# one hue per viscosity, one marker per refinement level.
EPS_COLORS = {5e-4: "#2a78d6", 1e-4: "#eb6834", 1e-5: "#1baf7a"}
LEVEL_MARKERS = ["o", "s", "^", "D"]
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"

RUN_RE = re.compile(r"^(StokesSecond|Womersley)_eps(\w+?)_order([12])_Cw([0-9.e+]+)$")


def eps_from_tag(tag):
    """Inverse of _eps_tag: '5em4' -> 5e-4."""
    return float(tag.replace("em", "e-").replace("ep", "e+"))


def delta(eps):
    return math.sqrt(2.0 * eps / OMEGA)


def exact_norm(case, eps, t=T_FINAL, n=400001):
    """L2((0,L)x(0,H)) norm of the exact x-velocity at time t."""
    y = np.linspace(0.0, H, n)
    if case == "StokesSecond":
        d = delta(eps)
        u = np.exp(-y / d) * np.cos(OMEGA * t - y / d)
    else:
        # Womersley, A = 1: u = Re[(1/(i w)) (1 - cosh(k z)/cosh(k H/2)) e^{iwt}],
        # z = y - H/2, k = sqrt(i w / eps); ratio written with decaying exponentials.
        k = np.sqrt(1j * OMEGA / eps)
        z = y - H / 2.0
        ratio = (np.exp(k * (np.abs(z) - H / 2.0)) + np.exp(-k * (np.abs(z) + H / 2.0))) \
            / (1.0 + np.exp(-k * H))
        u = np.real((1.0 - ratio) * np.exp(1j * OMEGA * t) / (1j * OMEGA))
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return math.sqrt(L * trapezoid(u * u, y))


def parse_logs(log_dir):
    """Map run name -> list of (penalty, bound) lines logged by the apps."""
    info = {}
    for f in glob.glob(os.path.join(log_dir, "physbc_conv_*.log")):
        with open(f, errors="replace") as fh:
            text = fh.read()
        m = re.search(r"BENCH=(\S+)", text)
        if not m:
            continue
        hits = re.findall(r"Nitsche: theta=(\S+), penalty=(\S+), coercivity bound \(theta=1\)=(\S+)",
                          text)
        if hits:
            info[m.group(1)] = [(float(p), float(b)) for _, p, b in hits]
    return info


def collect(data_dir, log_dir):
    logs = parse_logs(log_dir)
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(data_dir, "*_Cw*"))):
        name = os.path.basename(run_dir)
        m = RUN_RE.match(name)
        if not m:
            continue
        case, eps_tag, order, ptag = m.groups()
        eps, order, penalty = eps_from_tag(eps_tag), int(order), float(ptag)
        logged = logs.get(name, [])
        for f in sorted(glob.glob(os.path.join(run_dir, f"{name}_conv_order{order}_ref*_vars.csv"))):
            level = int(re.search(r"_ref(\d+)_", f).group(1))
            row = dict(case=case, order=order, eps=eps, level=level, penalty=penalty,
                       h_over_delta=H0 / 2 ** level / delta(eps),
                       solved_penalty=logged[0][0] if logged else float("nan"),
                       bound=logged[0][1] if logged else float("nan"))
            try:
                df = pd.read_csv(f)
            except Exception:
                df = pd.DataFrame()
            if df.empty or "u1_err_L2" not in df:
                row.update(status="failed", rel_err=float("nan"), max_rel_err=float("nan"),
                           mean_its=float("nan"), t_stop=float("nan"))
            else:
                # A run that stops before T with data has blown up: the solver
                # aborts once FGMRES meets a non-finite residual.
                norm = exact_norm(case, eps)
                err = df["u1_err_L2"].to_numpy(dtype=float)
                t_stop = float(df["time_full"].iloc[-1])
                ok = abs(t_stop - T_FINAL) < 1e-8 and np.all(np.isfinite(err))
                row.update(status="ok" if ok else "blew up",
                           rel_err=err[-1] / norm if ok else float("nan"),
                           max_rel_err=np.nanmax(err) / norm,
                           mean_its=float(df["num_it"].mean()), t_stop=t_stop)
            rows.append(row)
    return pd.DataFrame(rows)


def sci_tex(x):
    """5e-4 -> '5\\cdot 10^{-4}', 1e-5 -> '10^{-5}'."""
    mant, expo = f"{x:.0e}".split("e")
    power = f"10^{{{int(expo)}}}"
    return power if mant == "1" else f"{mant}\\cdot {power}"


def fmt_pct(x):
    if not math.isfinite(x):
        return "--"
    p = 100.0 * x
    return f"{p:.2g}" if p < 10 else f"{p:.0f}"


def latex_table(df, path):
    penalties = sorted(df["penalty"].unique())
    head = " & ".join(f"${p:g}$" for p in penalties)
    lines = [f"% generated by scripts/analyze_physbc_penalty_sweep.py",
             f"\\begin{{tabular}}{{lll{'r' * len(penalties)}}}",
             "\\hline",
             f"order & $\\epsilon$ & $h/\\delta$ & {head} \\\\ \\hline"]
    for case, title in CASES.items():
        sub = df[df["case"] == case]
        if sub.empty:
            continue
        lines.append(f"\\multicolumn{{{3 + len(penalties)}}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for (order, eps, level), g in sub.groupby(["order", "eps", "level"]):
            g = g.set_index("penalty")
            cells = ["" if p not in g.index
                     else "$\\times$" if g.loc[p, "status"] == "blew up"
                     else fmt_pct(g.loc[p, "rel_err"]) for p in penalties]
            lines.append(f"{order} & ${sci_tex(eps)}$ & ${g['h_over_delta'].iloc[0]:.1f}$ & "
                         + " & ".join(cells) + " \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}\\\\[2pt]")
    lines.append("{\\footnotesize Relative final-time $L^2(\\Omega)$ velocity error in \\%; "
                 "$\\times$: the run blew up before $T$.}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def style_axes(ax):
    ax.set_xscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)


def plot_case(df, case, column, ylabel, logy, path_stem):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sub = df[(df["case"] == case) & (df["status"] == "ok")]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    handles = {}
    for ax, order in zip(axes, (1, 2)):
        style_axes(ax)
        if logy:
            ax.set_yscale("log")
        g_order = sub[sub["order"] == order]
        bound = df[(df["case"] == case) & (df["order"] == order)]["bound"].max()
        if math.isfinite(bound):
            band = ax.axvspan(1.0, bound, color=GRID, alpha=0.6, linewidth=0)
            handles["$C_w$ below the coercivity bound for $\\theta=1$"] = band
        handles["thesis default $C_w$"] = ax.axvline(
            DEFAULT_PENALTY[order], color=INK_MUTED, linestyle="--", linewidth=1.0)
        for (eps, level), g in g_order.sort_values("eps", ascending=False).groupby(
                ["eps", "level"], sort=False):
            g = g.sort_values("penalty")
            label = f"$\\epsilon={sci_tex(eps)}$, $h/\\delta={g['h_over_delta'].iloc[0]:.1f}$"
            (line,) = ax.plot(g["penalty"], g[column], color=EPS_COLORS.get(eps, INK),
                              marker=LEVEL_MARKERS[level % len(LEVEL_MARKERS)], markersize=6,
                              linewidth=2.0, markeredgecolor="#fcfcfb", markeredgewidth=1.0,
                              label=label)
            handles[label] = line
        # Blown-up runs: a cross at the top edge, in the configuration's hue.
        blew = df[(df["case"] == case) & (df["order"] == order) & (df["status"] == "blew up")]
        for _, r in blew.iterrows():
            ax.plot(r["penalty"], 0.97, transform=ax.get_xaxis_transform(), marker="x",
                    markersize=7, markeredgewidth=2.0, linestyle="none",
                    color=EPS_COLORS.get(r["eps"], INK), clip_on=False)
            handles["blew up before $T$"] = Line2D([], [], marker="x", markersize=7,
                                                   markeredgewidth=2.0, linestyle="none", color=INK)
        lo = df[df["case"] == case]["penalty"].min()
        ax.set_xlim(lo / 1.5, df[df["case"] == case]["penalty"].max() * 1.5)
        ax.set_title(f"order {order}", fontsize=10, color=INK)
        ax.set_xlabel("Nitsche penalty $C_w$", color=INK)
    axes[0].set_ylabel(ylabel, color=INK)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(CASES[case], fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="out/data")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="out/plots/physbc_penalty")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    df = collect(args.data, args.logs)
    if df.empty:
        raise SystemExit(f"no penalty-sweep runs found under {args.data}")
    os.makedirs(args.out, exist_ok=True)
    df = df.sort_values(["case", "order", "eps", "level", "penalty"], ascending=[True, True, False, True, True])
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    latex_table(df, os.path.join(args.out, "penalty_table.tex"))
    for case in CASES:
        plot_case(df, case, "rel_err", "relative $L^2$ error at $T$", True,
                  os.path.join(args.out, f"{case}_error_vs_Cw"))
        plot_case(df, case, "mean_its", "mean FGMRES iterations per step", False,
                  os.path.join(args.out, f"{case}_iterations_vs_Cw"))

    failed = df[df["status"] != "ok"]
    print(f"{len(df)} runs, {len(failed)} not completed")
    for _, r in failed.iterrows():
        print(f"  {r['status'].upper()} {r['case']} order{r['order']} eps={r['eps']:g} "
              f"ref{r['level']} Cw={r['penalty']:g} t_stop={r['t_stop']:g}")
    mism = df[np.isfinite(df["solved_penalty"]) &
              (abs(df["solved_penalty"] / df["penalty"] - 1) > 1e-6)]
    if not mism.empty:
        print("  WARNING: solved penalty differs from the requested one in", len(mism), "runs")
    print(f"written to {args.out}/")


if __name__ == "__main__":
    main()
