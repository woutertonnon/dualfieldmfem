#!/usr/bin/env python3
"""Convergence / epsilon-robustness plots for the physical-BC studies.

Each viscosity epsilon is run as a separate benchmark case (its own
out/data/<name>/ directory), so a single h-convergence curve lives in one
directory. This script aggregates all epsilon for a given experiment family and
polynomial order into ONE log-log figure of the final-time L2(Omega) velocity
error versus the (relative) mesh size h = 2^-ref, with a reference O(h^order)
slope. That is exactly the thesis picture: first-/second-order convergence with
the error remaining bounded as epsilon -> 0.

Usage
-----
  # plot one family at one order
  python scripts/plot_physbc_convergence.py --family MMS2D_tangential --order 1

  # auto-discover and plot every (family, order) present under out/data/
  python scripts/plot_physbc_convergence.py

Families correspond to the benchmark prefixes:
  MMS2D_tangential, MMS2D_noslip, EthierSteinman3D, StokesSecond, Womersley

Output: out/plots/physbc/<family>_order<order>_convergence.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, no display needed
import matplotlib.pyplot as plt  # noqa: E402

# Directory name pattern: <family>_eps<tag>_order<order>
CASE_RE = re.compile(r"^(?P<family>.+)_eps(?P<tag>[0-9a-z]+)_order(?P<order>\d+)$")


def eps_from_tag(tag: str) -> float:
    """Inverse of _eps_tag(): '0'->0.0, '1em2'->1e-2, '1ep2'->1e2, '1em10'->1e-10."""
    if tag == "0":
        return 0.0
    m = re.match(r"^([0-9.]+)e([mp])(\d+)$", tag)
    if not m:
        # last resort: let float() try
        try:
            return float(tag)
        except ValueError:
            return math.nan
    mant, sign, exp = m.group(1), m.group(2), m.group(3)
    return float(mant) * (10.0 ** (int(exp) * (1 if sign == "p" else -1)))


def eps_label(eps: float) -> str:
    if eps == 0.0:
        return r"$\epsilon=0$"
    return rf"$\epsilon={eps:g}$"


def read_case_errors(name: str, order: int):
    """Return (refs, errors) sorted by refinement for one case directory."""
    refs, errs = [], []
    pat = re.compile(re.escape(name) + rf"_conv_order{order}_ref(\d+)_vars\.csv$")
    for f in glob.glob(f"out/data/{name}/{name}_conv_order{order}_ref*_vars.csv"):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        col = next((c for c in rows[-1] if "err" in c.lower()), None)
        if col is None:
            continue
        try:
            err = float(rows[-1][col])
        except (TypeError, ValueError):
            continue
        refs.append(int(m.group(1)))
        errs.append(err)
    order_idx = sorted(range(len(refs)), key=lambda i: refs[i])
    return [refs[i] for i in order_idx], [errs[i] for i in order_idx]


def discover_cases():
    """Map (family, order) -> {eps: case_name} from out/data/ directories."""
    families = defaultdict(dict)
    for d in sorted(glob.glob("out/data/*")):
        if not os.path.isdir(d):
            continue
        m = CASE_RE.match(os.path.basename(d))
        if not m:
            continue
        fam = m.group("family")
        order = int(m.group("order"))
        eps = eps_from_tag(m.group("tag"))
        families[(fam, order)][eps] = os.path.basename(d)
    return families


# Per-order line style (marker + linestyle); epsilon is encoded by colour.
ORDER_STYLE = {1: ("o", "-"), 2: ("s", "--"), 3: ("^", ":"), 4: ("v", "-.")}


def plot_family(family: str, order_map: dict, outdir: str):
    """One log-log error-vs-h figure for a family.

    `order_map` maps polynomial order -> {epsilon: case_name}. All available
    orders are overlaid on the same axes: colour encodes epsilon (shared across
    orders) and marker/linestyle encodes the order, so the figure shows both the
    epsilon-robustness and the designed first-/second-order convergence. An
    O(h^p) reference line is drawn for every order p present.
    """
    fig, ax = plt.subplots(figsize=(6.8, 5.0))

    # Shared colour per epsilon (largest epsilon first for a stable ordering).
    all_eps = sorted({e for m in order_map.values() for e in m}, reverse=True)
    cmap = plt.get_cmap("viridis", max(len(all_eps), 2))
    eps_color = {e: cmap(i) for i, e in enumerate(all_eps)}

    plotted = 0
    order_anchor = {}  # order -> (h, err) at the finest level (largest eps)
    for order in sorted(order_map):
        marker, ls = ORDER_STYLE.get(order, ("o", "-"))
        for eps in sorted(order_map[order], reverse=True):
            refs, errs = read_case_errors(order_map[order][eps], order)
            if not refs:
                continue
            hs = [2.0 ** (-r) for r in refs]  # relative mesh size
            ax.loglog(hs, errs, marker=marker, linestyle=ls, color=eps_color[eps],
                      linewidth=1.5, markersize=6)
            plotted += 1
            if order not in order_anchor:
                order_anchor[order] = (hs[-1], errs[-1])

    if plotted == 0:
        plt.close(fig)
        print(f"[skip] {family}: no data")
        return None

    # O(h^p) reference line for each order present, anchored at its finest point.
    for order, (h1, y1) in order_anchor.items():
        h0 = h1 * 2.0
        y0 = y1 * (2.0 ** order)
        ax.loglog([h0, h1], [y0, y1], "k--", linewidth=1.0)
        ax.text(math.sqrt(h0 * h1), math.sqrt(y0 * y1) * 1.18,
                rf"$O(h^{{{order}}})$", ha="center", va="bottom", fontsize=9)

    # Two-part legend: epsilon (colour) and order (marker/linestyle).
    from matplotlib.lines import Line2D
    eps_handles = [Line2D([0], [0], color=eps_color[e], marker="o", linestyle="-",
                          label=eps_label(e)) for e in all_eps]
    order_handles = [Line2D([0], [0], color="0.3", marker=ORDER_STYLE.get(o, ("o", "-"))[0],
                            linestyle=ORDER_STYLE.get(o, ("o", "-"))[1],
                            label=f"order {o}") for o in sorted(order_map)]
    leg1 = ax.legend(handles=eps_handles, loc="upper right", fontsize=9, title="viscosity")
    ax.add_artist(leg1)
    ax.legend(handles=order_handles, loc="lower left", fontsize=9, title="scheme")

    ax.set_xlabel(r"relative mesh size $h \;(=2^{-\mathrm{ref}})$")
    ax.set_ylabel(r"$\|u-u_h\|_{L^2(\Omega)}$ at final time")
    ax.set_title(family)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.invert_xaxis()  # coarse -> fine left to right
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    orders_tag = "".join(str(o) for o in sorted(order_map))
    out = os.path.join(outdir, f"{family}_order{orders_tag}_convergence.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[ok]  wrote {out}  ({plotted} curve(s), orders {sorted(order_map)})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default=None,
                    help="experiment prefix, e.g. MMS2D_tangential (default: all discovered)")
    ap.add_argument("--order", type=int, default=None,
                    help="polynomial order (default: all discovered)")
    ap.add_argument("--outdir", default="out/plots/physbc")
    args = ap.parse_args()

    cases = discover_cases()
    if not cases:
        print("[error] no physical-BC case directories found under out/data/ "
              "(expected <family>_eps<tag>_order<n>/).")
        return

    # Regroup discovered (family, order) cases into family -> {order: {eps: name}},
    # honouring the optional --family / --order filters, so each family is drawn as
    # a single figure overlaying all its orders.
    families = defaultdict(dict)
    for (fam, order), eps_to_case in cases.items():
        if args.family is not None and fam != args.family:
            continue
        if args.order is not None and order != args.order:
            continue
        families[fam][order] = eps_to_case

    if not families:
        print(f"[error] no cases match family={args.family} order={args.order}.")
        print("        available:", ", ".join(f"{f}/order{o}" for f, o in sorted(cases)))
        return

    for fam in sorted(families):
        plot_family(fam, families[fam], args.outdir)


if __name__ == "__main__":
    main()
