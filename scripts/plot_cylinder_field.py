#!/usr/bin/env python3
"""Plot the flow-around-cylinder solution for visual validation.

Reads the final-cycle ParaView VTU written by the order-2 solver and renders
velocity-magnitude filled contours (with streamlines) over the channel, plus a
pressure panel.  A correct steady Re=20 solution shows: parabolic inflow, flow
smoothly accelerating around the cylinder, a short steady recirculation behind
it, and flow LEAVING the outlet (confirming the do-nothing outflow works).

Usage:
    python scripts/plot_cylinder_field.py <case_dir_or_pvd> [-o out.png]
where <case_dir> is out/paraview/<outputfile> (the latest Cycle* is used).
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyvista as pv


def latest_vtu(path):
    if path.endswith(".vtu"):
        return path
    cycles = sorted(glob.glob(os.path.join(path, "Cycle*")))
    if not cycles:
        # maybe path is the parent; try one level down
        cycles = sorted(glob.glob(os.path.join(path, "*", "Cycle*")))
    if not cycles:
        sys.exit(f"no Cycle* under {path}")
    vtus = sorted(glob.glob(os.path.join(cycles[-1], "*.vtu")))
    if not vtus:
        sys.exit(f"no .vtu in {cycles[-1]}")
    return vtus[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("-o", "--out", default="cylinder_field.png")
    ap.add_argument("--cx", type=float, default=0.2)
    ap.add_argument("--cy", type=float, default=0.2)
    ap.add_argument("--r", type=float, default=0.05)
    args = ap.parse_args()

    vtu = latest_vtu(args.case)
    print(f"[plot] {vtu}")
    mesh = pv.read(vtu)
    pts = mesh.points[:, :2]
    x, y = pts[:, 0], pts[:, 1]

    u = np.asarray(mesh.point_data["u"])
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    umag = np.linalg.norm(u[:, :2], axis=1)
    p = np.asarray(mesh.point_data.get("p", np.zeros_like(x))).reshape(-1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.2))

    for ax, field, label, cmap in (
        (ax1, umag, r"$|\mathbf{u}|$", "viridis"),
        (ax2, p, r"$p$", "coolwarm"),
    ):
        tcf = ax.tricontourf(x, y, field, levels=40, cmap=cmap)
        fig.colorbar(tcf, ax=ax, shrink=0.9, label=label)
        ax.add_patch(plt.Circle((args.cx, args.cy), args.r,
                                 color="white", ec="k", zorder=5))
        ax.set_xlim(0, 2.2); ax.set_ylim(0, 0.41)
        ax.set_aspect("equal"); ax.set_ylabel("y")
    ax2.set_xlabel("x")
    ax1.set_title(f"velocity magnitude  ({os.path.basename(vtu)})")
    ax2.set_title("pressure")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
