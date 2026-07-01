#!/usr/bin/env python3
"""Lid-driven cavity: 2D field plot with contour lines (vortex structure).

Reads the final-cycle VTU, samples the velocity onto a uniform grid, and renders:
  - left : |u| filled contours + streamlines (coloured by speed)
  - right: streamfunction psi contour LINES -- the canonical cavity figure, whose
           closed contours mark the primary vortex and the secondary corner
           vortices.  psi is obtained by integrating u in y (psi = int_0^y u dy',
           psi=0 on the bottom wall), which is the streamfunction for a
           divergence-free field.

Usage:
    python scripts/plot_cavity_field.py <case_dir_or_vtu> [-o out.png] [-N 240]
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
    ap.add_argument("-o", "--out", default="cavity_field.png")
    ap.add_argument("-N", type=int, default=240, help="sampling grid resolution")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    vtu = latest_vtu(args.case)
    print(f"[cavity-field] {vtu}")
    mesh = pv.read(vtu)

    N = args.N
    grid = pv.ImageData(dimensions=(N, N, 1),
                        spacing=(1.0 / (N - 1), 1.0 / (N - 1), 1.0),
                        origin=(0.0, 0.0, 0.0))
    s = grid.sample(mesh)
    pd = s.point_data
    u = np.asarray(pd["u"])
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if "vtkValidPointMask" in pd:
        valid = np.asarray(pd["vtkValidPointMask"]).astype(bool)
        u[~valid] = 0.0

    U = u[:, 0].reshape(N, N)   # [iy, ix]
    V = u[:, 1].reshape(N, N)
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    speed = np.sqrt(U ** 2 + V ** 2)

    # Streamfunction: psi = int_0^y u dy'  (psi=0 on the bottom wall).
    dy = 1.0 / (N - 1)
    psi = np.zeros_like(U)
    psi[1:, :] = np.cumsum(0.5 * (U[1:, :] + U[:-1, :]) * dy, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))

    cf = ax1.contourf(x, y, speed, levels=40, cmap="viridis")
    fig.colorbar(cf, ax=ax1, shrink=0.9, label=r"$|\mathbf{u}|$")
    ax1.streamplot(x, y, U, V, color="w", density=1.4, linewidth=0.6, arrowsize=0.6)
    ax1.set_title("velocity magnitude + streamlines")

    # psi contour lines: dense small levels near 0 expose the corner vortices,
    # which have psi of opposite sign and tiny magnitude vs the primary vortex.
    pmin, pmax = psi.min(), psi.max()
    main_levels = np.linspace(pmin, pmax, 25)
    corner_levels = np.array([-1e-2, -3e-3, -1e-3, -3e-4, -1e-4,
                              1e-4, 3e-4, 1e-3, 3e-3, 1e-2]) * max(abs(pmin), pmax)
    levels = np.unique(np.concatenate([main_levels, corner_levels]))
    ax2.contour(x, y, psi, levels=levels, colors="k", linewidths=0.5)
    ax2.set_title(r"streamfunction $\psi$ (contour lines)")

    for ax in (ax1, ax2):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    if args.title:
        fig.suptitle(args.title)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"[cavity-field] wrote {args.out}")


if __name__ == "__main__":
    main()
