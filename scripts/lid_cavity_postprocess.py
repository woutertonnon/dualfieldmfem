#!/usr/bin/env python3
"""Lid-driven cavity: compare centerline velocity profiles against Ghia,Ghia,Shin.

Reads the final-cycle ParaView VTU written by the solver, samples the velocity on
the two centerlines of the unit square, and overlays the reference data of
Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387-411 (Tables I & II):
  - u_x along the VERTICAL centerline x=0.5   (vs y)
  - u_y along the HORIZONTAL centerline y=0.5 (vs x)

Usage:
    python scripts/lid_cavity_postprocess.py <case_dir_or_vtu> --Re 100 -o out.png

NOTE: the Ghia tables below are the canonical values for Re=100/400/1000; if using
for a thesis, sanity-check a couple of entries against the original paper.
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

# --- Ghia, Ghia & Shin (1982) reference data -------------------------------
# Table I: u along the vertical line through the geometric center (x=0.5).
GHIA_Y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344,
                   0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703,
                   0.0625, 0.0547, 0.0000])
GHIA_U = {
    100:  np.array([1.0, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332,
                    -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434,
                    -0.04775, -0.04192, -0.03717, 0.0]),
    400:  np.array([1.0, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256,
                    0.02135, -0.11477, -0.17119, -0.32726, -0.24299, -0.14612,
                    -0.10338, -0.09266, -0.08186, 0.0]),
    1000: np.array([1.0, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719,
                    0.05702, -0.06080, -0.10648, -0.27805, -0.38289, -0.29730,
                    -0.22220, -0.20196, -0.18109, 0.0]),
}
# Table II: v along the horizontal line through the geometric center (y=0.5).
GHIA_X = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594,
                   0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781,
                   0.0703, 0.0625, 0.0000])
GHIA_V = {
    100:  np.array([0.0, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914,
                    -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077,
                    0.12317, 0.10890, 0.10091, 0.09233, 0.0]),
    400:  np.array([0.0, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827,
                    -0.44993, -0.38598, 0.05186, 0.30174, 0.30203, 0.28124,
                    0.22965, 0.20920, 0.19713, 0.18360, 0.0]),
    1000: np.array([0.0, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550,
                    -0.42665, -0.31966, 0.02526, 0.32235, 0.33075, 0.37095,
                    0.32627, 0.30353, 0.29012, 0.27485, 0.0]),
}
# Known data-entry error in the widely-circulated Ghia,Ghia,Shin (1982) table:
# the Re=400 v-value at x=0.9063 (index 5, tabulated -0.23827) is inconsistent
# with a smooth profile through its neighbors (-0.44993 at 0.8594, -0.22847 at
# 0.9453) and is documented as a typo in the literature.  Excluded from the
# comparison rather than replaced (we don't fabricate benchmark data).
# {Re: [indices into GHIA_X/GHIA_V to drop]}
GHIA_V_BAD = {400: [5]}


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


def sample_line(mesh, p0, p1, n=400):
    line = mesh.sample_over_line(p0, p1, resolution=n - 1)
    u = np.asarray(line.point_data["u"])
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    pts = line.points
    return pts, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--Re", type=int, required=True, choices=[100, 400, 1000])
    ap.add_argument("-o", "--out", default="cavity_centerlines.png")
    args = ap.parse_args()

    vtu = latest_vtu(args.case)
    print(f"[cavity] {vtu}  (Re={args.Re})")
    mesh = pv.read(vtu)

    # Vertical centerline x=0.5: u_x vs y
    ptsV, uV = sample_line(mesh, (0.5, 0.0, 0.0), (0.5, 1.0, 0.0))
    yV, uxV = ptsV[:, 1], uV[:, 0]
    # Horizontal centerline y=0.5: u_y vs x
    ptsH, uH = sample_line(mesh, (0.0, 0.5, 0.0), (1.0, 0.5, 0.0))
    xH, uyH = ptsH[:, 0], uH[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    ax1.plot(uxV, yV, "-", lw=2, label="computed")
    ax1.plot(GHIA_U[args.Re], GHIA_Y, "o", ms=5, mfc="none", label="Ghia et al.")
    ax1.set_xlabel(r"$u_x$"); ax1.set_ylabel(r"$y$")
    ax1.set_title(f"vertical centerline $x=0.5$  (Re={args.Re})")
    ax1.grid(alpha=0.3); ax1.legend()

    bad = np.array(GHIA_V_BAD.get(args.Re, []), dtype=int)
    good = np.setdiff1d(np.arange(len(GHIA_X)), bad)
    ax2.plot(xH, uyH, "-", lw=2, label="computed")
    ax2.plot(GHIA_X[good], GHIA_V[args.Re][good], "o", ms=5, mfc="none",
             label="Ghia et al.")
    if bad.size:
        ax2.plot(GHIA_X[bad], GHIA_V[args.Re][bad], "x", ms=7, color="gray",
                 label="Ghia (known typo, excluded)")
    ax2.set_xlabel(r"$x$"); ax2.set_ylabel(r"$u_y$")
    ax2.set_title(f"horizontal centerline $y=0.5$  (Re={args.Re})")
    ax2.grid(alpha=0.3); ax2.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"[cavity] wrote {args.out}")

    # Quantitative error at the Ghia sample points (interp computed onto them).
    u_at = np.interp(GHIA_Y, yV[np.argsort(yV)], uxV[np.argsort(yV)])
    v_at = np.interp(GHIA_X, xH[np.argsort(xH)], uyH[np.argsort(xH)])
    eu = np.abs(u_at - GHIA_U[args.Re]).max()
    ev = np.abs((v_at - GHIA_V[args.Re])[good]).max()
    print(f"[cavity] max|u-Ghia|={eu:.4f}  max|v-Ghia|={ev:.4f}"
          f"{'  (excluded Ghia typo idx '+str(list(bad))+')' if bad.size else ''}")


if __name__ == "__main__":
    main()
