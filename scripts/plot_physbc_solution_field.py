#!/usr/bin/env python3
"""Render the computed velocity field of a boundary-layer case at ONE fixed mesh
for several viscosities, to show the layer sharpens cleanly (no instability).

For Stokes' second problem and Womersley flow the exact solution is
x-independent, so the streamwise velocity u_x should appear as clean horizontal
bands decaying/oscillating in y. Spurious x-variation or wiggles would signal an
instability; a smooth, ever-thinner layer as epsilon -> 0 is the healthy result.

For each viscosity the case is run at the requested refinement level with
ParaView output enabled; the final-cycle VTU is read and u_x plotted in one
panel of a shared-colourscale figure.

Usage:
  python scripts/plot_physbc_solution_field.py --family StokesSecond --order 1 --level 2
  python scripts/plot_physbc_solution_field.py --family Womersley   --order 1 --level 2
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402

sys.path.insert(0, "scripts")
import semilagrangian_benchmarks as slb  # noqa: E402

EXE = {1: "./build/semilagrangian_navierstokes_nitsche",
       2: "./build/semilagrangian_navierstokes_nitsche_order2"}
SOLVER = {1: "GMRES", 2: "MINRES"}
FAMILY = {"StokesSecond": slb.StokesSecondProblemSemiLag,
          "Womersley": slb.WomersleySemiLag}


def run_case(family, eps, order, level):
    """Run one case at `level` with ParaView output; return (name, final_vtu)."""
    cls = FAMILY[family]
    b = cls(executable=EXE[order], solver=SOLVER[order], nu=eps, order=order, refines=level + 1)
    # Write ParaView output only at the FINAL cycle (vis cadence = #steps), not
    # every step: every-step output of the wide order-2 meshes can fill the disk.
    nsteps = max(1, round(float(b.T) / float(b.dts(order, level))))
    b.SimulationHelper.visualisation = nsteps
    b.refinements = lambda o: [level]               # only this level
    b.SimulationHelper.generate_config_files(b.T, b.dts, b.refinements, b.orders, tol=1e-8)
    name = b.name
    cfg = f"data/config/{name}/{name}_conv_order{order}_ref{level}.json"
    os.makedirs(f"out/data/{name}", exist_ok=True)
    print(f"[run] {name} ref{level} (delta={math.sqrt(2*eps/(2*math.pi)):.4f})")
    subprocess.run([EXE[order], "-c", cfg], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stem = f"{name}_conv_order{order}_ref{level}"
    cycles = sorted(glob.glob(f"out/paraview/{name}/{stem}/Cycle*"))
    if not cycles:
        raise RuntimeError(f"no ParaView output for {name}")
    vtu = sorted(glob.glob(f"{cycles[-1]}/*.vtu"))[0]
    return name, vtu


def read_field(vtu):
    """Return (x, y, u_x) point arrays from a velocity VTU."""
    m = pv.read(vtu)
    key = "u1" if "u1" in m.point_data else ("u" if "u" in m.point_data else None)
    if key is None:
        raise RuntimeError(f"no velocity field in {vtu}: {list(m.point_data.keys())}")
    u = np.asarray(m.point_data[key])
    pts = np.asarray(m.points)
    return pts[:, 0], pts[:, 1], u[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=list(FAMILY), default="StokesSecond")
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-2, 5e-3, 2e-3])
    ap.add_argument("--ymax", type=float, default=None,
                    help="crop the y-axis to [0, ymax] to zoom into the layer")
    args = ap.parse_args()

    omega = 2.0 * math.pi
    panels = []
    for eps in args.eps:
        name, vtu = run_case(args.family, eps, args.order, args.level)
        x, y, ux = read_field(vtu)
        delta = math.sqrt(2 * eps / omega)
        panels.append((eps, delta, x, y, ux))

    vmax = max(np.abs(p[4]).max() for p in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.6),
                             sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (eps, delta, x, y, ux) in zip(axes, panels):
        tri = mtri.Triangulation(x, y)
        tcf = ax.tricontourf(tri, ux, levels=41, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.axhline(delta, color="k", ls=":", lw=1.0)
        ax.text(0.02, delta, r" $\delta$", va="bottom", ha="left", fontsize=9)
        ax.set_title(rf"$\epsilon={eps:g}$  ($\delta={delta:.3f}$)", fontsize=10)
        ax.set_xlabel(r"$x_1$")
        ax.set_aspect("equal")
        if args.ymax:
            ax.set_ylim(0, args.ymax)
    axes[0].set_ylabel(r"$x_2$")
    cbar = fig.colorbar(tcf, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(r"$u_{x_1}$ at final time")
    fig.suptitle(f"{args.family} — velocity field, order {args.order}, "
                 f"same mesh (ref{args.level})", fontsize=12)

    os.makedirs("out/plots/physbc", exist_ok=True)
    out = f"out/plots/physbc/{args.family}_order{args.order}_ref{args.level}_field.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
