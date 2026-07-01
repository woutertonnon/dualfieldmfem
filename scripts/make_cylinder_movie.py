#!/usr/bin/env python3
"""Render an mp4 of the flow-around-cylinder solution from saved ParaView cycles.

Reuses the already-written time-series (out/paraview/<case>/Cycle*/proc*.vtu) --
NO re-simulation needed.  Renders velocity-magnitude (+ pressure) filled contours
with a FIXED colour scale across all frames (so the animation doesn't flicker),
annotates the physical time, and encodes with ffmpeg.

Usage:
    python scripts/make_cylinder_movie.py <case_dir> -o movie.mp4 \
        --dt 3.125e-4 --fps 25 [--stride 1] [--fields u p]
where <case_dir> = out/paraview/<outputfile> (contains the Cycle* dirs).
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyvista as pv


def cycle_vtus(case):
    cycles = sorted(glob.glob(os.path.join(case, "Cycle*")))
    if not cycles:
        cycles = sorted(glob.glob(os.path.join(case, "*", "Cycle*")))
    out = []
    for c in cycles:
        vtus = sorted(glob.glob(os.path.join(c, "*.vtu")))
        if vtus:
            n = int(re.search(r"Cycle0*(\d+)", os.path.basename(c)).group(1))
            out.append((n, vtus[0]))
    if not out:
        sys.exit(f"no Cycle*/*.vtu under {case}")
    return out


def read_fields(vtu):
    mesh = pv.read(vtu)
    pts = mesh.points[:, :2]
    u = np.asarray(mesh.point_data["u"])
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    umag = np.linalg.norm(u[:, :2], axis=1)
    p = np.asarray(mesh.point_data.get("p", np.zeros(len(pts)))).reshape(-1)
    return pts[:, 0], pts[:, 1], umag, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("-o", "--out", default="cylinder_movie.mp4")
    ap.add_argument("--dt", type=float, default=None, help="timestep, for time labels")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth cycle")
    ap.add_argument("--fields", nargs="+", default=["u", "p"], choices=["u", "p"])
    ap.add_argument("--cx", type=float, default=0.2)
    ap.add_argument("--cy", type=float, default=0.2)
    ap.add_argument("--r", type=float, default=0.05)
    args = ap.parse_args()

    frames = cycle_vtus(args.case)[:: args.stride]
    print(f"[movie] {len(frames)} frames from {args.case}")

    # Pre-scan a subset for robust, FIXED colour ranges (avoid per-frame flicker).
    scan = frames[:: max(1, len(frames) // 15)]
    umax, pabs = 0.0, 0.0
    for _, vtu in scan:
        _, _, umag, p = read_fields(vtu)
        umax = max(umax, np.percentile(umag, 99.8))
        pabs = max(pabs, np.percentile(np.abs(p), 99.0))
    lv_u = np.linspace(0.0, umax, 41)
    lv_p = np.linspace(-pabs, pabs, 41)
    print(f"[movie] fixed ranges: |u| in [0,{umax:.3f}], p in [+-{pabs:.1f}]")

    panels = [("u", r"$|\mathbf{u}|$", "viridis", lv_u, "max"),
              ("p", r"$p$", "coolwarm", lv_p, "both")]
    panels = [q for q in panels if q[0] in args.fields]

    tmp = tempfile.mkdtemp(prefix="cylframes_")
    try:
        for i, (n, vtu) in enumerate(frames):
            x, y, umag, p = read_fields(vtu)
            data = {"u": umag, "p": p}
            fig, axes = plt.subplots(len(panels), 1,
                                     figsize=(11, 2.7 * len(panels)), squeeze=False)
            for ax, (key, label, cmap, lv, ext) in zip(axes[:, 0], panels):
                tcf = ax.tricontourf(x, y, data[key], levels=lv, cmap=cmap, extend=ext)
                fig.colorbar(tcf, ax=ax, shrink=0.9, label=label)
                ax.add_patch(plt.Circle((args.cx, args.cy), args.r,
                                        color="white", ec="k", zorder=5))
                ax.set_xlim(0, 2.2); ax.set_ylim(0, 0.41)
                ax.set_aspect("equal"); ax.set_ylabel("y")
            axes[-1, 0].set_xlabel("x")
            ttl = f"cycle {n}"
            if args.dt is not None:
                ttl += f"   t = {n * args.dt:.3f}"
            axes[0, 0].set_title(ttl)
            fig.tight_layout()
            fig.savefig(os.path.join(tmp, f"frame_{i:05d}.png"), dpi=120)
            plt.close(fig)
            if i % 25 == 0:
                print(f"[movie] frame {i}/{len(frames)}")

        cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
               "-i", os.path.join(tmp, "frame_%05d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.out]
        print("[movie]", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"[movie] wrote {args.out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
