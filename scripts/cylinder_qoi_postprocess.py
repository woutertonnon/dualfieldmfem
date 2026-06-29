#!/usr/bin/env python3
"""Post-process the flow-around-cylinder QoI CSV (cycle,t,cD,cL,dp,FD,FL).

Steady (Re=20): reports the final c_D, c_L, Delta p.
Unsteady (Re=100): reports max c_D, max c_L (over the last window) and the
Strouhal number St = D f / Ubar from the dominant frequency of c_L(t).

Usage:
    python scripts/cylinder_qoi_postprocess.py <qoi.csv> [--D 0.1] [--Ubar 1.0]
"""
import argparse
import csv
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--D", type=float, default=0.1)
    ap.add_argument("--Ubar", type=float, default=1.0)
    ap.add_argument("--tail", type=float, default=0.5,
                    help="fraction of the series (from the end) to analyse")
    args = ap.parse_args()

    t, cD, cL, dp = [], [], [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            t.append(float(row["t"])); cD.append(float(row["cD"]))
            cL.append(float(row["cL"])); dp.append(float(row["dp"]))
    t = np.array(t); cD = np.array(cD); cL = np.array(cL); dp = np.array(dp)
    if len(t) < 4:
        print("too few samples"); sys.exit(1)

    n0 = int((1.0 - args.tail) * len(t))
    tw, cDw, cLw, dpw = t[n0:], cD[n0:], cL[n0:], dp[n0:]

    amp = cLw.max() - cLw.min()
    print(f"samples={len(t)}  t in [{t[0]:.3f},{t[-1]:.3f}]")
    print(f"final:   cD={cD[-1]:.4f}  cL={cL[-1]:.5f}  dp={dp[-1]:.4f}")
    print(f"window:  cD_max={cDw.max():.4f}  cL_max={cLw.max():.4f}  "
          f"cL_amp={amp:.4f}  dp_mean={dpw.mean():.4f}")

    if amp < 1e-3:
        print("=> steady (no shedding); use the 'final' values.")
        return

    # Strouhal from the dominant c_L frequency (uniform dt assumed).
    dt = np.median(np.diff(tw))
    sig = cLw - cLw.mean()
    freqs = np.fft.rfftfreq(len(sig), d=dt)
    spec = np.abs(np.fft.rfft(sig))
    spec[0] = 0.0
    f_dom = freqs[np.argmax(spec)]
    St = args.D * f_dom / args.Ubar
    print(f"=> unsteady: shedding f={f_dom:.4f} Hz  St = D f/Ubar = {St:.4f}")


if __name__ == "__main__":
    main()
