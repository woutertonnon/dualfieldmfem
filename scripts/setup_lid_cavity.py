#!/usr/bin/env python3
"""Set up the 2D lid-driven cavity benchmark (thesis sec:nsphys:cavity).

Generates unstructured TRIANGULAR meshes of the unit square (the order-2 ND
small-edge space + SL/DEC require simplices; Cartesian/quad meshes do NOT work)
and the solver config JSONs for Re in {100,400,1000}, for both the SINGULAR lid
u=(1,0) and the REGULARIZED lid u=(16 x^2 (1-x)^2, 0) (vanishes at the corners,
so it avoids the corner singularities and gives clean convergence rates).

Boundary treatment: the operator applies the Nitsche penalty on ALL boundary
faces (StokesOperators.h:415, no marker), while only the lid (attr 14) carries a
nonzero boundary_data_u.  The other three walls therefore get homogeneous
(no-slip) weak Dirichlet automatically -- only [14] needs to be marked.

Mesh boundary attributes (from generate_rectangle_mesh): 11=left, 12=right,
13=bottom, 14=top(lid).

Usage:  python scripts/setup_lid_cavity.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dualfield_benchmarks import generate_rectangle_mesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(ROOT, "geo", "mesh")
CFG_DIR = os.path.join(ROOT, "data", "config")

# Lid ramp: smoothly bring the lid to its target so the impulsive start does not
# shock the SETTLS characteristic trace.  The steady state is ramp-independent.
#
# Only the SINGULAR lid u=(1,0) is used: it has the Ghia,Ghia,Shin reference data.
# The regularized lid (16 x^2 (1-x)^2) was dropped -- it has no exact/reference
# solution, and convergence rates are already established by the physbc studies.
RAMP = "tanh(4.0*t)"
LID = {
    "singular": f"out[0] = {RAMP}; out[1] = 0;",
}

# Re -> (viscosity=1/Re, mesh lc, dt, T).  Higher Re => thinner wall layers
# (finer mesh) and longer transient to steady.  Tune T from the ||u1|| plateau.
CASES = {
    100:  (1.0 / 100.0,  0.02,   0.02,  30.0),
    400:  (1.0 / 400.0,  0.02,   0.015, 45.0),
    1000: (1.0 / 1000.0, 0.0125, 0.01,  70.0),
}


def mesh_name(lc):
    return f"LidDrivenCavity2D_lc{str(lc).replace('.', 'p')}.msh"


def main():
    os.makedirs(MESH_DIR, exist_ok=True)
    for lc in sorted({c[1] for c in CASES.values()}):
        out = os.path.join(MESH_DIR, mesh_name(lc))
        if os.path.exists(out):
            print("mesh exists", out)
        else:
            generate_rectangle_mesh(Lx=1.0, Ly=1.0, lc=lc, out=out)

    for variant, bdry in LID.items():
        for Re, (nu, lc, dt, T) in CASES.items():
            name = f"LidDrivenCavity2D_{variant}_Re{Re}"
            cfgdir = os.path.join(CFG_DIR, name)
            os.makedirs(cfgdir, exist_ok=True)
            cfg = {
                "mesh": f"./geo/mesh/{mesh_name(lc)}",
                "solver": "MINRES",
                "visualisation": 1,
                "printlevel": 1,
                "outputfile": f"LidDrivenCavity2D/{name}",
                "order": 2,
                "refinements": 0,
                "tol": 1e-08,
                "dt": dt,
                "T": T,
                "viscosity": nu,
                "trace_order": 2,
                "settls_iterations": 1,
                "velocity_mode": "dihedral",
                "lid_attributes": [14],
                "force_data": "out[0] = 0; out[1] = 0;",
                "initial_data_u": "out[0] = 0; out[1] = 0;",
                "boundary_data_u": bdry,
            }
            path = os.path.join(cfgdir, name + ".json")
            with open(path, "w") as f:
                json.dump(cfg, f, indent=4)
            print("wrote", path)


if __name__ == "__main__":
    main()
