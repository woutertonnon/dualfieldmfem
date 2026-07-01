# Thesis benchmarks — how to run & post-process

End-to-end instructions for the three benchmark families used in the thesis
"Navier–Stokes with physical boundary conditions", run with the semi-Lagrangian
Nitsche solver:

1. **Physical-BC convergence studies** (`physbc`) — MMS-2D, Ethier–Steinman-3D,
   Stokes-2nd-problem, Womersley.
2. **2D flow around a cylinder** (Schäfer–Turek) — drag/lift/Strouhal + field/video.
3. **2D lid-driven cavity** — Ghia–Ghia–Shin centerlines + vortex structure.

Everything below assumes ETH **Euler** (SLURM); the runs are MPI + OpenMP. Local
machines are only used to view the pulled PNG/MP4 outputs.

---

## 0. Prerequisites (one-time, on Euler)

```bash
# toolchain (also needed at RUNTIME: io.h JIT-compiles the data snippets to ./tmp/*.so)
export MODULES="module load stack/2024-06 gcc/12.2.0 cmake suitesparse boost openmpi python"
eval "$MODULES"

# Python env for mesh/config generation + post-processing (NOT the solver)
python -m venv ~/physbc-venv
source ~/physbc-venv/bin/activate
pip install -r scripts/requirements_euler.txt          # gmsh, pyvista, matplotlib, numpy, ...

# gmsh needs libGLU on LD_LIBRARY_PATH (mesh generation fails without it)
export GLU_LIB=/cluster/software/stacks/2024-04/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-8.5.0/mesa-glu-9.0.2-pcs22pbrelfi5ztwuf3aqxgiiooauat2/lib
export LD_LIBRARY_PATH="$GLU_LIB:$LD_LIBRARY_PATH"     # path may move with the stack; find /cluster/software -name libGLU.so.1

# build the solver (order-1 and order-2 MPI binaries)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target \
  semilagrangian_navierstokes_nitsche_mpi \
  semilagrangian_navierstokes_nitsche_order2_mpi
```

For plotting/video, `pyvista` + `matplotlib` come from `~/physbc-venv`; **ffmpeg**
(for the cylinder movie) is `module load ffmpeg`. The order-2 velocity space
requires **unstructured triangular/tet meshes** — Cartesian/quad meshes do not
work (the small-edge Nédélec DOFs + the SL/DEC advection assume affine simplices).

---

## 1. Physical-BC convergence studies (`physbc`)

Full details in [`scripts/EULER_PHYSBC_README.md`](scripts/EULER_PHYSBC_README.md).
Each (experiment, viscosity ε, order) is one SLURM job that runs the whole
h/τ sweep (τ ∝ h) and writes a per-level CSV with the final-time L²(Ω) velocity
error (`u1_err_L2`). Meshes and configs are generated programmatically from the
registry in `scripts/semilagrangian_benchmarks.py`.

**Run:**
```bash
export MODULES="..." VENV=~/physbc-venv
export LD_LIBRARY_PATH="$GLU_LIB:$LD_LIBRARY_PATH"; source "$VENV/bin/activate"
bash scripts/submit_physbc_euler.sh                    # all 30 cases
bash scripts/submit_physbc_euler.sh StokesSecond Womersley   # or filter by prefix
# single case (debug):
sbatch --export=ALL,BENCH=MMS2D_tangential_eps1em3_order1 scripts/slurm_physbc_convergence_euler.sh
```

**Plot:**
```bash
python scripts/plot_physbc_convergence.py             # log-log error vs h, orders 1&2, per ε
#   -> out/plots/physbc/<family>_order12_convergence.png
python scripts/plot_physbc_solution_field.py --family StokesSecond --order 1 --level 2   # optional field snapshot
```

---

## 2. 2D flow around a cylinder (Schäfer–Turek)

Domain (0,2.2)×(0,0.41) minus a disk (D=0.1) at (0.2,0.2); parabolic inflow,
no-slip walls, consistent-Nitsche "do-nothing" pressure outflow. QoI: drag c_D,
lift c_L, pressure drop Δp, Strouhal St.

**Mesh** (committed; regenerate only if needed — needs gmsh+libGLU):
```bash
python geo/gmsh/flow_around_cylinder_2d.py                     # geo/mesh/flow_around_cylinder_2d.msh
python geo/gmsh/flow_around_cylinder_2d.py --lc-min 0.005      # ...flow_around_cylinder_2d_fine.msh (finer cylinder polygon)
```

**Run** (`scripts/slurm_flow_around_cylinder_2d_euler.sh`; env: `RE`→U_m, `ORDER`,
`DT`, `T`, `GAMMA` outflow penalty [applied as γ/h], `VMODE` dihedral|cg_projection,
`REFINEMENTS`, `VIS` VTU-output stride, `MESHFILE`, `NU`, `UM`):
```bash
# Re=20 (steady) / Re=100 (unsteady), 16 cores:
sbatch --export=ALL,RE=20  scripts/slurm_flow_around_cylinder_2d_euler.sh
sbatch --export=ALL,RE=100 scripts/slurm_flow_around_cylinder_2d_euler.sh
# Converged Re=100 (order 2, fine mesh, small dt) on 2 nodes x 64 cores:
sbatch --nodes=2 --ntasks=2 --cpus-per-task=64 --mem-per-cpu=1500M \
  --export=ALL,ORDER=2,RE=100,REFINEMENTS=0,DT=0.0003125,T=14,VIS=200,GAMMA=10,\
MESHFILE=geo/mesh/flow_around_cylinder_2d_fine.msh \
  scripts/slurm_flow_around_cylinder_2d_euler.sh
```
Outputs: QoI CSV `out/data/FlowAroundCylinder2D/<out>_qoi.csv` (cycle,t,cD,cL,dp,FD,FL);
VTU time series under `out/paraview/FlowAroundCylinder2D/<out>/Cycle*/`.

**Post-process** (in the venv):
```bash
# drag/lift maxima + Strouhal over a clean shedding window (use a long window for a crisp St):
python scripts/cylinder_qoi_postprocess.py out/data/FlowAroundCylinder2D/<out>_qoi.csv \
  --D 0.1 --Ubar 1.0 --tmin 11
# velocity + pressure field (final cycle):
python scripts/plot_cylinder_field.py out/paraview/FlowAroundCylinder2D/<out> -o field.png
# MP4 of the vortex street from the saved VTU cycles (NO re-simulation):
module load ffmpeg
python scripts/make_cylinder_movie.py out/paraview/FlowAroundCylinder2D/<out> \
  -o cyl.mp4 --dt 0.0003125 --fps 25            # --stride N to subsample, --fields u p
```
Frame cadence is set by `VIS` at run time (VIS=25 → ~21 real frames per shedding
period; VIS=200 → ~2–3). Re-time playback afterwards with ffmpeg `setpts` and
transcode to `.webm`/`.gif` if a player lacks H.264.

---

## 3. 2D lid-driven cavity

Unit square, unstructured triangles; all four walls are weak (Nitsche) Dirichlet
(only the lid, attr 14, is marked — the other walls get no-slip automatically).
Singular lid u=(1,0); Re=100/400/1000 via ν=1/Re. Compared to Ghia–Ghia–Shin (1982).

**Set up meshes + configs** (needs gmsh+libGLU; writes `geo/mesh/LidDrivenCavity2D_lc*.msh`
and `data/config/LidDrivenCavity2D_singular_Re*/`):
```bash
python scripts/setup_lid_cavity.py
```

**Run** one case (`scripts/slurm_lid_cavity_euler.sh`, `CFG=` selects the config):
```bash
sbatch --export=ALL,CFG=data/config/LidDrivenCavity2D_singular_Re100/LidDrivenCavity2D_singular_Re100.json \
  scripts/slurm_lid_cavity_euler.sh
# Re=1000 is solve-bound (~19 FGMRES iters/step) so extra cores give little speedup;
# 16 cores is fine (~6.5 h). Add --nodes=2 --ntasks=2 --cpus-per-task=64 only if desired.
```
Outputs: `out/data/LidDrivenCavity2D/<name>_vars.csv` (monitor `||u1||` for
steadiness); VTU under `out/paraview/LidDrivenCavity2D/<name>/`.

**Post-process** (in the venv):
```bash
# centerline velocity profiles vs Ghia (u along x=0.5, v along y=0.5) + max error:
python scripts/lid_cavity_postprocess.py out/paraview/LidDrivenCavity2D/<name> --Re 100 -o centerlines.png
# 2D field with contour lines: |u|+streamlines and streamfunction psi (primary + corner vortices):
python scripts/plot_cavity_field.py out/paraview/LidDrivenCavity2D/<name> -o field.png -N 240
```

> **Ghia table caveat:** the widely-circulated Ghia (1982) v-velocity table has a
> documented typo at **Re=400, x=0.9063** (−0.23827 is wrong). It is excluded from
> the comparison in `lid_cavity_postprocess.py` (`GHIA_V_BAD`) and marked as a gray
> "×" on the plot — do **not** read a solver discrepancy there as a bug.

---

## Notes

- Runs live on Euler; pull the PNG/MP4 outputs locally with `scp` to view.
- Long runs: launch, then poll `squeue`/the `_vars.csv` step count; SSH to Euler
  needs the VPN.
- Configs and generated meshes are gitignored (regenerated by the tooling); the
  four thesis-benchmark meshes are committed as `.gitignore` exceptions.
