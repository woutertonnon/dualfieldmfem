#!/usr/bin/env python3
"""
Plot velocity field on a 2D slice through 3D ParaView VTU data.
The slice plane always passes through the centre of the domain bounding box.
Outputs are saved to out/plots/<name>/ (or out/plots/<-o>/) in PNG, PDF, and EPS.

Options:
    --name NAME          Case name (required). Matches a folder under
                         data/visualisation/paraview/, e.g. RigidRotation.
    --cycle CYCLES       Cycle(s) to plot: a single number, a comma-separated
                         list (0,10,20,30), or a range start:step:stop
                         (0:10:100). Default: last available.
    --normal {x,y,z}     Slice-plane normal direction. Default: z.
    --variant VARIANT    Subfolder variant, e.g. conv_order2_ref0.
                         Default: first found (warns if multiple exist).
    --basedir DIR        Root directory for case folders.
                         Default: data/visualisation/paraview.
    --stream-density D   Streamline density for streamplot. Default: 1.5.
    --format FMT         Output formats: comma-separated list of png, pdf, eps.
                         Default: png,pdf,eps (all three).
    -o FOLDER            Output subfolder name under out/plots/. Default: NAME.

Examples:
    python scripts/plot_slice.py --name RigidRotation --normal z
    python scripts/plot_slice.py --name RigidRotation --cycle 50 --normal z --stream-density 2.0
    python scripts/plot_slice.py --name RigidRotation --cycle 0,10,20,30 --normal z
    python scripts/plot_slice.py --name RigidRotation --cycle 0:10:100 --normal z
    python scripts/plot_slice.py --name RigidRotation --variant conv_order2_ref0 --normal y
    python scripts/plot_slice.py --name RigidRotation --normal z --format png
    python scripts/plot_slice.py --name RigidRotation --normal z -o my_experiment
"""

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.ticker import ScalarFormatter
import vtk
from vtk.util.numpy_support import vtk_to_numpy

plt.rcParams.update({"font.size": 14})


def load_vtu(datadir: str, cycle: int) -> vtk.vtkUnstructuredGrid:
    """Load the VTU file for a given cycle."""
    cycle_dir = os.path.join(datadir, f"Cycle{cycle:06d}")
    vtu_files = sorted(glob.glob(os.path.join(cycle_dir, "*.vtu")))
    if not vtu_files:
        raise FileNotFoundError(f"No .vtu files in {cycle_dir}")
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_files[0])
    reader.Update()
    return reader.GetOutput()


def slice_grid(grid: vtk.vtkUnstructuredGrid,
               normal: np.ndarray,
               origin: np.ndarray) -> vtk.vtkPolyData:
    """Cut the unstructured grid with a plane and return the slice polydata."""
    plane = vtk.vtkPlane()
    plane.SetOrigin(*origin)
    plane.SetNormal(*normal)

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(grid)
    cutter.Update()
    return cutter.GetOutput()


def normal_from_name(name: str) -> np.ndarray:
    """Convert 'x', 'y', or 'z' to a unit normal vector."""
    return {"x": np.array([1, 0, 0]),
            "y": np.array([0, 1, 0]),
            "z": np.array([0, 0, 1])}[name.lower()]


def inplane_axes(normal: np.ndarray):
    """Return (axis0, axis1) index pair for the 2D plane coordinates."""
    idx = int(np.argmax(np.abs(normal)))
    axes = [i for i in range(3) if i != idx]
    return axes[0], axes[1]


def parse_cycles(cycle_str: str) -> list[int]:
    """Parse cycle specification: single int, comma-separated, or start:step:stop."""
    if ":" in cycle_str:
        parts = cycle_str.split(":")
        if len(parts) == 3:
            start, step, stop = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            start, stop = int(parts[0]), int(parts[1])
            step = 1
        else:
            raise ValueError(f"Invalid range: '{cycle_str}'. Use start:step:stop or start:stop.")
        return list(range(start, stop + 1, step))
    else:
        return [int(c.strip()) for c in cycle_str.split(",")]


def make_colorbar(fig, mappable, label, ax=None):
    """Add a colorbar with scientific notation formatting."""
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cb.ax.yaxis.set_major_formatter(fmt)
    cb.ax.yaxis.get_offset_text().set_fontsize(10)
    return cb


def get_field_data(sliced, field_name):
    """Extract a vector field from sliced polydata. Returns None if not found."""
    arr = sliced.GetPointData().GetArray(field_name)
    if arr is None:
        return None
    return vtk_to_numpy(arr)


def plot_velocity_panel(ax, triang, x, y, u, v, speed, xi, yi, stream_density, title):
    """Plot velocity magnitude with streamlines on a single axes."""
    tpc = ax.tripcolor(triang, speed, shading="gouraud", cmap="viridis")
    interp_u = tri.LinearTriInterpolator(triang, u)
    interp_v = tri.LinearTriInterpolator(triang, v)
    Xi, Yi = np.meshgrid(xi, yi)
    Ui = interp_u(Xi, Yi)
    Vi = interp_v(Xi, Yi)
    ax.streamplot(xi, yi, Ui, Vi, color="white", linewidth=0.5,
                  density=stream_density, arrowsize=0.6)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13)
    return tpc


def plot_cycle(datadir, cycle, args, normal, centre, outdir, formats):
    """Load, slice, and plot u1, u2, w1, w2 in a (2,2) subplot for a single cycle."""
    grid = load_vtu(datadir, cycle)
    sliced = slice_grid(grid, normal, centre)

    npts = sliced.GetNumberOfPoints()
    if npts == 0:
        print(f"Cycle {cycle}: slice produced 0 points — skipping.")
        return

    pts = vtk_to_numpy(sliced.GetPoints().GetData())
    ax0, ax1 = inplane_axes(normal)
    axis_labels = {0: "x", 1: "y", 2: "z"}
    x = pts[:, ax0]
    y = pts[:, ax1]
    triang = tri.Triangulation(x, y)

    ngrid = 200
    xi = np.linspace(x.min(), x.max(), ngrid)
    yi = np.linspace(y.min(), y.max(), ngrid)

    norm_idx = int(np.argmax(np.abs(normal)))
    slice_coord = centre[norm_idx]

    # Detect available fields
    avail = [sliced.GetPointData().GetArrayName(i)
             for i in range(sliced.GetPointData().GetNumberOfArrays())]

    # Define the 4 panels: (field_name, title, is_velocity)
    panels = [
        ("u1", r"$\mathbf{u}_1$ (H(curl))", True),
        ("u2", r"$\mathbf{u}_2$ (H(div))",  True),
        ("w1", r"$\mathbf{w}_1$",            True),
        ("w2", r"$\mathbf{w}_2$",            True),
    ]

    # Filter to only fields that exist
    panels = [(f, t, v) for (f, t, v) in panels if f in avail]
    if not panels:
        print(f"Cycle {cycle}: no u1/u2/w1/w2 fields found. Available: {avail}")
        return

    # Pad to 4 panels for (2,2) layout
    nrows, ncols = 2, 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12))
    axes_flat = axes.flatten()

    for idx, ax in enumerate(axes_flat):
        if idx >= len(panels):
            ax.set_visible(False)
            continue

        field_name, title, is_vel = panels[idx]
        vel = get_field_data(sliced, field_name)
        if vel is None:
            ax.set_visible(False)
            continue

        u_comp = vel[:, ax0]
        v_comp = vel[:, ax1]
        speed = np.sqrt(u_comp**2 + v_comp**2)

        tpc = plot_velocity_panel(ax, triang, x, y, u_comp, v_comp, speed,
                                  xi, yi, args.stream_density, title)
        make_colorbar(fig, tpc, r"$|\mathbf{" + field_name + r"}|$", ax=ax)
        ax.set_xlabel(rf"${axis_labels[ax0]}$")
        ax.set_ylabel(rf"${axis_labels[ax1]}$")

    fig.suptitle(f"Cycle {cycle}, slice {axis_labels[norm_idx]}={slice_coord:.3f}",
                 fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    stem = f"fields_cycle{cycle:06d}_{axis_labels[norm_idx]}{slice_coord:.3f}"
    for ext in formats:
        path = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)

    # --- Pressure subplot (p1 and p2 side by side if available) ---
    p_fields = [(f, t) for f, t in [("p1", r"$p_1$"), ("p2", r"$p_2$"),
                                     ("p0", r"$p_0$"), ("p3", r"$p_3$")]
                if f in avail]
    if p_fields:
        ncols_p = min(len(p_fields), 2)
        fig_p, axes_p = plt.subplots(1, ncols_p, figsize=(7 * ncols_p, 6))
        if ncols_p == 1:
            axes_p = [axes_p]

        for idx, (pname, plabel) in enumerate(p_fields[:ncols_p]):
            ax = axes_p[idx]
            p = vtk_to_numpy(sliced.GetPointData().GetArray(pname)).ravel()
            vlim = max(abs(p.min()), abs(p.max()))
            if vlim == 0:
                vlim = 1.0
            tpc = ax.tripcolor(triang, p, shading="gouraud", cmap="RdBu_r",
                               vmin=-vlim, vmax=vlim)
            make_colorbar(fig_p, tpc, plabel, ax=ax)
            ax.set_xlabel(rf"${axis_labels[ax0]}$")
            ax.set_ylabel(rf"${axis_labels[ax1]}$")
            ax.set_aspect("equal")
            ax.set_title(plabel, fontsize=13)

        fig_p.suptitle(f"Pressure — Cycle {cycle}", fontsize=15, y=0.98)
        fig_p.tight_layout(rect=[0, 0, 1, 0.96])

        p_stem = f"pressure_cycle{cycle:06d}_{axis_labels[norm_idx]}{slice_coord:.3f}"
        for ext in formats:
            path = os.path.join(outdir, f"{p_stem}.{ext}")
            fig_p.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Saved {path}")
        plt.close(fig_p)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", type=str, required=True,
                        help="Case name, e.g. RigidRotation or LidDrivenCavity3Dnoconvection")
    parser.add_argument("--basedir", type=str,
                        default="data/visualisation/paraview",
                        help="Root directory containing case folders (default: data/visualisation/paraview)")
    parser.add_argument("--variant", type=str, default=None,
                        help="Subfolder variant, e.g. conv_order2_ref0 (default: first found)")
    parser.add_argument("--cycle", type=str, default=None,
                        help="Cycle(s): single number, comma-separated (0,10,20), "
                             "or range start:step:stop (0:10:100). Default: last available.")
    parser.add_argument("--normal", type=str, default="z", choices=["x", "y", "z"],
                        help="Slice-plane normal direction (default: z)")
    parser.add_argument("--stream-density", type=float, default=1.5,
                        help="Streamline density for streamplot (default: 1.5)")
    parser.add_argument("--format", type=str, default="png,pdf,eps",
                        help="Output formats: comma-separated list of png, pdf, eps (default: png,pdf,eps)")
    parser.add_argument("-o", type=str, default=None,
                        help="Output subfolder name under out/plots/ (default: --name value)")
    args = parser.parse_args()

    # Parse output formats
    formats = [f.strip() for f in args.format.split(",")]
    for f in formats:
        if f not in ("png", "pdf", "eps"):
            raise ValueError(f"Unsupported format '{f}'. Use png, pdf, or eps.")

    # Resolve datadir from --name and --variant
    case_dir = os.path.join(args.basedir, args.name)
    if not os.path.isdir(case_dir):
        raise FileNotFoundError(f"Case directory not found: {case_dir}")
    if args.variant is not None:
        datadir = os.path.join(case_dir, f"{args.name}_{args.variant}")
    else:
        subdirs = sorted(d for d in os.listdir(case_dir)
                         if os.path.isdir(os.path.join(case_dir, d)))
        if not subdirs:
            raise FileNotFoundError(f"No variant subdirectories in {case_dir}")
        if len(subdirs) == 1:
            datadir = os.path.join(case_dir, subdirs[0])
        else:
            print("Available variants:")
            for i, s in enumerate(subdirs):
                print(f"  [{i}] {s}")
            choice = input("Select variant number: ").strip()
            try:
                idx = int(choice)
                datadir = os.path.join(case_dir, subdirs[idx])
            except (ValueError, IndexError):
                raise ValueError(f"Invalid selection '{choice}'. Use 0-{len(subdirs)-1}.")
    if not os.path.isdir(datadir):
        raise FileNotFoundError(f"Variant directory not found: {datadir}")

    # Resolve cycles
    if args.cycle is None:
        cycle_dirs = sorted(glob.glob(os.path.join(datadir, "Cycle*")))
        if not cycle_dirs:
            raise FileNotFoundError(f"No Cycle* directories in {datadir}")
        cycles = [int(os.path.basename(cycle_dirs[-1]).replace("Cycle", ""))]
        print(f"Using last cycle: {cycles[0]}")
    else:
        cycles = parse_cycles(args.cycle)

    # Get domain centre from the first cycle
    grid0 = load_vtu(datadir, cycles[0])
    bounds = grid0.GetBounds()
    centre = np.array([(bounds[0]+bounds[1])/2,
                       (bounds[2]+bounds[3])/2,
                       (bounds[4]+bounds[5])/2])
    normal = normal_from_name(args.normal)

    outdir = os.path.join("out", "plots", args.o if args.o else args.name)
    if os.path.isdir(outdir):
        for f in os.listdir(outdir):
            os.remove(os.path.join(outdir, f))
    os.makedirs(outdir, exist_ok=True)

    for cycle in cycles:
        plot_cycle(datadir, cycle, args, normal, centre, outdir, formats)


if __name__ == "__main__":
    main()
