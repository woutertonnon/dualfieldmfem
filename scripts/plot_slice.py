#!/usr/bin/env python3
"""
Plot velocity field on a 2D slice through 3D ParaView VTU data.
The slice plane always passes through the centre of the domain bounding box.
Outputs are saved to out/plots/<name>/ (or out/plots/<-o>/) in PNG, PDF, and EPS.

Options:
    --name NAME          Case name (required). Matches a folder under
                         out/paraview/, e.g. RigidRotation.
    --cycle CYCLES       Cycle(s) to plot: a single number, a comma-separated
                         list (0,10,20,30), or a range start:step:stop
                         (0:10:100). Default: last available.
    --normal {x,y,z}     Slice-plane normal direction. Default: z.
    --variant VARIANT    Subfolder variant, e.g. conv_order2_ref0.
                         Default: first found (warns if multiple exist).
    --basedir DIR        Root directory for case folders.
                         Default: out/paraview.
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
from matplotlib.path import Path as MplPath
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


def _extract_triangle_connectivity(poly: vtk.vtkPolyData) -> np.ndarray:
    """Extract triangle connectivity from vtkPolyData polys.

    Returns
    -------
    np.ndarray
        Array of shape (ntri, 3) with point indices.
    """
    polys = poly.GetPolys()
    if polys is None or polys.GetNumberOfCells() == 0:
        return np.empty((0, 3), dtype=np.int64)

    conn = vtk_to_numpy(polys.GetData())
    triangles = []
    i = 0
    n = len(conn)

    while i < n:
        nverts = int(conn[i])
        i += 1
        if nverts <= 0:
            continue

        ids = conn[i : i + nverts]
        i += nverts

        if nverts == 3:
            triangles.append(ids)
        elif nverts > 3:
            # Fallback fan triangulation (triangle filter should usually avoid this).
            for j in range(1, nverts - 1):
                triangles.append([ids[0], ids[j], ids[j + 1]])

    if not triangles:
        return np.empty((0, 3), dtype=np.int64)

    return np.asarray(triangles, dtype=np.int64)


def _triangle_quality_mask(triang: tri.Triangulation) -> np.ndarray:
    """Build a conservative mask for degenerate/sliver triangles."""
    tris = triang.triangles
    if tris is None or len(tris) == 0:
        return np.zeros(0, dtype=bool)

    x = triang.x
    y = triang.y

    x0 = x[tris[:, 0]]
    y0 = y[tris[:, 0]]
    x1 = x[tris[:, 1]]
    y1 = y[tris[:, 1]]
    x2 = x[tris[:, 2]]
    y2 = y[tris[:, 2]]

    area2 = np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

    bbox_area = max((x.max() - x.min()) * (y.max() - y.min()), 1e-30)
    tiny = area2 < (2.0 * bbox_area * 1e-12)

    l01 = (x1 - x0) ** 2 + (y1 - y0) ** 2
    l12 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    l20 = (x0 - x2) ** 2 + (y0 - y2) ** 2
    max_len2 = np.maximum(np.maximum(l01, l12), l20)

    # area2/max_len2 is near zero for very skinny triangles.
    with np.errstate(divide="ignore", invalid="ignore"):
        shape_q = np.where(max_len2 > 0.0, area2 / max_len2, 0.0)
    skinny = shape_q < 1e-10

    return tiny | skinny


def _mask_delaunay_outside_slice(
    triang: tri.Triangulation,
    poly: vtk.vtkPolyData,
    ax0: int,
    ax1: int,
) -> np.ndarray:
    """Mask Delaunay triangles whose centroids are outside the sliced cells."""
    tris = triang.triangles
    if tris is None or len(tris) == 0:
        return np.zeros(0, dtype=bool)

    x = triang.x
    y = triang.y
    cx = (x[tris[:, 0]] + x[tris[:, 1]] + x[tris[:, 2]]) / 3.0
    cy = (y[tris[:, 0]] + y[tris[:, 1]] + y[tris[:, 2]]) / 3.0
    centroids = np.column_stack([cx, cy])

    inside = np.zeros(len(tris), dtype=bool)
    n_cells = poly.GetNumberOfCells()

    for ci in range(n_cells):
        cell = poly.GetCell(ci)
        if cell is None:
            continue
        npts = cell.GetNumberOfPoints()
        if npts < 3:
            continue

        ids = [cell.GetPointId(j) for j in range(npts)]
        px = x[ids]
        py = y[ids]

        pts2d = np.column_stack([px, py])
        # Remove duplicate consecutive points.
        keep = np.ones(len(pts2d), dtype=bool)
        if len(pts2d) > 1:
            keep[1:] = np.any(np.abs(pts2d[1:] - pts2d[:-1]) > 1e-14, axis=1)
        pts2d = pts2d[keep]
        if len(pts2d) < 3:
            continue

        path = MplPath(pts2d)
        inside |= path.contains_points(centroids, radius=1e-12)

        if inside.all():
            break

    return ~inside


def build_slice_triangulation(
    sliced: vtk.vtkPolyData,
    ax0: int,
    ax1: int,
    mode: str = "vtk",
    debug: bool = False,
):
    """Return (poly_for_plot, triangulation, x, y) for the 2D slice."""
    if mode == "vtk":
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(sliced)
        cleaner.Update()

        tri_filter = vtk.vtkTriangleFilter()
        tri_filter.SetInputData(cleaner.GetOutput())
        tri_filter.Update()

        poly = tri_filter.GetOutput()
        pts = vtk_to_numpy(poly.GetPoints().GetData())
        x = pts[:, ax0]
        y = pts[:, ax1]

        triangles = _extract_triangle_connectivity(poly)
        if triangles.size == 0:
            raise RuntimeError("Slice triangulation failed: no triangles found in VTK polydata")
        triang = tri.Triangulation(x, y, triangles=triangles)
    else:
        poly = sliced
        pts = vtk_to_numpy(poly.GetPoints().GetData())
        x = pts[:, ax0]
        y = pts[:, ax1]
        triang = tri.Triangulation(x, y)

    mask = _triangle_quality_mask(triang)
    if mode == "delaunay":
        outside = _mask_delaunay_outside_slice(triang, poly, ax0, ax1)
        if outside.size:
            mask = np.logical_or(mask, outside)

    if mask.size and np.any(mask):
        triang.set_mask(mask)

    try:
        triang.get_trifinder()
    except RuntimeError as exc:
        if mode == "vtk":
            if debug:
                print(f"[debug] VTK triangulation invalid ({exc}); falling back to Delaunay.")
            return build_slice_triangulation(sliced, ax0, ax1, mode="delaunay", debug=debug)
        raise

    if debug:
        n_masked = int(np.count_nonzero(mask)) if mask.size else 0
        n_total = len(triang.triangles)
        print(
            f"[debug] Triangulation mode={mode}, points={len(x)}, triangles={n_total}, masked={n_masked}"
        )

    return poly, triang, x, y


def plot_velocity_panel(
    ax,
    triang,
    u,
    v,
    speed,
    xi,
    yi,
    stream_density,
    title,
    shading,
    clip_percentile,
):
    """Plot velocity magnitude with streamlines on a single axes."""
    vmax = None
    speed_plot = speed
    if clip_percentile < 100.0:
        vmax = float(np.nanpercentile(speed, clip_percentile))
        if np.isfinite(vmax) and vmax > 0.0:
            speed_plot = np.clip(speed, 0.0, vmax)
        else:
            vmax = None

    tpc = ax.tripcolor(triang, speed_plot, shading=shading, cmap="viridis", vmin=0.0, vmax=vmax)

    try:
        interp_u = tri.LinearTriInterpolator(triang, u)
        interp_v = tri.LinearTriInterpolator(triang, v)
        Xi, Yi = np.meshgrid(xi, yi)
        Ui = interp_u(Xi, Yi)
        Vi = interp_v(Xi, Yi)

        Ui = np.ma.masked_invalid(Ui)
        Vi = np.ma.masked_invalid(Vi)
        if np.ma.count(Ui) > 0 and np.ma.count(Vi) > 0:
            ax.streamplot(
                xi,
                yi,
                Ui,
                Vi,
                color="white",
                linewidth=0.5,
                density=stream_density,
                arrowsize=0.6,
            )
    except RuntimeError:
        # Keep the color panel even if streamline interpolation fails.
        pass

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13)
    return tpc


def plot_cycle(datadir, cycle, args, normal, centre, outdir, formats):
    """Load, slice, and plot u1, u2, w1, w2 in a (2,2) subplot for a single cycle."""
    grid = load_vtu(datadir, cycle)
    sliced_raw = slice_grid(grid, normal, centre)

    npts = sliced_raw.GetNumberOfPoints()
    if npts == 0:
        print(f"Cycle {cycle}: slice produced 0 points — skipping.")
        return

    ax0, ax1 = inplane_axes(normal)
    axis_labels = {0: "x", 1: "y", 2: "z"}
    sliced, triang, x, y = build_slice_triangulation(
        sliced_raw,
        ax0,
        ax1,
        mode=args.triangulation,
        debug=args.debug_slice,
    )

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

        if args.debug_slice:
            imax = int(np.nanargmax(speed))
            print(
                f"[debug] {field_name}: |.|_max={speed[imax]:.6g} at ({x[imax]:.6g}, {y[imax]:.6g}), "
                f"p99.5={np.nanpercentile(speed, 99.5):.6g}"
            )

        tpc = plot_velocity_panel(
            ax,
            triang,
            u_comp,
            v_comp,
            speed,
            xi,
            yi,
            args.stream_density,
            title,
            args.shading,
            args.clip_percentile,
        )
        make_colorbar(fig, tpc, r"$|\mathbf{" + field_name + r"}|$", ax=ax)
        ax.set_xlabel(rf"${axis_labels[ax0]}$")
        ax.set_ylabel(rf"${axis_labels[ax1]}$")

    fig.suptitle(f"Cycle {cycle}, slice {axis_labels[norm_idx]}={slice_coord:.3f}",
                 fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

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

            abs_p = np.abs(p)
            if args.clip_percentile < 100.0:
                vlim = float(np.nanpercentile(abs_p, args.clip_percentile))
            else:
                vlim = float(np.nanmax(abs_p))
            if vlim == 0:
                vlim = 1.0

            p_plot = np.clip(p, -vlim, vlim)
            tpc = ax.tripcolor(
                triang,
                p_plot,
                shading=args.shading,
                cmap="RdBu_r",
                vmin=-vlim,
                vmax=vlim,
            )
            make_colorbar(fig_p, tpc, plabel, ax=ax)
            ax.set_xlabel(rf"${axis_labels[ax0]}$")
            ax.set_ylabel(rf"${axis_labels[ax1]}$")
            ax.set_aspect("equal")
            ax.set_title(plabel, fontsize=13)

        fig_p.suptitle(f"Pressure — Cycle {cycle}", fontsize=15, y=0.98)
        fig_p.tight_layout(rect=(0, 0, 1, 0.96))

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
                        default="out/paraview",
                        help="Root directory containing case folders (default: out/paraview)")
    parser.add_argument("--variant", type=str, default=None,
                        help="Subfolder variant, e.g. conv_order2_ref0 (default: first found)")
    parser.add_argument("--cycle", type=str, default=None,
                        help="Cycle(s): single number, comma-separated (0,10,20), "
                             "or range start:step:stop (0:10:100). Default: last available.")
    parser.add_argument("--normal", type=str, default="z", choices=["x", "y", "z"],
                        help="Slice-plane normal direction (default: z)")
    parser.add_argument("--stream-density", type=float, default=1.5,
                        help="Streamline density for streamplot (default: 1.5)")
    parser.add_argument("--triangulation", type=str, default="vtk", choices=["vtk", "delaunay"],
                        help="Triangulation source for plotting (default: vtk)")
    parser.add_argument("--shading", type=str, default="flat", choices=["flat", "gouraud"],
                        help="Shading mode for tripcolor (default: flat)")
    parser.add_argument("--clip-percentile", type=float, default=99.5,
                        help="Clip color values to this percentile (0,100]; 100 disables clipping")
    parser.add_argument("--debug-slice", action="store_true",
                        help="Print triangulation and field extrema debug information")
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

    if not (0.0 < args.clip_percentile <= 100.0):
        raise ValueError("--clip-percentile must be in (0, 100].")

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
