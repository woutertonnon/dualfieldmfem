"""
Schaefer-Turek 2D-2 benchmark with a ROUND cylinder.

Domain: [0, 2.2] x [0, 0.41]
Cylinder: diameter D = 0.1, center (0.2, 0.2).

This is the 2D analogue of flow_around_cylinder_3d.py.  With viscosity
nu = 0.001 and the parabolic inflow below (U_mean = 1.0, D = 0.1) the
Reynolds number is Re = U_mean * D / nu = 100 — the canonical unsteady
vortex-shedding (von Karman street) benchmark.

Boundary physical groups (matching the 3D cylinder / cube benchmarks):
  tag 2 = inlet     (x = 0)
  tag 3 = outlet    (x = 2.2)
  tag 4 = side_walls (y = 0, y = 0.41)
  tag 5 = cylinder
  tag 1 = fluid area

Usage:
  python flow_around_cylinder_2d.py [--lc-min 0.01] [--lc-max 0.04] \
                                    [--d-mid 0.15] [--steepness 12] [-o output.msh]

The mesh is graded: fine (lc_min) near the cylinder, smoothly (tanh sigmoid)
coarsening to lc_max far from the cylinder.  The transition midpoint is at
distance d_mid; steepness controls how quickly the size grows.  Because the
MG solver starts from this mesh as its coarsest level, choose lc_min so that
the cylinder surface is already reasonably resolved (lc_min ~ 0.01 gives
~32 edges around the circumference; lc_min ~ 0.02 gives ~16).
"""
import gmsh
import sys


def make_flow_around_cylinder(lc_min=0.01, lc_max=0.04, d_mid=0.15,
                              steepness=12.0,
                              out="flow_around_cylinder_2d.msh"):
    """
    Generate graded 2D mesh: lc_min near the cylinder, smoothly (sigmoid/tanh)
    growing to lc_max far from the cylinder curve.

    Parameters
    ----------
    lc_min     : float  — element size on the cylinder surface
    lc_max     : float  — element size far from the cylinder
    d_mid      : float  — distance at which lc is halfway between min and max
    steepness  : float  — controls how sharp the transition is (higher = sharper)
    out        : str    — output mesh file path
    """
    gmsh.initialize()
    gmsh.model.add("flow_around_cylinder_2d")
    occ = gmsh.model.occ

    # Domain dimensions (Schaefer-Turek 2D benchmark)
    xmin, xmax = 0.0, 2.2
    ymin, ymax = 0.0, 0.41

    # Cylinder: D = 0.1, center at (0.2, 0.2)
    R = 0.05
    cx, cy = 0.2, 0.2

    # Create rectangle and disk
    rect = occ.addRectangle(xmin, ymin, 0.0, xmax - xmin, ymax - ymin)
    disk = occ.addDisk(cx, cy, 0.0, R, R)

    # Boolean difference: fluid = rectangle - disk
    fluid, _ = occ.cut([(2, rect)], [(2, disk)])
    occ.synchronize()

    fluid_tag = fluid[0][1]

    # Identify boundary curves by bounding box
    eps = 1e-6

    def curves_in_box(x0, y0, x1, y1):
        return [c for (d, c) in gmsh.model.getEntitiesInBoundingBox(
            x0, y0, -eps, x1, y1, eps, dim=1)]

    inlet = curves_in_box(xmin - eps, ymin - eps,
                          xmin + eps, ymax + eps)
    outlet = curves_in_box(xmax - eps, ymin - eps,
                           xmax + eps, ymax + eps)

    ymin_c = curves_in_box(xmin - eps, ymin - eps,
                           xmax + eps, ymin + eps)
    ymax_c = curves_in_box(xmin - eps, ymax - eps,
                           xmax + eps, ymax + eps)

    side_walls = list(set(ymin_c + ymax_c))

    # Cylinder curve: every boundary curve that isn't inlet/outlet/side_walls
    all_bdr = [c for (d, c) in gmsh.model.getBoundary(
        [(2, fluid_tag)], oriented=False, recursive=False)]
    known = set(inlet + outlet + side_walls)
    cylinder = [c for c in all_bdr if c not in known]

    # Physical groups — tags match the 3D cylinder / cube benchmarks
    gmsh.model.addPhysicalGroup(2, [fluid_tag], tag=1)
    gmsh.model.setPhysicalName(2, 1, "fluid")

    gmsh.model.addPhysicalGroup(1, inlet, tag=2)
    gmsh.model.setPhysicalName(1, 2, "inlet")

    gmsh.model.addPhysicalGroup(1, outlet, tag=3)
    gmsh.model.setPhysicalName(1, 3, "outlet")

    gmsh.model.addPhysicalGroup(1, side_walls, tag=4)
    gmsh.model.setPhysicalName(1, 4, "side_walls")

    gmsh.model.addPhysicalGroup(1, cylinder, tag=5)
    gmsh.model.setPhysicalName(1, 5, "cylinder")

    print(f"inlet:      {inlet}")
    print(f"outlet:     {outlet}")
    print(f"side_walls: {side_walls}")
    print(f"cylinder:   {cylinder}")

    # ------------------------------------------------------------------
    # Graded mesh sizing via Gmsh fields — smooth sigmoid transition
    # ------------------------------------------------------------------
    field = gmsh.model.mesh.field

    # Distance field from the cylinder curve
    f_dist = field.add("Distance")
    field.setNumbers(f_dist, "CurvesList", cylinder)

    # Smooth sigmoid: lc(d) = lc_min + (lc_max - lc_min) * 0.5*(1 + tanh(s*(d - d_mid)))
    # where d = F1 (distance field), s = steepness, d_mid = midpoint distance.
    avg = 0.5 * (lc_min + lc_max)
    amp = 0.5 * (lc_max - lc_min)
    expr = f"{avg} + {amp} * Tanh({steepness} * (F1 - {d_mid}))"

    f_sigmoid = field.add("MathEval")
    field.setString(f_sigmoid, "F", expr)

    field.setAsBackgroundMesh(f_sigmoid)

    print(f"Size field: {expr}")

    # Let the background field control sizing, not points/boundaries
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay (2D)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.model.mesh.generate(2)
    gmsh.write(out)

    # Stats
    _, _, node_tags = gmsh.model.mesh.getNodes()
    print(f"Wrote {out}: {len(node_tags)} nodes, "
          f"lc_min={lc_min}, lc_max={lc_max}, "
          f"d_mid={d_mid}, steepness={steepness}")

    gmsh.finalize()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Graded 2D mesh for flow around a cylinder")
    parser.add_argument("--lc-min", type=float, default=0.01,
                        help="Element size on the cylinder surface (default: 0.01)")
    parser.add_argument("--lc-max", type=float, default=0.04,
                        help="Element size far from the cylinder (default: 0.04)")
    parser.add_argument("--d-mid", type=float, default=0.15,
                        help="Distance at which size is halfway (default: 0.15)")
    parser.add_argument("--steepness", type=float, default=12.0,
                        help="Sigmoid steepness (default: 12.0)")
    parser.add_argument("-o", "--output", type=str,
                        default="geo/mesh/flow_around_cylinder_2d.msh",
                        help="Output mesh file")
    opts = parser.parse_args()
    make_flow_around_cylinder(lc_min=opts.lc_min, lc_max=opts.lc_max,
                              d_mid=opts.d_mid, steepness=opts.steepness,
                              out=opts.output)
