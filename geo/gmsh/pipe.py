import gmsh
import sys

def make_pipe(L=1.0, R=0.1, lc=None, out="pipe.msh"):
    gmsh.initialize()
    gmsh.model.add("pipe")

    # Geometry: cylinder along x from (0,0,0) to (L,0,0)
    vol_tag = gmsh.model.occ.addCylinder(0, 0, 0, L, 0, 0, R)
    gmsh.model.occ.synchronize()

    # Mesh sizing
    if lc is None:
        lc = min(R, L) / 10.0  # decent default
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

    # Get boundary surfaces and classify them: inlet, outlet, wall
    # Boundary returns dimTags; for a volume boundary, we get surfaces (dim=2)
    surfs = gmsh.model.getBoundary([(3, vol_tag)], oriented=False, recursive=False)

    inlet = []
    outlet = []
    wall = []

    eps = 1e-9
    for (dim, s) in surfs:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, s)
        # End caps are planar at x=0 or x=L
        if abs(xmin - 0.0) < eps and abs(xmax - 0.0) < eps:
            inlet.append(s)
        elif abs(xmin - L) < eps and abs(xmax - L) < eps:
            outlet.append(s)
        else:
            wall.append(s)

    # Physical groups (useful for BCs/materials in solvers)
    gmsh.model.addPhysicalGroup(3, [vol_tag], tag=1)
    gmsh.model.setPhysicalName(3, 1, "fluid")

    if inlet:
        gmsh.model.addPhysicalGroup(2, inlet, tag=11)
        gmsh.model.setPhysicalName(2, 11, "inlet")
    if outlet:
        gmsh.model.addPhysicalGroup(2, outlet, tag=12)
        gmsh.model.setPhysicalName(2, 12, "outlet")
    if wall:
        gmsh.model.addPhysicalGroup(2, wall, tag=13)
        gmsh.model.setPhysicalName(2, 13, "wall")

    # Generate 3D mesh
    gmsh.model.mesh.generate(3)

    # Write mesh
    gmsh.write(out)
    gmsh.finalize()
    print(f"Wrote {out} (L={L}, R={R}, lc={lc})")

if __name__ == "__main__":
    # Optional CLI: python pipe.py L R lc out.msh
    args = sys.argv[1:]
    L = float(args[0]) if len(args) > 0 else 1.0
    R = float(args[1]) if len(args) > 1 else 0.1
    lc = float(args[2]) if len(args) > 2 else None
    out = args[3] if len(args) > 3 else "pipe.msh"
    make_pipe(L=L, R=R, lc=lc, out=out)
