import sympy as sp
import SimIO
import gmsh
import os

class benchmark:
    """
    Harness for running benchmark tests in a reproducible manner.

    This class wires together:

    - a :class:`SimIO.SimulationHelper`-like object to generate/run configurations
    - a :class:`SimIO.SimulationDataProcessor`-like object to collect and plot results
    - parameter sweeps for time step, mesh refinement, and polynomial order
    """
    def __init__(self,name,SimulationHelper,SimulationDataProcessor, dts, refinements, orders):
        """
        Docstring for __init__
        
        :param self: Description
        :param name (string): Description
        :param SimulationHelper (SimIO.SimulationHelper): Description
        :param SimulationDataProcessor (SimIO.SimulationDataProcessor): Description
        :param dts (Callable[[int,int],float]): timestep for order (first argument) and refinement (second argument)
        :param refinements (Callable[int,List[int]]): mesh refinements for order (argument)
        :param orders (List[int]): orders that are considered
        """
        self.name = name
        self.SimulationHelper = SimulationHelper
        self.SimulationDataProcessor = SimulationDataProcessor
        self.dts = dts
        self.refinements = refinements
        self.orders = orders

    def run_euler(self):
        """Run the full parameter sweep using the Euler backend."""
        self.SimulationHelper.generate_config_files(1.,self.dts,self.refinements,self.orders,tol=1e-7)
        self.SimulationHelper.run_all_configs_euler()

    def run_local(self):
        """Run the full parameter sweep locally."""
        self.SimulationHelper.run_convergence(1.,self.dts,self.refinements,self.orders,tol=1e-7)

    def plot_local(self):
        """Collect and plot results from local runs."""
        self.SimulationDataProcessor.collect_data()
        self.SimulationDataProcessor.plot_convergence()

    def plot_euler(self):
        """Pull Euler results and plot them using the local pipeline."""
        self.SimulationDataProcessor.pull_data_from_euler()
        self.plot_local()

class LidDrivenCavity3D(benchmark):
    """
    Three-dimensional lid-driven cavity benchmark.

    This benchmark sets up a Navier–Stokes problem on a unit cube with
    a smooth lid-driven boundary condition and graded mesh refinement.
    """
    def __init__(self):
        """
        Construct the 3D lid-driven cavity benchmark.
        """
        Lz = 1


        x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
        coords = [x0, x1, x2]
        nu = .001
        cut_off = lambda t: 2 - sp.cosh(((t-0.5)*2.)**10*1.317)
        tr_u = 5*sp.Matrix([cut_off(x0)*cut_off(x1)*x2/Lz,sp.Integer(0),sp.Integer(0)])
        init_u = sp.Matrix([0,0,0])
        name = "LidDrivenCavity3D"
        meshname = "./geo/mesh/"+ name + ".msh" 
        visualisation = 1
        printlevel = 2

        self.generate_mesh(Lx=1,Ly=1,Lz=Lz,lc=.25,out=meshname)
        dts=lambda order, refinements: 0.01
        refinements=lambda order: [0] if order==1 else [0]
        orders=[1]

        SimulationHelper = SimIO.NavierStokesSimulationHelper(SimIO.IBVPNavierStokes(u_init=init_u,nu=nu,coords=coords,t=t, u_boundary=tr_u),name,mesh=meshname,visualisation=visualisation,printlevel=printlevel)
        SimulationDataProcessor = SimIO.SimulationDataProcessor(name)
        super().__init__(name=name,SimulationHelper=SimulationHelper,SimulationDataProcessor=SimulationDataProcessor,dts=dts,refinements=refinements,orders=orders)

    def generate_mesh(self, Lx=1.0, Ly=1.0, Lz=1.0, lc=None, out="./geo/mesh/box.msh"):
        """
        Generate a 3D tetrahedral mesh of a rectangular box using Gmsh.

        The mesh is created using OpenCASCADE (OCC) geometry and includes:

        - A single volume representing the domain.
        - Physical groups for all boundary faces (xmin, xmax, ymin, ymax, zmin, zmax).
        - Graded refinement near:
          * the top face (``z = Lz``),
          * the face at ``x = Lx``.
        - Background mesh size field based on a distance-threshold strategy.
        - MFEM-compatible MSH 2.2 ASCII output.

        :param Lx: Length of the domain in the x-direction.
        :type Lx: float
        :param Ly: Length of the domain in the y-direction.
        :type Ly: float
        :param Lz: Length of the domain in the z-direction.
        :type Lz: float
        :param lc: Base mesh size. If ``None``, defaults to
                   ``min(Lx, Ly, Lz) / 10``.
        :type lc: float or None
        :param out: Output path for the generated ``.msh`` file.
        :type out: str

        :raises RuntimeError: If the volume or required boundary faces
                              cannot be identified.
        :returns: None
        :rtype: None
        """
        import os
        import gmsh

        gmsh.initialize()
        gmsh.model.add("box")

        # Geometry (OCC)
        gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
        gmsh.model.occ.synchronize()

        # Robustly grab the actual volume tag after synchronize (OCC can change tags)
        vols = gmsh.model.getEntities(3)
        if len(vols) != 1:
            gmsh.finalize()
            raise RuntimeError(f"Expected exactly 1 volume, found {len(vols)}: {vols}")
        vol = vols[0][1]

        # Base mesh size
        if lc is None:
            lc = min(Lx, Ly, Lz) / 10.0

        # Boundary surfaces of the volume
        surfs = gmsh.model.getBoundary([(3, vol)], oriented=False, recursive=False)

        xmin_faces, xmax_faces = [], []
        ymin_faces, ymax_faces = [], []
        zmin_faces, zmax_faces = [], []

        eps = 1e-5
        for (dim, s) in surfs:
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(dim, s)
            print(x0)
            print(x1)
            print(y0)
            print(y1)
            print(z0)
            print(z1)

            if abs(x0 - 0.0) < eps and abs(x1 - 0.0) < eps:
                xmin_faces.append(s)
            elif abs(x0 - Lx) < eps and abs(x1 - Lx) < eps:
                xmax_faces.append(s)
            elif abs(y0 - 0.0) < eps and abs(y1 - 0.0) < eps:
                ymin_faces.append(s)
            elif abs(y0 - Ly) < eps and abs(y1 - Ly) < eps:
                ymax_faces.append(s)
            elif abs(z0 - 0.0) < eps and abs(z1 - 0.0) < eps:
                zmin_faces.append(s)
            elif abs(z0 - Lz) < eps and abs(z1 - Lz) < eps:
                zmax_faces.append(s)
        refined_faces = zmax_faces[:]
        refined_faces.extend(xmax_faces[:])

        # -----------------------------
        # Graded refinement (as requested)
        #   1) finer near top face z = Lz
        #   2) finer near bottom edge (z = 0, x = Lx)
        # -----------------------------

        # "Far" size and refined sizes (tune ratios if you like)
        lc_far = lc
        lc_top = lc / 3.0     # near z = Lz

        # Transition thicknesses (smooth grading)
        d_top_inner, d_top_outer = 0.05 * Lz, 0.25 * Lz

        # Don't force uniform mesh sizing when using background fields
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(lc_top, lc_far))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max(lc_top, lc_far))

        # Field A: refine near top face(s) z = Lz
        if (not zmax_faces) or (not ymax_faces):
            gmsh.finalize()
            raise RuntimeError("Could not identify z=Lz or y=Ly faces for top refinement (zmax_faces is empty).")


        f_dist_refinement = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist_refinement, "SurfacesList", refined_faces)

        f_thr_refinement = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thr_refinement, "InField", f_dist_refinement)
        gmsh.model.mesh.field.setNumber(f_thr_refinement, "SizeMin", lc_top)
        gmsh.model.mesh.field.setNumber(f_thr_refinement, "SizeMax", lc_far)
        gmsh.model.mesh.field.setNumber(f_thr_refinement, "DistMin", d_top_inner)
        gmsh.model.mesh.field.setNumber(f_thr_refinement, "DistMax", d_top_outer)

        fields_to_min = [f_thr_refinement]

        # Combine via minimum (most refined wins)
        f_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", fields_to_min)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

        # Physical groups (critical for MFEM element/boundary attributes)
        phys_vol = gmsh.model.addPhysicalGroup(3, [vol])
        gmsh.model.setPhysicalName(3, phys_vol, "domain")

        def add_surface_group(tag, name, faces):
            if faces:
                pg = gmsh.model.addPhysicalGroup(2, faces, tag=tag)
                gmsh.model.setPhysicalName(2, pg, name)

        add_surface_group(11, "xmin", xmin_faces)
        add_surface_group(12, "xmax", xmax_faces)
        add_surface_group(13, "ymin", ymin_faces)
        add_surface_group(14, "ymax", ymax_faces)
        add_surface_group(15, "zmin", zmin_faces)
        add_surface_group(16, "zmax", zmax_faces)

        # Generate 3D mesh
        gmsh.model.mesh.generate(3)

        # MFEM-friendly MSH2.2 ASCII output, and write ONLY physical-tagged entities
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)

        # Optional sanity check: physical group must contain the volume
        if not gmsh.model.getEntitiesForPhysicalGroup(3, phys_vol):
            gmsh.finalize()
            raise RuntimeError("Physical Volume 'domain' is empty; MFEM would see element attributes = 0.")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

        # Write mesh
        gmsh.write(out)
        gmsh.finalize()
        print(f"Wrote {out} (Lx={Lx}, Ly={Ly}, Lz={Lz}, lc={lc})")


if __name__ == "__main__":
    bench = LidDrivenCavity3D()
    bench.run_local()

