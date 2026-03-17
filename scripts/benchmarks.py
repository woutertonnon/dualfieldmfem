import sympy as sp
import SimIO
import gmsh
import os


def generate_box_mesh(Lx=1.0, Ly=1.0, Lz=1.0, lc=None, out="./geo/mesh/box.msh"):
    """
    Generate a 3D tetrahedral mesh of a rectangular box using Gmsh.

    The mesh is created using OpenCASCADE (OCC) geometry and includes:

    - A single volume representing the domain.
    - Physical groups for all boundary faces (xmin, xmax, ymin, ymax, zmin, zmax)
      with tags 11–16.
    - Graded refinement near the top face (``z = Lz``) and the face at ``x = Lx``.
    - Background mesh size field based on a distance-threshold strategy.
    - MFEM-compatible MSH 2.2 ASCII output.

    :param Lx: Length of the domain in the x-direction.
    :type Lx: float
    :param Ly: Length of the domain in the y-direction.
    :type Ly: float
    :param Lz: Length of the domain in the z-direction.
    :type Lz: float
    :param lc: Base mesh size. Defaults to ``min(Lx, Ly, Lz) / 10`` if ``None``.
    :type lc: float or None
    :param out: Output path for the generated ``.msh`` file.
    :type out: str

    :raises RuntimeError: If the volume or required boundary faces cannot be identified.
    :returns: None
    """
    gmsh.initialize()
    gmsh.model.add("box")

    gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
    gmsh.model.occ.synchronize()

    vols = gmsh.model.getEntities(3)
    if len(vols) != 1:
        gmsh.finalize()
        raise RuntimeError(f"Expected exactly 1 volume, found {len(vols)}: {vols}")
    vol = vols[0][1]

    if lc is None:
        lc = min(Lx, Ly, Lz) / 10.0

    surfs = gmsh.model.getBoundary([(3, vol)], oriented=False, recursive=False)

    xmin_faces, xmax_faces = [], []
    ymin_faces, ymax_faces = [], []
    zmin_faces, zmax_faces = [], []

    eps = 1e-5
    for (dim, s) in surfs:
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(dim, s)
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

    if not zmax_faces or not ymax_faces:
        gmsh.finalize()
        raise RuntimeError("Could not identify z=Lz or y=Ly faces for top refinement.")

    # Graded refinement near z=Lz and x=Lx
    refined_faces = zmax_faces + xmax_faces
    lc_far = lc
    lc_top = lc / 3.0
    d_inner, d_outer = 0.05 * Lz, 0.25 * Lz

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(lc_top, lc_far))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max(lc_top, lc_far))

    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList", refined_faces)

    f_thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", lc_top)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", lc_far)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMin", d_inner)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMax", d_outer)

    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_thr])
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

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

    gmsh.model.mesh.generate(3)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)

    if not gmsh.model.getEntitiesForPhysicalGroup(3, phys_vol):
        gmsh.finalize()
        raise RuntimeError("Physical Volume 'domain' is empty; MFEM would see element attributes = 0.")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    gmsh.write(out)
    gmsh.finalize()
    print(f"Wrote {out} (Lx={Lx}, Ly={Ly}, Lz={Lz}, lc={lc})")


class benchmark:
    """
    Harness for running benchmark tests in a reproducible manner.

    This class wires together:

    - a :class:`SimIO.SimulationHelper`-like object to generate/run configurations
    - a :class:`SimIO.SimulationDataProcessor`-like object to collect and plot results
    - parameter sweeps for time step, mesh refinement, and polynomial order
    """
    def __init__(self, name, SimulationHelper, SimulationDataProcessor,
                 dts, T, refinements, orders):
        """
        :param name: Benchmark name, used for output file naming.
        :type name: str
        :param SimulationHelper: Helper that generates configs and runs the solver.
        :type SimulationHelper: SimIO.SimulationHelper
        :param SimulationDataProcessor: Processor for collecting and plotting results.
        :type SimulationDataProcessor: SimIO.SimulationDataProcessor
        :param dts: Time step as a function of (order, refinement).
        :type dts: Callable[[int, int], float]
        :param T: End time.
        :type T: float
        :param refinements: List of mesh refinements as a function of order.
        :type refinements: Callable[[int], list[int]]
        :param orders: Polynomial orders to sweep over.
        :type orders: list[int]
        """
        self.name = name
        self.SimulationHelper = SimulationHelper
        self.SimulationDataProcessor = SimulationDataProcessor
        self.dts = dts
        self.T = T
        self.refinements = refinements
        self.orders = orders

    def run_euler(self):
        """Run the full parameter sweep using the Euler backend."""
        self.SimulationHelper.generate_config_files(
            self.T, self.dts, self.refinements, self.orders, tol=1e-7)
        self.SimulationHelper.run_all_configs_euler()

    def run_local(self):
        """Run the full parameter sweep locally."""
        self.SimulationHelper.run_convergence(
            self.T, self.dts, self.refinements, self.orders, tol=1e-7)

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

    Solves the Navier–Stokes equations on the unit cube [0,1]^3 with a
    smooth boundary-layer velocity profile on the top and side walls:

        u_bc = 5 * cut_off(x) * cut_off(y) * (z / Lz),  v_bc = w_bc = 0

    where ``cut_off`` tapers the velocity to zero near each edge, avoiding
    corner singularities. The mesh is graded near ``z = Lz`` and ``x = Lx``
    to resolve the shear layers. ``nu = 0.001``, so Re ~ 5000.
    """
    def __init__(self, executable="./build/hcurl_dualfieldnavierstokes_nitsche"):
        Lz = 1
        T = 10000

        x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
        coords = [x0, x1, x2]
        nu = 0.001
        # Smooth top-hat on [0,1]: ~1 in the interior, ~0 within 'edge' of each end.
        # Smaller eps → sharper transition; avoids corner singularities on the lid.
        def cut_off(s, edge=sp.Float(0.05), eps=sp.Float(0.03)):
            return sp.Rational(1, 2) * (sp.tanh((s - edge)/eps) - sp.tanh((s - (1 - edge))/eps))

        # One-sided ramp: ~1 only within 'edge' of z=Lz (the lid), ~0 elsewhere.
        # Replaces the linear x2/Lz so side walls see ~0 prescribed velocity (no-slip).
        def ramp_top(z, edge=sp.Float(0.04), eps=sp.Float(0.02)):
            return sp.Rational(1, 2) * (1 + sp.tanh((z - (Lz - edge)) / eps))

        # tanh(t): smooth, differentiable time ramp from 0→1 with unit timescale.
        tr_u = 5 * sp.Matrix([
            sp.tanh(t) * cut_off(x0) * cut_off(x1) * ramp_top(x2),
            sp.Integer(0),
            sp.Integer(0),
        ])
        init_u = sp.Matrix([0, 0, 0])
        name = "LidDrivenCavity3Dnoconvection"
        meshname = "./geo/mesh/" + name + ".msh"

        generate_box_mesh(Lx=1, Ly=1, Lz=Lz, lc=0.25, out=meshname)

        SimulationHelper = SimIO.NavierStokesSimulationHelper(
            SimIO.IBVPNavierStokes(
                u_init=init_u, nu=nu, coords=coords, t=t, u_boundary=tr_u),
            name, mesh=meshname, visualisation=1, printlevel=2,
            executable=executable)
        SimulationDataProcessor = SimIO.SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=SimulationHelper,
            SimulationDataProcessor=SimulationDataProcessor,
            dts=lambda order, refinements: 10.,
            T=T,
            refinements=lambda order: [0] if order == 1 else [0],
            orders=[1])


class LidDrivenCavity3DExact(benchmark):
    """
    Three-dimensional lid-driven cavity with attribute-based boundary conditions.

    The lid (top face, boundary attribute 16 = zmax) is driven with a smooth
    velocity profile ``u_x = U * cutoff(x) * cutoff(y) * tanh(t)``.  All
    other faces (attributes 11-15) receive homogeneous Dirichlet (u = 0)
    through the Nitsche bilinear form without any RHS contribution.

    The spatial cutoff avoids the corner singularity where the lid meets the
    stationary walls.  No z-ramp is needed because the lid velocity is only
    applied on the top face via ``lid_attributes = [16]``.
    """
    def __init__(self, executable="./build/hcurl_dualfieldnavierstokes_nitsche"):
        Lz = 1
        T = 10000

        x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
        coords = [x0, x1, x2]
        nu = 0.001

        # Lid velocity: only applied on zmax face (attribute 16)
        tr_u = 5 * sp.Matrix([
            1.,
            sp.Integer(0),
            sp.Integer(0),
        ])
        init_u = sp.Matrix([0, 0, 0])
        name = "LidDrivenCavity3DExactParallel"
        meshname = "./geo/mesh/" + name + ".msh"

        generate_box_mesh(Lx=1, Ly=1, Lz=Lz, lc=0.25, out=meshname)

        SimulationHelper = SimIO.NavierStokesSimulationHelper(
            SimIO.IBVPNavierStokes(
                u_init=init_u, nu=nu, coords=coords, t=t, u_boundary=tr_u,
                lid_attributes=[16]),
            name, mesh=meshname, visualisation=10, printlevel=2,
            executable=executable)
        SimulationDataProcessor = SimIO.SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=SimulationHelper,
            SimulationDataProcessor=SimulationDataProcessor,
            dts=lambda order, refinements: .1,
            T=T,
            refinements=lambda order: [2] if order == 1 else [0],
            orders=[1])


class ConstantField(benchmark):
    """
    Constant uniform-flow benchmark.

    Verifies that the solver reproduces the exact steady-state solution
    ``u = (1, 0, 0)``, ``p = 0`` on the unit cube with matching Dirichlet
    data on all boundaries.  Because the exact solution satisfies both the
    momentum equation and the divergence-free constraint, the discretisation
    error should remain at machine precision for any mesh and polynomial order.
    ``nu = 1``.
    """
    def __init__(self, executable="./build/hcurl_dualfieldnavierstokes_nitsche"):
        T = 100

        x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
        coords = [x0, x1, x2]
        nu = .001
        u = sp.Matrix([1, 0, 0])
        p = sp.Integer(0)
        init_u = sp.Matrix([0, 0, 0])
        name = "ConstantField"
        meshname = "./geo/mesh/ConstantField.msh"


        generate_box_mesh(Lx=1, Ly=1, Lz=1, lc=0.4, out=meshname)

        SimulationHelper = SimIO.NavierStokesSimulationHelper(
            SimIO.IBVPNavierStokesSolution(
                u=u, p=p, nu=nu, coords=coords, t=t,
                u_boundary=u, u_init=init_u),
            name, mesh=meshname, visualisation=1, printlevel=2,
            executable=executable)
        SimulationDataProcessor = SimIO.SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=SimulationHelper,
            SimulationDataProcessor=SimulationDataProcessor,
            dts=lambda order, refinements: 0.01,
            T=T,
            refinements=lambda order: [0],
            orders=[1])


class RigidRotation(benchmark):
    """
    Constant uniform-flow benchmark.

    Verifies that the solver reproduces the exact steady-state solution
    ``u = (1, 0, 0)``, ``p = 0`` on the unit cube with matching Dirichlet
    data on all boundaries.  Because the exact solution satisfies both the
    momentum equation and the divergence-free constraint, the discretisation
    error should remain at machine precision for any mesh and polynomial order.
    ``nu = 1``.
    """
    def __init__(self, executable="./build/hcurl_dualfieldnavierstokes_nitsche"):
        T = 100

        x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
        coords = [x0, x1, x2]
        nu = .001
        u = sp.Matrix([-x1, x0, 0])
        p = sp.Integer(0)
        init_u = sp.Matrix([0, 0, 0])
        name = "RigidRotationSingleField"
        meshname = "./geo/mesh/RigidRotation.msh"


        generate_box_mesh(Lx=1, Ly=1, Lz=1, lc=0.4, out=meshname)

        SimulationHelper = SimIO.NavierStokesSimulationHelper(
            SimIO.IBVPNavierStokesSolution(
                u=u, p=p, nu=nu, coords=coords, t=t,
                u_boundary=u, u_init=init_u),
            name, mesh=meshname, visualisation=100, printlevel=2,
            executable=executable)
        SimulationDataProcessor = SimIO.SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=SimulationHelper,
            SimulationDataProcessor=SimulationDataProcessor,
            dts=lambda order, refinements: 0.01,
            T=T,
            refinements=lambda order: [0],
            orders=[1])




if __name__ == "__main__":
    bench = LidDrivenCavity3DExact("./build/dualfieldnavierstokes_nitsche")
    #bench = RigidRotation()
    bench.run_local()
