import sympy as sp
import gmsh
import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

try:
    from simbench_adapters.mfem_ns import (
        IBVPNavierStokes,
        IBVPNavierStokesSolution,
        ManufacturedNavierStokes,
        NavierStokesBenchmarkHelper,
    )
    from simbench_core import SimulationDataProcessor
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from simbench_adapters.mfem_ns import (
        IBVPNavierStokes,
        IBVPNavierStokesSolution,
        ManufacturedNavierStokes,
        NavierStokesBenchmarkHelper,
    )
    from simbench_core import SimulationDataProcessor


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
    gmsh.option.setNumber("General.Terminal", 0)

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


def compile_latex_report(tex_path="tex_reports/dualfield_benchmark_report.tex", runs=1):
    """Compile the benchmark LaTeX report with pdflatex."""
    report = Path(tex_path)
    if not report.is_absolute():
        report = Path(__file__).resolve().parents[1] / report
    report = report.resolve()

    if not report.exists():
        print(f"[warn] Report source not found: {report}; skipping LaTeX compile.")
        return None

    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise RuntimeError("pdflatex not found in PATH")

    cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", report.name]
    for _ in range(max(1, int(runs))):
        proc = subprocess.run(
            cmd,
            cwd=str(report.parent),
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            log_file = report.with_suffix(".log")
            raise RuntimeError(
                f"pdflatex failed for {report.name} (exit {proc.returncode}); "
                f"see {log_file}"
            )

    pdf_file = report.with_suffix(".pdf")
    print(f"[info] Compiled LaTeX report: {pdf_file}")
    return pdf_file


class benchmark:
    """
    Harness for running benchmark tests in a reproducible manner.

    This class wires together:

        - a simulation helper object to generate/run configurations
        - a simulation data processor object to collect and plot results
    - parameter sweeps for time step, mesh refinement, and polynomial order
    """
    def __init__(self, name, SimulationHelper, SimulationDataProcessor,
                 dts, T, refinements, orders, tol=1e-10, slice_normal="z"):
        """
        :param name: Benchmark name, used for output file naming.
        :type name: str
        :param SimulationHelper: Helper that generates configs and runs the solver.
         :type SimulationHelper: object
        :param SimulationDataProcessor: Processor for collecting and plotting results.
         :type SimulationDataProcessor: object
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
        self.tol = tol
        self.slice_normal = slice_normal

    def run_euler(self):
        """Run the full parameter sweep using the Euler backend."""
        self.SimulationHelper.generate_config_files(
            self.T, self.dts, self.refinements, self.orders, tol=self.tol)
        self.SimulationHelper.run_all_configs_euler()

    def run_local(self):
        """Run the full parameter sweep locally."""
        out_dir = Path("out") / "data" / self.name
        vis_dir = Path("out") / "paraview" / self.name
        shutil.rmtree(out_dir, ignore_errors=True)
        shutil.rmtree(vis_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.SimulationHelper.run_convergence(
            self.T, self.dts, self.refinements, self.orders, tol=self.tol)

    def plot_local(self):
        """Collect and plot results from local runs."""
        plot_dir = Path("out") / "plots" / self.name
        shutil.rmtree(plot_dir, ignore_errors=True)

        out_dir = Path("out") / "data" / self.name
        csv_files = sorted(out_dir.glob(f"{self.name}_conv_order*_ref*_vars.csv"))
        if not csv_files:
            raise RuntimeError(f"No CSV files found in {out_dir}")

        with csv_files[0].open("r", encoding="utf-8") as f:
            header = f.readline().strip()
        columns = {item.strip() for item in header.split(",") if item.strip()}

        error_candidates = [c for c in ("u1_err_L2", "u2_err_L2") if c in columns]
        cons_candidates = [c for c in ("||u1||", "||u2||", "u1*w1", "u2*w2") if c in columns]

        has_multiple_refs = any(len(list(self.refinements(o))) > 1 for o in self.orders)
        if error_candidates and has_multiple_refs:
            self.SimulationDataProcessor.error_columns = error_candidates
            self.SimulationDataProcessor.collect_data()
            self.SimulationDataProcessor.plot_convergence()

        if cons_candidates:
            self.SimulationDataProcessor.cons_columns = cons_candidates
            self.SimulationDataProcessor.plot_conserved_variables()

        if not error_candidates and not cons_candidates:
            raise RuntimeError(f"No supported plotting columns found in {csv_files[0]}")

    def plot_euler(self):
        """Pull Euler results and plot them using the local pipeline."""
        self.SimulationDataProcessor.pull_data_from_euler()
        self.plot_local()

    def _default_slice_variant(self):
        order = max(self.orders)
        refinement = max(self.refinements(order))
        return order, refinement

    def _copy_files_tree(self, source_dir, target_dir):
        source = Path(source_dir)
        target = Path(target_dir)
        if not source.exists():
            return 0

        copied = 0
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(source)
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            copied += 1
        return copied

    def _organize_conservation_tree(self, bundle_dir, order, refinement):
        """Keep only conservation plots matching the bundle's order/ref."""
        cons_dir = Path(bundle_dir) / "conservation"
        if not cons_dir.exists():
            return

        suffix = f"_order{order}_ref{refinement}."
        for item in list(cons_dir.iterdir()):
            if not item.is_file():
                continue
            if suffix not in item.name:
                item.unlink()


    def plot_slices(self, normal=None, order=None, refinement=None, cycle=None, output_subdir=None, quiet=False):
        """Generate 2D slice plots via scripts/plot_slice.py."""
        if order is None or refinement is None:
            def_order, def_ref = self._default_slice_variant()
            order = def_order if order is None else order
            refinement = def_ref if refinement is None else refinement

        variant = f"conv_order{order}_ref{refinement}"
        script = Path(__file__).resolve().parent / "plot_slice.py"
        cmd = [
            sys.executable,
            str(script),
            "--name",
            self.name,
            "--variant",
            variant,
            "--normal",
            normal or self.slice_normal,
            "--format",
            "png",
            "-o",
            output_subdir
            if output_subdir
            else f"{self.name}_slice_{normal or self.slice_normal}_o{order}_r{refinement}",
        ]
        if cycle is not None:
            cmd.extend(["--cycle", str(cycle)])
        if quiet:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                err_text = (proc.stderr or proc.stdout or "").strip()
                if err_text:
                    err_lines = [ln for ln in err_text.splitlines() if ln.strip()]
                    err = err_lines[-1] if err_lines else err_text
                else:
                    err = f"plot_slice.py failed with exit code {proc.returncode}"
                raise RuntimeError(err)
        else:
            subprocess.run(cmd, check=True)

    def bundle_plots(self, normal=None, order=None, refinement=None, cycle=None):
        """Collect outputs into out/plots/benchmarks/<name>/.

        Layout::

            <name>/
              convergence/              -- convergence-rate plots (all orders/refs)
              <name>_rX_oY/             -- per-variant bundle
                conservation/           -- only this order/ref
                slice/                  -- only this order/ref
        """
        plots_root = Path("out") / "plots"
        bench_root = plots_root / "benchmarks" / self.name
        source_dir = plots_root / self.name          # raw output from plot_local

        shutil.rmtree(bench_root, ignore_errors=True)

        # -- Convergence: shared across all variants ----------------------------
        src_conv = source_dir / "convergence"
        if src_conv.exists():
            dst_conv = bench_root / "convergence"
            shutil.rmtree(dst_conv, ignore_errors=True)
            self._copy_files_tree(src_conv, dst_conv)

        # -- Per-variant bundles ------------------------------------------------
        paraview_case_dir = Path("out") / "paraview" / self.name
        has_paraview_case = paraview_case_dir.exists()
        if not has_paraview_case:
            print(f"[info] No ParaView output for {self.name}; skipping slice plots.")

        if order is not None and refinement is not None:
            pairs = [(int(order), int(refinement))]
        else:
            pairs = [(int(o), int(r)) for o in self.orders for r in self.refinements(o)]

        for o, r in pairs:
            bundle_dir = bench_root / f"{self.name}_r{r}_o{o}"
            shutil.rmtree(bundle_dir, ignore_errors=True)
            bundle_dir.mkdir(parents=True, exist_ok=True)

            # Conservation: copy all, then prune to this variant.
            src_cons = source_dir / "conservation"
            if src_cons.exists():
                self._copy_files_tree(src_cons, bundle_dir / "conservation")
                self._organize_conservation_tree(bundle_dir, o, r)

            # Slice.
            variant_dir = paraview_case_dir / f"{self.name}_conv_order{o}_ref{r}"
            if has_paraview_case and variant_dir.exists():
                tmp_rel = Path("benchmarks") / self.name / f"__tmp_slice_{self.name}_r{r}_o{o}"
                tmp_dir = plots_root / tmp_rel
                try:
                    self.plot_slices(
                        normal=normal,
                        order=o,
                        refinement=r,
                        cycle=cycle,
                        output_subdir=str(tmp_rel),
                        quiet=True,
                    )
                    self._copy_files_tree(tmp_dir, bundle_dir / "slice")
                except Exception as exc:
                    print(f"[warn] Slice plotting skipped for {self.name} (order{o}, ref{r}): {exc}")
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            print(f"[info] Bundled plots: {bundle_dir}")

        print(f"[info] Convergence plots: {bench_root / 'convergence'}")
        return bench_root

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
    def __init__(self, executable="./build/dualfieldnavierstokes_nitsche", solver="GMRES"):
        Lz = 1
        T = 5

        x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
        coords = [x0, x1, x2]
        nu = 0.001

        # Lid velocity: only applied on zmax face (attribute 16), with smooth
        # startup ramp to avoid impulsive t=0 forcing.
        ramp = sp.tanh(2 * t)
        tr_u = ramp * sp.Matrix([
            sp.Integer(1),
            sp.Integer(0),
            sp.Integer(0),
        ])
        init_u = sp.Matrix([0, 0, 0])
        name = "LidDrivenCavity3DExactParallel"
        meshname = "./geo/mesh/" + name + ".msh"

        generate_box_mesh(Lx=1, Ly=1, Lz=Lz, lc=0.5, out=meshname)

        exact = IBVPNavierStokes(
            u_init=init_u, nu=nu, coords=coords, t=t, u_boundary=tr_u,
            lid_attributes=[16])
        SimulationHelper = NavierStokesBenchmarkHelper(
            exact, name, mesh=meshname, visualisation=1, printlevel=0,
            executable=executable, linear_solver=solver)
        data_processor = SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=SimulationHelper,
            SimulationDataProcessor=data_processor,
            dts=lambda order, refinements: 0.2,
            T=T,
            refinements=lambda order: [0],
            orders=[1,2],
            slice_normal="y")


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
    def __init__(self, executable="./build/dualfieldnavierstokes_nitsche", solver="GMRES"):
        T = 0.2

        x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
        coords = [x0, x1, x2]
        nu = .001
        u = sp.Matrix([1, 0, 0])
        p = sp.Integer(0)
        init_u = sp.Matrix([0, 0, 0])
        name = "ConstantField"
        meshname = "./geo/mesh/ConstantField.msh"


        generate_box_mesh(Lx=1, Ly=1, Lz=1, lc=0.4, out=meshname)

        exact = IBVPNavierStokesSolution(
            u=u, p=p, nu=nu, coords=coords, t=t,
            u_boundary=u, u_init=init_u)
        SimulationHelper = NavierStokesBenchmarkHelper(
            exact, name, mesh=meshname, visualisation=1, printlevel=2,
            executable=executable, linear_solver=solver)
        data_processor = SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=SimulationHelper,
            SimulationDataProcessor=data_processor,
            dts=lambda order, refinements: 0.1,
            T=T,
            refinements=lambda order: [0],
            orders=[1,2],
            slice_normal="z")


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
    def __init__(self, executable="./build/dualfieldnavierstokes_nitsche", solver="GMRES"):
        T = 5.

        x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
        coords = [x0, x1, x2]
        nu = .001
        u = sp.Matrix([-x1, x0, 0])
        p = sp.Integer(0)
        name = "RigidRotationSingleField"
        meshname = "./geo/mesh/RigidRotation.msh"
        #meshname = "./extern/mfem/data/ref-cube.mesh"


        generate_box_mesh(Lx=1, Ly=1, Lz=1, lc=0.4, out=meshname)

        # Manufactured setup: with p=0 this produces force=(-2x,-2y,0)
        # and uses u_init=u, so the rigid rotation is an exact steady state.
        exact = IBVPNavierStokesSolution(u=u, p=p, nu=nu, coords=coords, t=t, u_init=sp.Matrix([0,0,0]),u_boundary=u)
        SimulationHelper = NavierStokesBenchmarkHelper(
            exact,
            name,
            mesh=meshname,
            visualisation=1,
            printlevel=2,
            executable=executable,
            linear_solver=solver,
            # Keep RT at least first-order so affine rigid rotation is
            # representable in the H(div) branch and does not pollute u1.
        )
        data_processor = SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=SimulationHelper,
            SimulationDataProcessor=data_processor,
            dts=lambda order, refinements: 0.1,
            T=T,
            refinements=lambda order: [0],
            orders=[1,2],
            slice_normal="z")


class TaylorGreenCombinedConvergence(benchmark):
    """Combined space-time convergence benchmark on an unstructured cube.

    The benchmark uses a divergence-free Taylor--Green-type manufactured
    solution on ``[0,1]^3`` and ties the timestep to mesh refinement so both
    temporal and spatial errors are reduced together.
    """

    def __init__(self, executable="./build/dualfieldnavierstokes_nitsche", solver="GMRES"):
        T = 0.125
        name = "TaylorGreenCombinedConvergence"
        meshname = "./extern/mfem/data/ref-cube.mesh"

        x0, x1, x2, t = sp.symbols("x0 x1 x2 t", real=True)
        coords = [x0, x1, x2]
        nu = sp.Float(0.01)
        # Avoid mesh-commensurate aliasing on the ref-cube base mesh.
        # k=2*pi can make lowest-order edge moments vanish on coarse levels,
        # which destroys observed convergence rates.
        k = sp.Float(1.0)
        decay = sp.exp(-sp.Float(2.0) * nu * k * k * t)

        u = sp.Matrix([
            sp.Mul(sp.sin(k * x0), sp.cos(k * x1), decay),
            sp.Mul(-1, sp.cos(k * x0), sp.sin(k * x1), decay),
            sp.Integer(0),
        ])

        p = sp.Integer(0)
        exact = ManufacturedNavierStokes(u=u, p=p, nu=nu, coords=coords, t=t)

        simulation_helper = NavierStokesBenchmarkHelper(
            exact,
            name,
            mesh=meshname,
            visualisation=0,
            printlevel=1,
            executable=executable,
            linear_solver=solver,
        )
        data_processor = SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=simulation_helper,
            SimulationDataProcessor=data_processor,
            # Keep T/dt integer on every refinement level so each case is
            # compared at the same physical end time (t_full = T).
            dts=lambda order, refinement: 0.5 / (8 * (2 ** (refinement*order))),
            T=T,
            refinements=lambda order: range(0,5-order),
            orders=[1, 2],
            slice_normal="z",
        )




if __name__ == "__main__":
    benchmark_map = {
        "LidDrivenCavity3DExact": LidDrivenCavity3DExact,
        "ConstantField": ConstantField,
        "RigidRotation": RigidRotation,
        "TaylorGreenCombinedConvergence": TaylorGreenCombinedConvergence,
    }

    parser = argparse.ArgumentParser(description="Run MFEM Navier-Stokes benchmarks")
    parser.add_argument(
        "--benchmark",
        default="all",
        choices=["all", *benchmark_map.keys()],
        help="Benchmark class to run, or 'all' to run every benchmark",
    )
    parser.add_argument(
        "--mode",
        default="local",
        choices=["local", "euler", "plot-local", "plot-euler"],
        help="Execution mode",
    )
    parser.add_argument(
        "--solver",
        default="GMRES_AMSADS",
        help="Linear solver string written into generated configs (e.g. GMRES, GMRES_AMSADS)",
    )
    parser.add_argument(
        "--executable",
        default="./build/dualfieldnavierstokes_nitsche",
        help="Override executable path used by the selected benchmark",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Linear solver relative tolerance written to generated configs",
    )
    parser.add_argument(
        "--plot-slice",
        action="store_true",
        help="Also generate a representative slice plot via plot_slice.py",
    )
    parser.add_argument(
        "--slice-normal",
        choices=["x", "y", "z"],
        default=None,
        help="Override slice normal (default is benchmark-specific)",
    )
    parser.add_argument(
        "--slice-order",
        type=int,
        default=None,
        help="Override polynomial order used for the slice variant",
    )
    parser.add_argument(
        "--slice-refinement",
        type=int,
        default=None,
        help="Override refinement used for the slice variant",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if one benchmark fails",
    )
    parser.add_argument(
        "--no-compile-report",
        action="store_true",
        help="Skip compiling tex_reports/benchmark_report.tex after plotting",
    )
    args = parser.parse_args()

    selected = benchmark_map.keys() if args.benchmark == "all" else [args.benchmark]
    failures = []

    for bench_name in selected:
        try:
            bench_cls = benchmark_map[bench_name]
            solver_name = args.solver
            if solver_name == "GMRES_AMSADS":
                if bench_name in ("LidDrivenCavity3DExact", "RigidRotation", "ConstantField"):
                    print(f"[warn] {bench_name}: GMRES_AMSADS can break down; using GMRES.")
                    solver_name = "GMRES"
            if args.executable:
                bench = bench_cls(executable=args.executable, solver=solver_name)
            else:
                bench = bench_cls(solver=solver_name)
            bench.tol = float(args.tol)

            if args.mode == "local":
                bench.run_local()
                bench.plot_local()
            elif args.mode == "euler":
                bench.run_euler()
            elif args.mode == "plot-local":
                bench.plot_local()
            else:
                bench.plot_euler()

            if args.mode in ("local", "plot-local", "plot-euler"):
                bench.bundle_plots(
                    normal=args.slice_normal,
                    order=args.slice_order,
                    refinement=args.slice_refinement,
                )

            if args.plot_slice:
                bench.plot_slices(
                    normal=args.slice_normal,
                    order=args.slice_order,
                    refinement=args.slice_refinement,
                )
        except SystemExit as exc:
            msg = f"{bench_name} failed with SystemExit({exc.code})"
            failures.append(msg)
            print(f"[error] {msg}")
            if args.fail_fast:
                raise
        except Exception as exc:
            msg = f"{bench_name} failed: {exc}"
            failures.append(msg)
            print(f"[error] {msg}")
            if args.fail_fast:
                raise

    if args.mode in ("local", "plot-local", "plot-euler") and not args.no_compile_report:
        try:
            compile_latex_report()
        except Exception as exc:
            msg = f"Report compilation failed: {exc}"
            failures.append(msg)
            print(f"[error] {msg}")
            if args.fail_fast:
                raise

    if failures:
        print("[warn] Some benchmarks failed:")
        for item in failures:
            print(f"  - {item}")
