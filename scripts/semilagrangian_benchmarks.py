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

# Import shared utilities from the dual-field benchmark script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dualfield_benchmarks import generate_box_mesh, compile_latex_report, benchmark


EXECUTABLE = "./build/semilagrangian_navierstokes_nitsche"

# Benchmark default: Euler tracing with edge-dihedral arrival velocity averaging.
EULER_DIHEDRAL_OVERRIDES = {
    "trace_order": 1,
    "settls_iterations": 2,
    "vertex_velocity_mode": "edge_dihedral",
}


class ConstantFieldSemiLag(benchmark):
    """Constant uniform-flow benchmark for the semi-Lagrangian solver.

    Verifies that the solver reproduces the exact steady-state solution
    u = (1, 0, 0), p = 0 on the unit cube with matching Dirichlet data.
    ND1 only (order restricted to 1).
    """
    def __init__(self, executable=EXECUTABLE, solver="GMRES"):
        T = 0.2

        x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
        coords = [x0, x1, x2]
        nu = .001
        u = sp.Matrix([1, 0, 0])
        p = sp.Integer(0)
        init_u = sp.Matrix([0, 0, 0])
        name = "ConstantFieldSemiLag"
        meshname = "./geo/mesh/ConstantField.msh"

        generate_box_mesh(Lx=1, Ly=1, Lz=1, lc=0.4, out=meshname)

        exact = IBVPNavierStokesSolution(
            u=u, p=p, nu=nu, coords=coords, t=t,
            u_boundary=u, u_init=init_u)
        SimulationHelper = NavierStokesBenchmarkHelper(
            exact, name, mesh=meshname, visualisation=1, printlevel=2,
            executable=executable, linear_solver=solver,
            legacy_overrides=EULER_DIHEDRAL_OVERRIDES)
        data_processor = SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=SimulationHelper,
            SimulationDataProcessor=data_processor,
            dts=lambda order, refinements: 0.1,
            T=T,
            refinements=lambda order: [0],
            orders=[1],
            slice_normal="z")


class RigidRotationSemiLag(benchmark):
    """Rigid rotation benchmark for the semi-Lagrangian solver.

    Solid-body rotation u = (-y, x, 0), p = 0 on the unit cube.
    ND1 only (order restricted to 1).
    """
    def __init__(self, executable=EXECUTABLE, solver="GMRES"):
        T = 5.

        x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
        coords = [x0, x1, x2]
        nu = .001
        u = sp.Matrix([-x1, x0, 0])
        p = sp.Integer(0)
        name = "RigidRotationSemiLag"
        meshname = "./geo/mesh/RigidRotation.msh"

        generate_box_mesh(Lx=1, Ly=1, Lz=1, lc=0.4, out=meshname)

        exact = IBVPNavierStokesSolution(
            u=u, p=p, nu=nu, coords=coords, t=t,
            u_init=sp.Matrix([0, 0, 0]), u_boundary=u)
        SimulationHelper = NavierStokesBenchmarkHelper(
            exact, name, mesh=meshname, visualisation=1, printlevel=2,
            executable=executable, linear_solver=solver,
            legacy_overrides=EULER_DIHEDRAL_OVERRIDES)
        data_processor = SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=SimulationHelper,
            SimulationDataProcessor=data_processor,
            dts=lambda order, refinements: 0.025,
            T=T,
            refinements=lambda order: [0],
            orders=[1],
            slice_normal="z")


class TaylorGreenSemiLag(benchmark):
    """Taylor-Green convergence benchmark for the semi-Lagrangian solver.

    Combined space-time convergence on an unstructured cube. ND1 only.
    """
    def __init__(self, executable=EXECUTABLE, solver="GMRES"):
        T = 0.125
        name = "TaylorGreenSemiLag"
        meshname = "./extern/mfem/data/ref-cube.mesh"

        x0, x1, x2, t = sp.symbols("x0 x1 x2 t", real=True)
        coords = [x0, x1, x2]
        nu = sp.Float(0.01)
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
            exact, name, mesh=meshname, visualisation=0, printlevel=1,
            executable=executable, linear_solver=solver,
            legacy_overrides=EULER_DIHEDRAL_OVERRIDES)
        data_processor = SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=simulation_helper,
            SimulationDataProcessor=data_processor,
            dts=lambda order, refinement: 0.5 / (8 * (2 ** refinement)),
            T=T,
            refinements=lambda order: range(0, 5-order),
            orders=[1],
            slice_normal="z",
        )


class TravelingABCSemiLag(benchmark):
    """Traveling Arnold-Beltrami-Childress (ABC) flow benchmark.

    Advection-dominated, strongly time-dependent manufactured solution on the
    unit cube [0,1]^3.  The velocity field

        u1 = A sin(k(z-ct)) + C cos(k(y-ct))
        u2 = B sin(k(x-ct)) + A cos(k(z-ct))
        u3 = C sin(k(y-ct)) + B cos(k(x-ct))

    with A=sqrt(3), B=sqrt(2), C=1, k=2pi, c=1, p=0 is divergence-free and
    satisfies the Navier-Stokes equations with a manufactured forcing term
    computed automatically by ManufacturedNavierStokes.
    """
    def __init__(self, executable=EXECUTABLE, solver="GMRES"):
        T = 0.25
        name = "TravelingABCSemiLag"
        meshname = "./extern/mfem/data/ref-cube.mesh"

        x0, x1, x2, t = sp.symbols("x0 x1 x2 t", real=True)
        coords = [x0, x1, x2]
        nu = sp.Float(0.001)

        A = sp.sqrt(3)
        B = sp.sqrt(2)
        C = sp.Integer(1)
        k = 2 * sp.pi
        c = sp.Integer(1)

        alpha = k * (x0 - c * t)
        beta = k * (x1 - c * t)
        gamma = k * (x2 - c * t)

        u = sp.Matrix([
            A * sp.sin(gamma) + C * sp.cos(beta),
            B * sp.sin(alpha) + A * sp.cos(gamma),
            C * sp.sin(beta) + B * sp.cos(alpha),
        ])
        p = sp.Integer(0)

        exact = ManufacturedNavierStokes(u=u, p=p, nu=nu, coords=coords, t=t)

        simulation_helper = NavierStokesBenchmarkHelper(
            exact, name, mesh=meshname, visualisation=0, printlevel=1,
            executable=executable, linear_solver=solver,
            legacy_overrides=EULER_DIHEDRAL_OVERRIDES)
        data_processor = SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=simulation_helper,
            SimulationDataProcessor=data_processor,
            dts=lambda order, refinement: 0.5 / (8 * (2 ** refinement)),
            T=T,
            refinements=lambda order: range(0, 5 - order),
            orders=[1],
            slice_normal="z",
        )


class LidDrivenCavity3DExactSemiLag(benchmark):
    """3D lid-driven cavity for the semi-Lagrangian solver.

    Top lid (attribute 16) driven by u_x = tanh(2t), all other faces
    homogeneous Dirichlet. ND1 only.
    """
    def __init__(self, executable=EXECUTABLE, solver="GMRES"):
        Lz = 1
        T = 15

        x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
        coords = [x0, x1, x2]
        nu = 0.001

        ramp = sp.tanh(2 * t)
        tr_u = ramp * sp.Matrix([
            sp.Integer(1),
            sp.Integer(0),
            sp.Integer(0),
        ])
        init_u = sp.Matrix([0, 0, 0])
        name = "LidDrivenCavity3DExactSemiLag"
        meshname = "./geo/mesh/" + name + ".msh"

        generate_box_mesh(Lx=1, Ly=1, Lz=Lz, lc=0.5, out=meshname)

        exact = IBVPNavierStokes(
            u_init=init_u, nu=nu, coords=coords, t=t, u_boundary=tr_u,
            lid_attributes=[16])
        SimulationHelper = NavierStokesBenchmarkHelper(
            exact, name, mesh=meshname, visualisation=1, printlevel=0,
            executable=executable, linear_solver=solver,
            legacy_overrides=EULER_DIHEDRAL_OVERRIDES)
        data_processor = SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=SimulationHelper,
            SimulationDataProcessor=data_processor,
            dts=lambda order, refinements: 0.025,
            T=T,
            refinements=lambda order: [0],
            orders=[1],
            slice_normal="y")


if __name__ == "__main__":
    benchmark_map = {
        "ConstantFieldSemiLag": ConstantFieldSemiLag,
        "RigidRotationSemiLag": RigidRotationSemiLag,
        "TaylorGreenSemiLag": TaylorGreenSemiLag,
        "TravelingABCSemiLag": TravelingABCSemiLag,
        "LidDrivenCavity3DExactSemiLag": LidDrivenCavity3DExactSemiLag,
    }

    parser = argparse.ArgumentParser(description="Run semi-Lagrangian Navier-Stokes benchmarks")
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
        default="GMRES",
        help="Linear solver string written into generated configs",
    )
    parser.add_argument(
        "--executable",
        default=EXECUTABLE,
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
        help="Skip compiling tex_reports/semilagrangian_benchmark_report.tex after plotting",
    )
    args = parser.parse_args()

    selected = benchmark_map.keys() if args.benchmark == "all" else [args.benchmark]
    failures = []

    for bench_name in selected:
        try:
            bench_cls = benchmark_map[bench_name]
            solver_name = args.solver
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
            compile_latex_report("tex_reports/semilagrangian_benchmark_report.tex")
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
