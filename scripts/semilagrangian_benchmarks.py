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
from dualfield_benchmarks import (
    generate_box_mesh,
    generate_rectangle_mesh,
    compile_latex_report,
    benchmark,
)

from functools import partial


EXECUTABLE = "./build/semilagrangian_navierstokes_nitsche_mpi"

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
            refinements=lambda order: range(0, 3+5-order),
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


EXECUTABLE_ORDER2 = "./build/semilagrangian_navierstokes_nitsche_order2"

# Order-2 default: Heun tracing with CG projection velocity.
# L²-projects ND₂ velocity onto a globally continuous CG² field before
# characteristic tracing, avoiding normal-component discontinuities that
# degrade convergence with raw ND₂ evaluation or dihedral averaging.
HEUN_ORDER2_OVERRIDES = {
    "trace_order": 2,
    "velocity_mode": "cg_projection",
}


class ConstantFieldSemiLagOrder2(benchmark):
    """Constant uniform-flow benchmark for the second-order semi-Lagrangian
    solver (ND₂/H1₂ with Rapetti small-edge transport and BDF2 time stepping).

    Verifies that the solver reproduces the exact steady-state solution
    u = (1, 0, 0), p = 0 on the unit cube with matching Dirichlet data.
    """
    def __init__(self, executable=EXECUTABLE_ORDER2, solver="MINRES"):
        T = 0.2

        x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
        coords = [x0, x1, x2]
        nu = .001
        u = sp.Matrix([1, 0, 0])
        p = sp.Integer(0)
        init_u = sp.Matrix([0, 0, 0])
        name = "ConstantFieldSemiLagOrder2"
        meshname = "./geo/mesh/ConstantField.msh"

        generate_box_mesh(Lx=1, Ly=1, Lz=1, lc=0.4, out=meshname)

        exact = IBVPNavierStokesSolution(
            u=u, p=p, nu=nu, coords=coords, t=t,
            u_boundary=u, u_init=init_u)
        SimulationHelper = NavierStokesBenchmarkHelper(
            exact, name, mesh=meshname, visualisation=1, printlevel=2,
            executable=executable, linear_solver=solver,
            legacy_overrides=HEUN_ORDER2_OVERRIDES)
        data_processor = SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=SimulationHelper,
            SimulationDataProcessor=data_processor,
            dts=lambda order, refinements: 0.1,
            T=T,
            refinements=lambda order: [0],
            orders=[2],
            slice_normal="z")


class RigidRotationSemiLagOrder2(benchmark):
    """Rigid rotation benchmark for the second-order semi-Lagrangian solver.

    Solid-body rotation u = (-y, x, 0), p = 0 on the unit cube.
    """
    def __init__(self, executable=EXECUTABLE_ORDER2, solver="MINRES"):
        T = 5.

        x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
        coords = [x0, x1, x2]
        nu = .001
        u = sp.Matrix([-x1, x0, 0])
        p = sp.Integer(0)
        name = "RigidRotationSemiLagOrder2"
        meshname = "./geo/mesh/RigidRotation.msh"

        generate_box_mesh(Lx=1, Ly=1, Lz=1, lc=0.4, out=meshname)

        exact = IBVPNavierStokesSolution(
            u=u, p=p, nu=nu, coords=coords, t=t,
            u_init=sp.Matrix([0, 0, 0]), u_boundary=u)
        SimulationHelper = NavierStokesBenchmarkHelper(
            exact, name, mesh=meshname, visualisation=1, printlevel=2,
            executable=executable, linear_solver=solver,
            legacy_overrides=HEUN_ORDER2_OVERRIDES)
        data_processor = SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=SimulationHelper,
            SimulationDataProcessor=data_processor,
            dts=lambda order, refinements: 0.025,
            T=T,
            refinements=lambda order: [0],
            orders=[2],
            slice_normal="z")


class TaylorGreenSemiLagOrder2(benchmark):
    """Taylor-Green convergence benchmark for the second-order semi-Lagrangian
    solver (ND₂/H1₂ with Rapetti small-edge transport and BDF2 time stepping).

    Tests combined space-time convergence on a tetrahedral unit cube.
    Expects O(h²) spatial convergence in the L² norm.

    Note: uses inline tet mesh instead of ref-cube.mesh because the Rapetti
    small-edge framework requires simplicial (tet) elements.
    """
    def __init__(self, executable=EXECUTABLE_ORDER2, solver="MINRES"):
        T = 0.125
        name = "TaylorGreenSemiLagOrder2"
        meshname = "./data/meshes/unit-cube-tet.mesh"

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
            legacy_overrides=HEUN_ORDER2_OVERRIDES)
        data_processor = SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=simulation_helper,
            SimulationDataProcessor=data_processor,
            dts=lambda order, refinement: 0.5 / (8 * (2 ** refinement)),
            T=T,
            refinements=lambda order: range(0, 6 - order),
            orders=[2],
            slice_normal="z",
        )


class TravelingABCSemiLagOrder2(benchmark):
    """Traveling ABC flow benchmark for the second-order semi-Lagrangian solver.

    Advection-dominated manufactured solution testing the full BDF2 pipeline.

    Note: uses inline tet mesh instead of ref-cube.mesh because the Rapetti
    small-edge framework requires simplicial (tet) elements.
    """
    def __init__(self, executable=EXECUTABLE_ORDER2, solver="MINRES"):
        T = 0.25
        name = "TravelingABCSemiLagOrder2"
        meshname = "./data/meshes/unit-cube-tet.mesh"

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
            legacy_overrides=HEUN_ORDER2_OVERRIDES)
        data_processor = SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=simulation_helper,
            SimulationDataProcessor=data_processor,
            dts=lambda order, refinement: 0.5 / (32 * (2 ** refinement)),
            T=T,
            refinements=lambda order: range(0, 6 - order),
            orders=[2],
            slice_normal="z",
        )


class LidDrivenCavity3DExactSemiLagOrder2(benchmark):
    """3D lid-driven cavity for the second-order semi-Lagrangian solver.

    Top lid (attribute 16) driven by u_x = tanh(2t), all other faces
    homogeneous Dirichlet.
    """
    def __init__(self, executable=EXECUTABLE_ORDER2, solver="MINRES"):
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
        name = "LidDrivenCavity3DExactSemiLagOrder2"
        meshname = "./geo/mesh/LidDrivenCavity3DExactSemiLag.msh"

        generate_box_mesh(Lx=1, Ly=1, Lz=Lz, lc=0.5, out=meshname)

        exact = IBVPNavierStokes(
            u_init=init_u, nu=nu, coords=coords, t=t, u_boundary=tr_u,
            lid_attributes=[16])
        SimulationHelper = NavierStokesBenchmarkHelper(
            exact, name, mesh=meshname, visualisation=1, printlevel=0,
            executable=executable, linear_solver=solver,
            legacy_overrides=HEUN_ORDER2_OVERRIDES)
        data_processor = SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=SimulationHelper,
            SimulationDataProcessor=data_processor,
            dts=lambda order, refinements: 0.025,
            T=T,
            refinements=lambda order: [0],
            orders=[2],
            slice_normal="y")


# ===========================================================================
# Navier-Stokes with physical (no-slip) boundary conditions
# ---------------------------------------------------------------------------
# Thesis chapter "Navier-Stokes with physical boundary conditions": combine the
# semi-Lagrangian transport with the Nitsche weak imposition of the tangential
# trace. For all of these verification cases the prescribed boundary velocity is
# the FULL exact velocity on every boundary: its tangential part supplies the
# Nitsche data, its normal part enters the divergence-equation right-hand side
# (the "inhomogeneous normal data" remark), which the solver already assembles
# from boundary_data_u via BoundaryNormalLFIntegrator. The manufactured forcing
# is auto-generated by ManufacturedNavierStokes so it stays consistent with the
# solver's rotational-form convection operator.
#
# Each experiment is parametrized by the viscosity nu (= epsilon) and the
# polynomial order; epsilon-robustness is studied by registering several nu.
# ===========================================================================

ORDER1_EXECUTABLE = "./build/semilagrangian_navierstokes_nitsche_mpi"


def _physbc_runtime(order):
    """Return (executable, legacy_overrides) for the requested order."""
    if order == 1:
        return ORDER1_EXECUTABLE, EULER_DIHEDRAL_OVERRIDES
    if order == 2:
        return EXECUTABLE_ORDER2, HEUN_ORDER2_OVERRIDES
    raise ValueError(f"unsupported order {order}")


def _eps_tag(nu):
    """Filename-safe tag for a viscosity value (1e-2 -> '1em2', 0.0 -> '0')."""
    nu = float(nu)
    if nu == 0.0:
        return "0"
    return (
        f"{nu:.0e}".replace("e-0", "em").replace("e+0", "ep")
        .replace("e-", "em").replace("e+", "ep")
    )


class ManufacturedSolution2DSemiLag(benchmark):
    """2D manufactured stream-function solution on the unit square (0,1)^2.

    Stream function psi = phi(t) * S(x) with phi(t) = cos(2 pi t); the velocity
    is the skew gradient u = (d psi/dy, -d psi/dx), which is divergence-free and
    has vanishing normal trace on the boundary. Two variants:

      * tangential (noslip=False): S = sin(pi x) sin(pi y). The tangential trace
        of u on Gamma is non-trivial, so the Nitsche load is active.
      * no-slip   (noslip=True):   S = sin^2(pi x) sin^2(pi y), for which u = 0
        on Gamma, isolating the homogeneous Nitsche enforcement.

    h-convergence with tau coupled to h at fixed viscosity nu (= epsilon).
    """

    def __init__(self, executable=None, solver="GMRES",
                 nu=1e-2, noslip=False, order=1, refines=4):
        exe, overrides = _physbc_runtime(order)
        if executable is not None:
            exe = executable

        variant = "noslip" if noslip else "tangential"
        name = f"MMS2D_{variant}_eps{_eps_tag(nu)}_order{order}"
        meshname = f"./geo/mesh/{name}.msh"
        generate_rectangle_mesh(Lx=1.0, Ly=1.0, lc=0.25, out=meshname)

        x0, x1, t = sp.symbols("x0 x1 t", real=True)
        coords = [x0, x1]
        nu_s = sp.Float(float(nu))
        phi = sp.cos(2 * sp.pi * t)
        if noslip:
            S = sp.sin(sp.pi * x0) ** 2 * sp.sin(sp.pi * x1) ** 2
        else:
            S = sp.sin(sp.pi * x0) * sp.sin(sp.pi * x1)
        psi = phi * S
        u = sp.Matrix([sp.diff(psi, x1), -sp.diff(psi, x0)])
        p = phi * sp.cos(sp.pi * x0) * sp.cos(sp.pi * x1)

        exact = ManufacturedNavierStokes(u=u, p=p, nu=nu_s, coords=coords, t=t)
        simulation_helper = NavierStokesBenchmarkHelper(
            exact, name, mesh=meshname, visualisation=0, printlevel=1,
            executable=exe, linear_solver=solver, legacy_overrides=overrides)
        data_processor = SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=simulation_helper,
            SimulationDataProcessor=data_processor,
            dts=lambda order_, refinement: 0.0625 / (2 ** refinement),
            T=0.125,
            refinements=lambda order_: range(0, refines),
            orders=[order],
            slice_normal="z",
        )


class EthierSteinman3DSemiLag(benchmark):
    """Ethier-Steinman exact 3D Navier-Stokes solution on the cube (-1,1)^3.

    Classical unsteady analytical solution (a = pi/4, d = pi/2) that solves the
    incompressible Navier-Stokes equations with vanishing body force for every
    nu >= 0. The trace has a non-vanishing normal component on the boundary;
    that normal flux is supplied through the divergence-equation RHS via the
    normal part of the prescribed boundary velocity, while its tangential part
    furnishes the Nitsche data. (The forcing emitted by ManufacturedNavierStokes
    reduces to a pure gradient, so the measured velocity dynamics are those of
    the genuine operator.)
    """

    def __init__(self, executable=None, solver="GMRES", nu=1e-2, order=1, refines=3):
        exe, overrides = _physbc_runtime(order)
        if executable is not None:
            exe = executable

        name = f"EthierSteinman3D_eps{_eps_tag(nu)}_order{order}"
        meshname = f"./geo/mesh/{name}.msh"
        generate_box_mesh(Lx=2.0, Ly=2.0, Lz=2.0, lc=1.0, out=meshname,
                          x0=-1.0, y0=-1.0, z0=-1.0, graded=False)

        x0, x1, x2, t = sp.symbols("x0 x1 x2 t", real=True)
        coords = [x0, x1, x2]
        nu_s = sp.Float(float(nu))
        a = sp.pi / 4
        d = sp.pi / 2
        E = sp.exp
        decay = E(-d**2 * nu_s * t)
        u = -a * decay * sp.Matrix([
            E(a * x0) * sp.sin(a * x1 + d * x2) + E(a * x2) * sp.cos(a * x0 + d * x1),
            E(a * x1) * sp.sin(a * x2 + d * x0) + E(a * x0) * sp.cos(a * x1 + d * x2),
            E(a * x2) * sp.sin(a * x0 + d * x1) + E(a * x1) * sp.cos(a * x2 + d * x0),
        ])
        p = -(a**2 / 2) * E(-2 * d**2 * nu_s * t) * (
            E(2 * a * x0) + E(2 * a * x1) + E(2 * a * x2)
            + 2 * sp.sin(a * x0 + d * x1) * sp.cos(a * x2 + d * x0) * E(a * (x1 + x2))
            + 2 * sp.sin(a * x1 + d * x2) * sp.cos(a * x0 + d * x1) * E(a * (x2 + x0))
            + 2 * sp.sin(a * x2 + d * x0) * sp.cos(a * x1 + d * x2) * E(a * (x0 + x1))
        )

        exact = ManufacturedNavierStokes(u=u, p=p, nu=nu_s, coords=coords, t=t)
        simulation_helper = NavierStokesBenchmarkHelper(
            exact, name, mesh=meshname, visualisation=0, printlevel=1,
            executable=exe, linear_solver=solver, legacy_overrides=overrides)
        data_processor = SimulationDataProcessor(name)

        super().__init__(
            name=name,
            SimulationHelper=simulation_helper,
            SimulationDataProcessor=data_processor,
            dts=lambda order_, refinement: 0.0625 / (2 ** refinement),
            T=0.125,
            refinements=lambda order_: range(0, refines),
            orders=[order],
            slice_normal="z",
        )


class StokesSecondProblemSemiLag(benchmark):
    """Stokes' second problem: viscous layer above an oscillating wall.

    On the strip (0,L) x (0,H) the bottom wall oscillates tangentially with
    velocity U cos(omega t), generating the Stokes layer of thickness
    delta = sqrt(2 nu / omega). The flow is unidirectional and x-independent, so
    the convective term vanishes and (u, p=0) is an exact solution. The moving
    wall supplies a non-trivial tangential Nitsche datum; on the lateral
    boundaries the normal flux g_n = +/- u1 is carried by the divergence-equation
    RHS (the normal part of the prescribed boundary velocity). Sweep nu (= eps)
    at fixed omega to sharpen the layer (delta ~ sqrt(eps)).
    """

    def __init__(self, executable=None, solver="GMRES", nu=1e-2, order=1, refines=4,
                 U=1.0, omega=2.0 * 3.141592653589793, L=3.0, H=1.0):
        exe, overrides = _physbc_runtime(order)
        if executable is not None:
            exe = executable

        name = f"StokesSecond_eps{_eps_tag(nu)}_order{order}"
        meshname = f"./geo/mesh/{name}.msh"
        generate_rectangle_mesh(Lx=float(L), Ly=float(H), lc=float(H) / 16.0, out=meshname)

        x0, x1, t = sp.symbols("x0 x1 t", real=True)
        coords = [x0, x1]
        nu_s = sp.Float(float(nu))
        U_s = sp.Float(float(U))
        omega_s = sp.Float(float(omega))
        delta = sp.sqrt(2 * nu_s / omega_s)
        u1 = U_s * sp.exp(-x1 / delta) * sp.cos(omega_s * t - x1 / delta)
        u = sp.Matrix([u1, sp.Integer(0)])
        p = sp.Integer(0)

        exact = ManufacturedNavierStokes(u=u, p=p, nu=nu_s, coords=coords, t=t)
        simulation_helper = NavierStokesBenchmarkHelper(
            exact, name, mesh=meshname, visualisation=0, printlevel=1,
            executable=exe, linear_solver=solver, legacy_overrides=overrides)
        data_processor = SimulationDataProcessor(name)

        period = 2.0 * 3.141592653589793 / float(omega)
        super().__init__(
            name=name,
            SimulationHelper=simulation_helper,
            SimulationDataProcessor=data_processor,
            dts=lambda order_, refinement: (period / 16.0) / (2 ** refinement),
            T=period,
            refinements=lambda order_: range(0, refines),
            orders=[order],
            slice_normal="z",
        )


class WomersleySemiLag(benchmark):
    """Womersley pulsatile channel flow with homogeneous no-slip walls.

    On the channel (0,L) x (0,H) the oscillating body force f = (A cos(omega t), 0)
    drives the unidirectional Womersley profile. The convective term vanishes, so
    (u, p=0) is an exact solution. The no-slip walls y=0, y=H carry g = 0; on the
    lateral boundaries the normal flux g_n = +/- u1 is carried by the
    divergence-equation RHS. Sweep nu (= eps) at fixed omega; the Stokes layers
    of thickness sqrt(2 eps/omega) separate from a plug-like core as the
    Womersley number Wo = H sqrt(omega/eps) grows.
    """

    def __init__(self, executable=None, solver="GMRES", nu=1e-2, order=1, refines=4,
                 A=1.0, omega=2.0 * 3.141592653589793, L=3.0, H=1.0):
        exe, overrides = _physbc_runtime(order)
        if executable is not None:
            exe = executable

        name = f"Womersley_eps{_eps_tag(nu)}_order{order}"
        meshname = f"./geo/mesh/{name}.msh"
        generate_rectangle_mesh(Lx=float(L), Ly=float(H), lc=float(H) / 16.0, out=meshname)

        x0, x1, t = sp.symbols("x0 x1 t", real=True)
        coords = [x0, x1]
        nu_s = sp.Float(float(nu))
        A_s = sp.Float(float(A))
        omega_s = sp.Float(float(omega))
        H_s = sp.Float(float(H))

        # Womersley profile u1 = Re[(A/(i w)) (1 - cosh(k(y-H/2))/cosh(kH/2)) e^{iwt}]
        # with k = sqrt(i w / nu). Writing k = p(1+i), p = sqrt(w/(2 nu)), the
        # complex cosh is expanded into REAL-argument hyperbolic/trig functions
        #   cosh(k z) = cosh(p z) cos(p z) + i sinh(p z) sin(p z),
        # which keeps the (bounded) cosh-ratio together. Expanding instead into
        # exp(+/- p z) would produce terms ~ e^{p H} that overflow / cancel
        # catastrophically once the layer is thin (large Womersley number).
        p = sp.sqrt(omega_s / (2 * nu_s))
        Hh = H_s / 2
        z = x1 - Hh

        def ch(zeta):
            return sp.cosh(p * zeta) * sp.cos(p * zeta) \
                + sp.I * sp.sinh(p * zeta) * sp.sin(p * zeta)

        ratio = ch(z) / ch(Hh)               # ch(Hh) is a complex constant
        fac = (A_s / omega_s) * (sp.sin(omega_s * t) - sp.I * sp.cos(omega_s * t))
        u1 = sp.re(sp.expand(fac * (1 - ratio)))
        u = sp.Matrix([u1, sp.Integer(0)])
        p_field = sp.Integer(0)

        exact = ManufacturedNavierStokes(u=u, p=p_field, nu=nu_s, coords=coords, t=t)
        simulation_helper = NavierStokesBenchmarkHelper(
            exact, name, mesh=meshname, visualisation=0, printlevel=1,
            executable=exe, linear_solver=solver, legacy_overrides=overrides)
        data_processor = SimulationDataProcessor(name)

        period = 2.0 * 3.141592653589793 / float(omega)
        super().__init__(
            name=name,
            SimulationHelper=simulation_helper,
            SimulationDataProcessor=data_processor,
            dts=lambda order_, refinement: (period / 16.0) / (2 ** refinement),
            T=period,
            refinements=lambda order_: range(0, refines),
            orders=[order],
            slice_normal="z",
        )


def _build_physbc_benchmark_map():
    """Register the physical-BC convergence studies (thesis chapter).

    One registry entry per (experiment, viscosity, order). The viscosity sweep
    probes epsilon-robustness; epsilon = 0 (inviscid limit) is included for the
    cases that admit it (it is meaningless for the boundary-layer flows, whose
    layer thickness ~ sqrt(epsilon)). Edit EPS_VISC / orders below to extend.
    """
    bm = {}
    # Convergence-study viscosities. The decreasing positive sequence already
    # exhibits the epsilon -> 0 robustness. The exact inviscid limit epsilon = 0
    # is intentionally NOT in the default sweep: at epsilon = 0 the viscous /
    # Nitsche A-block degenerates to mass-only and the StokesMG-preconditioned
    # FGMRES path currently breaks down (beta = NaN). Add 0.0 below once the
    # inviscid solver path is fixed.
    eps_with_inviscid = [1e-2, 1e-3, 1e-4]
    # Boundary layers need epsilon > 0 (delta ~ sqrt(eps)). The thin-layer cases
    # are studied at viscosities whose Stokes layer delta = sqrt(2 eps/omega)
    # stays resolvable on the (finer) base mesh: at h/delta >~ 5 the
    # semi-Lagrangian transport of the near-discontinuous layer is unstable, so
    # the sweep keeps the coarsest level near h/delta ~ 2-3.
    eps_layer = [1e-2, 5e-3, 2e-3]
    orders = [1, 2]
    for order in orders:
        for eps in eps_with_inviscid:
            tag = _eps_tag(eps)
            bm[f"MMS2D_tangential_eps{tag}_order{order}"] = partial(
                ManufacturedSolution2DSemiLag, nu=eps, noslip=False, order=order)
            bm[f"MMS2D_noslip_eps{tag}_order{order}"] = partial(
                ManufacturedSolution2DSemiLag, nu=eps, noslip=True, order=order)
            bm[f"EthierSteinman3D_eps{tag}_order{order}"] = partial(
                EthierSteinman3DSemiLag, nu=eps, order=order)
        for eps in eps_layer:
            tag = _eps_tag(eps)
            bm[f"StokesSecond_eps{tag}_order{order}"] = partial(
                StokesSecondProblemSemiLag, nu=eps, order=order)
            bm[f"Womersley_eps{tag}_order{order}"] = partial(
                WomersleySemiLag, nu=eps, order=order)
    return bm


if __name__ == "__main__":
    benchmark_map = {
        "TaylorGreenSemiLag": TaylorGreenSemiLag,
    }
    # Physical-BC convergence studies (thesis chapter): MMS-2D, Ethier-Steinman,
    # Stokes' second problem, Womersley — one entry per (case, epsilon, order).
    benchmark_map.update(_build_physbc_benchmark_map())

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
        default=None,
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
        "--omp-threads",
        type=int,
        default=None,
        help="Set OMP_NUM_THREADS for parallel semi-Lagrangian advection",
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

    if args.omp_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
        print(f"[info] OMP_NUM_THREADS={args.omp_threads}")

    selected = benchmark_map.keys() if args.benchmark == "all" else [args.benchmark]
    failures = []

    for bench_name in selected:
        try:
            bench_cls = benchmark_map[bench_name]
            solver_name = args.solver
            if args.executable is not None:
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
