import sympy as sp
from sympy.printing import ccode
from typing import List
from pathlib import Path
import shutil
import json
import subprocess
import os
import paramiko
from paramiko.ssh_exception import SSHException, ChannelException
import sys
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import numpy as np
from matplotlib.ticker import MultipleLocator
from typing import Callable, Iterable

class ExactManipulationsSpace:
    """
    Symbolic differential operators for vector calculus in 3D.

    This class provides symbolic implementations of common vector calculus
    operators using :mod:`sympy`, including gradient, divergence, curl,
    Laplacian, and convective terms.

    :param coords: Spatial coordinate symbols (e.g. ``[x, y, z]``).
    :type coords: list[sympy.Symbol]
    """
    
    def __init__(self, coords):
        """
        Initialize the symbolic manipulation space.

        :param coords: Coordinate symbols defining the spatial domain.
        :type coords: list[sympy.Symbol]
        """
        self.coords = coords


    def grad(self, scalar_field):
        """
        Compute the gradient of a scalar field.

        :param scalar_field: Scalar symbolic expression.
        :type scalar_field: sympy.Expr
        :returns: Gradient vector.
        :rtype: sympy.Matrix
        """
        return sp.Matrix([sp.diff(scalar_field, c) for c in self.coords])

    def laplacian_vector(self, vec_field):
        """
        Compute the component-wise Laplacian of a vector field.

        :param vec_field: Vector field.
        :type vec_field: sympy.Matrix
        :returns: Vector Laplacian.
        :rtype: sympy.Matrix
        """

        n = vec_field.rows
        lap = sp.Matrix.zeros(n, 1)
        for i in range(n):
            lap_i = 0
            for c in self.coords:
                lap_i += sp.diff(vec_field[i], c, 2)
            lap[i] = lap_i
        return lap

    def convective_term(self, u):
        """
        Compute the convective term using the identity

        .. math::
            (\\mathbf{u} \\cdot \\nabla) \\mathbf{u}
            = -\\mathbf{u} \\times (\\nabla \\times \\mathbf{u})

        :param u: Velocity vector field.
        :type u: sympy.Matrix
        :returns: Convective term.
        :rtype: sympy.Matrix
        """
        conv = -u.cross(self.curl(u))#
        return conv

    def divergence(self, u):
        """
        Compute the divergence of a vector field.

        :param u: Vector field.
        :type u: sympy.Matrix
        :returns: Scalar divergence.
        :rtype: sympy.Expr
        """
        term = 0
        for j, c in enumerate(self.coords):
            term += sp.diff(u[j],c)
        return term


    def curl(self, u):
        """
        Compute the 3D curl of a vector field.

        Assumes coordinates are ordered as ``[x, y, z]``.

        :param u: Vector field ``Matrix([u1, u2, u3])``.
        :type u: sympy.Matrix
        :returns: Curl vector.
        :rtype: sympy.Matrix
        """
        x, y, z = self.coords
        u1, u2, u3 = u
        w1 = sp.diff(u3, y) - sp.diff(u2, z)
        w2 = sp.diff(u1, z) - sp.diff(u3, x)
        w3 = sp.diff(u2, x) - sp.diff(u1, y)
        return sp.Matrix([w1, w2, w3])


class manufacturedStokes(ExactManipulationsSpace):
    """
    Manufactured solution for the steady Stokes problem.

    This class represents a symbolic manufactured solution of the form

    .. math::

        m \\mathbf{u} - \\nu \\Delta \\mathbf{u} + \\nabla p = \\mathbf{f},

    using the vector identity

    .. math::

        \\Delta \\mathbf{u} = - \\nabla \\times (\\nabla \\times \\mathbf{u})
        \\quad \\text{(for divergence-free fields)}.

    The right-hand side is constructed symbolically from the provided
    velocity and pressure fields.

    :param u: Prescribed velocity field.
    :type u: sympy.Matrix
    :param p: Prescribed pressure field.
    :type p: sympy.Expr
    :param mass: Mass coefficient (reaction term).
    :type mass: sympy.Expr or float
    :param nu: Kinematic viscosity.
    :type nu: sympy.Expr or float
    :param coords: Spatial coordinate symbols (e.g. ``[x, y, z]``).
    :type coords: list[sympy.Symbol]
    """

    def __init__(self, u, p, mass, nu, coords):
        """
        Initialize the manufactured Stokes solution.

        :param u: Prescribed velocity field.
        :type u: sympy.Matrix
        :param p: Prescribed pressure field.
        :type p: sympy.Expr
        :param mass: Mass coefficient (reaction term).
        :type mass: sympy.Expr or float
        :param nu: Kinematic viscosity.
        :type nu: sympy.Expr or float
        :param coords: Spatial coordinate symbols (e.g. ``[x, y, z]``).
        :type coords: list[sympy.Symbol]
        """
        super().__init__(coords)
        self.u = u
        self.p = p
        self.mass = mass
        self.nu = nu

    def get_mass(self):
        """
        Return the mass coefficient.

        :returns: Mass coefficient.
        :rtype: sympy.Expr or float
        """
        return self.mass

    def get_viscosity(self):
        """
        Return the viscosity parameter.

        :returns: Kinematic viscosity.
        :rtype: sympy.Expr or float
        """
        return self.nu

    def get_u(self):
        """
        Return the prescribed velocity field.

        :returns: Velocity field.
        :rtype: sympy.Matrix
        """
        return self.u
    
    def get_vorticity(self):
        """
        Compute the vorticity field.

        .. math::
            \\boldsymbol{\\omega} = \\nabla \\times \\mathbf{u}

        :returns: Vorticity vector.
        :rtype: sympy.Matrix
        """
        return self.curl(self.u)

    def get_rhs(self):
        """
        Compute the forcing term corresponding to the manufactured solution.

        The right-hand side is constructed as

        .. math::

            \\mathbf{f} =
            m \\mathbf{u}
            + \\nu \\nabla \\times (\\nabla \\times \\mathbf{u})
            + \\nabla p.

        :returns: Symbolic right-hand side vector.
        :rtype: sympy.Matrix
        """
        return  self.mass*self.u + self.nu*self.curl(self.curl(self.u)) + self.grad(self.p)

class ExactManipulationsSpaceTime(ExactManipulationsSpace):
    """
    Extension of :class:`ExactManipulationsSpace` to space–time settings.

    This class adds symbolic time differentiation to the spatial
    vector calculus operators provided by the base class.
    """
    
    def __init__(self, coords, t):
        """
        Initialize the space–time symbolic manipulation space.

        :param coords: Spatial coordinate symbols (e.g. ``[x, y, z]``).
        :type coords: list[sympy.Symbol]
        :param t: Time symbol.
        :type t: sympy.Symbol
        """
        super().__init__(coords)
        self.t = t

    def time_derivative(self, u, t):
        """
        Compute the time derivative of a symbolic expression.

        .. math::
            \\frac{\\partial u}{\\partial t}

        :param u: Symbolic expression or vector field.
        :type u: sympy.Expr or sympy.Matrix
        :param t: Time variable.
        :type t: sympy.Symbol
        :returns: Time derivative of ``u``.
        :rtype: sympy.Expr or sympy.Matrix
        """
        return sp.diff(u,t)


class IBVPNavierStokes(ExactManipulationsSpaceTime):
    """
    Symbolic representation of an initial-boundary value problem (IBVP)
    for the incompressible Navier–Stokes equations in 3D.

    The problem has the form

    .. math::

        \\partial_t \\mathbf{u}
        + (\\mathbf{u} \\cdot \\nabla) \\mathbf{u}
        - \\nu \\Delta \\mathbf{u}
        + \\nabla p
        = \\mathbf{f},

    with prescribed initial condition

    .. math::

        \\mathbf{u}(\\cdot, 0) = \\mathbf{u}_0,

    and boundary condition

    .. math::

        \\mathbf{u}|_{\\partial \\Omega} = \\mathbf{u}_\\text{boundary}.

    :param u_init: Initial velocity field (3×1 vector).
    :type u_init: sympy.Matrix
    :param nu: Kinematic viscosity.
    :type nu: float
    :param coords: Spatial coordinate symbols (e.g. ``[x, y, z]``).
    :type coords: list[sympy.Symbol]
    :param t: Time symbol.
    :type t: sympy.Symbol
    :param u_boundary: Prescribed boundary velocity.
                       Defaults to zero.
    :type u_boundary: sympy.Matrix
    :param force: External forcing term.
                  Defaults to zero.
    :type force: sympy.Matrix
    :raises ValueError: If ``u_init`` is not a 3×1 vector.
    """
    def __init__(self, u_init: sp.Matrix, nu: float, coords, t, u_boundary = sp.Matrix([0,0,0]), force = sp.Matrix([0, 0, 0])): 
        """
        Initialize the Navier–Stokes IBVP.
        """       
        expected_rows, expected_cols = 3, 1
        if (u_init.rows, u_init.cols) != (expected_rows, expected_cols):
            raise ValueError(
                f"u\_init must be {expected_rows}x{expected_cols} (a 3-dimensional vector), "
                f"but got {u.rows}x{u.cols}"
            )
        super().__init__(coords,t)
        self.nu = nu
        self.u_init = u_init
        self.u_boundary = u_boundary
        self.force = force
        self.vorticity_init = self.curl(u_init)

    def get_u_init(self):
        """
        Return the initial velocity field.

        :returns: Initial velocity.
        :rtype: sympy.Matrix
        """
        return self.u_init

    def get_u_boundary(self):
        """
        Return the prescribed boundary velocity.

        :returns: Boundary velocity.
        :rtype: sympy.Matrix
        """
        return self.u_boundary

    def get_vorticity_init(self):
        """
        Return the initial vorticity field.

        .. math::
            \\boldsymbol{\\omega}_0 = \\nabla \\times \\mathbf{u}_0

        :returns: Initial vorticity.
        :rtype: sympy.Matrix
        """
        return self.vorticity_init
    
    def get_viscosity(self):
        """
        Return the viscosity parameter.

        :returns: Kinematic viscosity.
        :rtype: float
        """
        return self.nu
    
    def get_force(self):
        """
        Return the external forcing term.

        :returns: Forcing vector.
        :rtype: sympy.Matrix
        """
        return self.force
    
class IBVPNavierStokesSolution(IBVPNavierStokes):
    """
    Symbolic exact solution of the incompressible Navier–Stokes IBVP.

    This class represents a fully specified velocity–pressure pair
    :math:`(\\mathbf{u}, p)` depending on space and time. The initial
    condition is automatically extracted as

    .. math::

        \\mathbf{u}_0 = \\mathbf{u}(\\cdot, 0).

    The class inherits the Navier–Stokes IBVP structure from
    :class:`IBVPNavierStokes`.

    :param u: Time-dependent velocity field.
    :type u: sympy.Matrix
    :param p: Time-dependent pressure field.
    :type p: sympy.Expr
    :param nu: Kinematic viscosity.
    :type nu: float
    :param coords: Spatial coordinate symbols (e.g. ``[x, y, z]``).
    :type coords: list[sympy.Symbol]
    :param t: Time symbol.
    :type t: sympy.Symbol
    :param u_boundary: Prescribed boundary velocity.
                       Defaults to zero.
    :type u_boundary: sympy.Matrix
    :param force: External forcing term.
                  Defaults to zero.
    :type force: sympy.Matrix
    """
    def __init__(self, u: sp.Matrix, p: sp.Expr, nu: float, coords, t, u_boundary = sp.Matrix([0,0,0]), force = sp.Matrix([0,0,0])):
        """
        Initialize the symbolic Navier–Stokes solution.
        """
        self.t = t
        self.u = u
        self.p = p
        super().__init__(u.subs(self.t,0), nu, coords, t, u_boundary, force)

    def get_p(self):
        """
        Return the pressure field.

        :returns: Pressure.
        :rtype: sympy.Expr
        """
        return self.p

    def get_u(self):
        """
        Return the velocity field.

        :returns: Velocity field.
        :rtype: sympy.Matrix
        """
        return self.u
    
    def get_vorticity(self):
        """
        Compute the vorticity of the velocity field.

        .. math::
            \\boldsymbol{\\omega} = \\nabla \\times \\mathbf{u}

        :returns: Vorticity vector.
        :rtype: sympy.Matrix
        """
        return self.curl(self.u)
    
class manufacturedNavierStokes(IBVPNavierStokesSolution):
    """
    Manufactured solution for the incompressible Navier–Stokes equations.

    Given symbolic velocity and pressure fields :math:`(\\mathbf{u}, p)`,
    this class constructs the corresponding forcing term so that
    :math:`(\\mathbf{u}, p)` is an exact solution of

    .. math::

        \\partial_t \\mathbf{u}
        + (\\mathbf{u} \\cdot \\nabla) \\mathbf{u}
        + \\nabla p
        - \\nu \\Delta \\mathbf{u}
        = \\mathbf{f}.

    The right-hand side is computed symbolically.

    :param u: Time-dependent velocity field.
    :type u: sympy.Matrix
    :param p: Time-dependent pressure field.
    :type p: sympy.Expr
    :param nu: Kinematic viscosity.
    :type nu: float
    :param coords: Spatial coordinate symbols.
    :type coords: list[sympy.Symbol]
    :param t: Time symbol.
    :type t: sympy.Symbol
    """
    def __init__(self, u: sp.Matrix, p: sp.Expr, nu: float, coords, t):
        """
        Initialize the manufactured Navier–Stokes solution.

        The forcing term is computed automatically so that
        the provided velocity and pressure satisfy the
        Navier–Stokes equations.
        """
        super().__init__(u.subs(t,0), p, nu, coords, t)
        super().__init__(u.subs(t,0), p, nu, coords, t, u, self.navier_stokes_rhs(u,p,nu,t))
        

    def navier_stokes_rhs(self, u, p, nu,t):
        """
        Compute the Navier–Stokes forcing term corresponding to
        a manufactured solution.

        The forcing is constructed as

        .. math::

            \\mathbf{f}
            =
            \\partial_t \\mathbf{u}
            + (\\mathbf{u} \\cdot \\nabla) \\mathbf{u}
            + \\nabla p
            - \\nu \\Delta \\mathbf{u}.

        :param u: Velocity field.
        :type u: sympy.Matrix
        :param p: Pressure field.
        :type p: sympy.Expr
        :param nu: Kinematic viscosity.
        :type nu: float
        :param t: Time symbol.
        :type t: sympy.Symbol
        :returns: Symbolic forcing term.
        :rtype: sympy.Matrix
        """
        dudt = self.time_derivative(u,t)
        conv = self.convective_term(u)
        grad_p = self.grad(p)
        lap_u = self.laplacian_vector(u)
        return dudt + conv + grad_p - nu * lap_u



class SimulationHelper:
    """
    Base helper class for running simulation sweeps and managing configuration I/O.

    This class provides utilities to:
    - convert SymPy expressions to C-style strings for code generation,
    - generate per-(order, refinement) JSON config files,
    - run all generated configs locally via an external executable,
    - submit/run configs on the Euler cluster via SSH.

    Subclasses must implement :meth:`base_config_file`.

    :param name: Simulation/benchmark name used for config and output folder names.
    :type name: str
    """

    def __init__(self, name: str):
        """
        Initialize the helper.

        :param name: Simulation/benchmark name.
        :type name: str
        """
        self.name = name

    def expr_to_c(self,expr):
        """
        Convert a SymPy expression to a compact C expression string.

        The conversion uses :func:`sympy.printing.ccode.ccode` and then rewrites
        coordinate symbols ``x0, x1, x2`` to array access ``x[0], x[1], x[2]``.

        :param expr: SymPy expression to convert.
        :type expr: sympy.Expr
        :returns: C expression string with ``x[i]`` indexing and no spaces.
        :rtype: str
        """
        s = ccode(expr)  # e.g. uses M_PI, pow, etc.
        s = s.replace('x0', 'x[0]').replace('x1', 'x[1]').replace('x2', 'x[2]')
        s = s.replace(' ', '')  # remove spaces to match your style
        return s

    def sp_vector_to_str(self,expr):
        """
        Convert a SymPy vector expression into C assignment statements.

        Produces a single string of the form::

            out[0] = <expr0>; out[1] = <expr1>; out[2] = <expr2>;

        :param expr: Vector-valued SymPy expression.
        :type expr: sympy.Matrix
        :returns: Concatenated C assignment statements.
        :rtype: str
        """
        vec = [self.expr_to_c(comp) for comp in expr]
        out_str = ""
        for idx, comp in enumerate(vec):
            out_str = out_str + "out["+str(idx) +"] = " + str(comp) + ";"
        return out_str

    def generate_config_files(self, T: float, dt: Callable[[int,int],float], refinements: Callable[int,Iterable[int]], orders: List[int], tol=1e-8):
        """
        Generate JSON configuration files for a convergence/sweep study.

        Config files are written to ``./data/config/<name>/``. Any existing
        directory at that location is removed and recreated.

        For each ``order`` in ``orders`` and each ``refinement`` in
        ``refinements(order)``, this method writes a file named::

            <name>_conv_order<order>_ref<refinement>.json

        :param T: Final simulation time.
        :type T: float
        :param dt: Timestep selection rule.
                   Signature: ``dt(order: int, refinement: int) -> float``.
        :type dt: collections.abc.Callable[[int, int], float]
        :param refinements: Refinement selection rule.
                            Signature: ``refinements(order: int) -> Iterable[int]``.
        :type refinements: collections.abc.Callable[[int], collections.abc.Iterable[int]]
        :param orders: Polynomial orders to run.
        :type orders: list[int]
        :param tol: Solver tolerance written to the config file.
        :type tol: float
        :returns: None
        :rtype: None
        """
        directory = Path("./data/config/"+self.name)
        
        shutil.rmtree(directory, ignore_errors=True)

        # Recreate directory
        directory.mkdir(parents=True, exist_ok=True)
  
        for order in orders:
            for refinement in refinements(order):
                filename_no_extension = self.name + "_conv_order"+str(order)+"_ref"+str(refinement)
                config = self.base_config_file()
                config["outputfile"] = self.name + "/" + filename_no_extension
                config["dt"] = dt(order,refinement)
                config["T"] = T
                config["refinements"] = refinement 
                config["order"] = order
                config["tol"] = tol

                filename = filename_no_extension + ".json"
                p = directory / Path(filename)

                with p.open("w") as f:
                    json.dump(config, f, indent=4)

    def base_config_file(self) -> dict:
        """
        Return a base configuration dictionary used by :meth:`generate_config_files`.

        Subclasses must override this method.

        :raises NotImplementedError: Always, unless overridden in a subclass.
        :returns: Base configuration dictionary.
        :rtype: dict
        """
        raise NotImplementedError("Error: base_config_file() was not implemented for the class SimulationHelper!")


    def run_convergence(self, T: float, dt: Callable[[int,int],float], refinements: Callable[int,Iterable[int]], orders: List[int], tol=1e-8):
        """
        Generate configs and run all cases locally.

        This is a convenience wrapper calling :meth:`generate_config_files`
        followed by :meth:`run_all_configs`.

        :param T: Final simulation time.
        :type T: float
        :param dt: Timestep rule ``dt(order, refinement) -> float``.
        :type dt: collections.abc.Callable[[int, int], float]
        :param refinements: Refinement rule ``refinements(order) -> Iterable[int]``.
        :type refinements: collections.abc.Callable[[int], collections.abc.Iterable[int]]
        :param orders: Polynomial orders to run.
        :type orders: list[int]
        :param tol: Solver tolerance written to the config file.
        :type tol: float
        :returns: None
        :rtype: None
        """
        self.generate_config_files(T,dt,refinements,orders,tol)
        self.run_all_configs()

    def run_all_configs(self):
        """
        Run all generated configuration files locally using an external executable.

        Expects configuration files in::

            ./data/config/<name>/

        Writes outputs to::

            ./out/data/<name>/

        Uses the executable::

            ./build/MEHCscheme

        :raises RuntimeError: If the executable cannot be found or executed.
        :returns: None
        :rtype: None
        """
        EXECUTABLE = "./build/MEHCscheme"
        config_directory = "./data/config/" + self.name +"/"
        out_directory = "./out/data/" + self.name + "/"
        os.makedirs(out_directory, exist_ok=True)

        files = os.listdir(config_directory)
        print(files)

        for file in files:
            print("file: ", file)
            # pass the option flag and the config path as separate list items
            cmd = [EXECUTABLE, "-c", config_directory + str(file)]
            print("Running:", " ".join(cmd))

            try:
                result = subprocess.run(
                    cmd,
                    text=True,
                    check=False,
                )
            except FileNotFoundError as e:
                raise RuntimeError(f"Failed to run {EXECUTABLE}: {e}")

    def run_remote_command(self, client: paramiko.SSHClient, command: str) -> str:
        """
        Run a command on a connected SSH client and return stdout.

        :param client: Connected SSH client.
        :type client: paramiko.SSHClient
        :param command: Command string to run remotely.
        :type command: str
        :returns: Command stdout (decoded).
        :rtype: str
        :raises RuntimeError: If the command cannot be started, the connection drops,
                              or the remote command returns a non-zero exit code.
        """
        try:
            # 1) Try to start the command
            stdin, stdout, stderr = client.exec_command(command)
        except SSHException as e:
            # Problems starting the command (channel allocation, etc.)
            raise RuntimeError(f"Failed to start remote command {command!r}: {e}") from e

        try:
            # 2) Wait for it to finish and read output
            exit_status = stdout.channel.recv_exit_status()  # this can raise on disconnect

            out = stdout.read().decode(errors="replace")
            err = stderr.read().decode(errors="replace")
        except (SSHException, ChannelException, socket.error, socket.timeout) as e:
            # Network dropped, channel died, etc.
            raise RuntimeError(f"Connection error while running {command!r}: {e}") from e

        # 3) Remote command finished but may have failed
        if exit_status != 0:
            raise RuntimeError(
                f"Remote command {command!r} failed with exit status {exit_status}.\n"
                f"stderr: {err.strip()}"
            )

        return out

    def run_all_configs_euler(self, time="24:00:00", mempercpu="512G", cpuspertask="1"):        
        """
        Submit all generated configuration files to the Euler cluster via SSH.

        This method:
        - connects to ``euler.ethz.ch`` using an SSH key,
        - updates/builds the project on the cluster,
        - uploads config files to the remote config directory,
        - submits one SLURM job per config via ``sbatch``.

        :param time: SLURM wall time limit (e.g. ``"24:00:00"``).
        :type time: str
        :param mempercpu: SLURM memory per CPU (e.g. ``"512G"``).
        :type mempercpu: str
        :param cpuspertask: SLURM CPUs per task (e.g. ``"1"``).
        :type cpuspertask: str
        :returns: None
        :rtype: None
        :raises RuntimeError: If remote commands fail (via internal helpers).
        """
        hostname = "euler.ethz.ch"
        username = "wtonnon"
        key_path = os.path.expanduser("~/.ssh/id_ed25519") 
        #key = paramiko.RSAKey.from_private_key_file(key_path)
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            client.connect(
                hostname=hostname,
                username=username,
                key_filename=key_path,
                port=22,
                timeout=10,
            )

            stdin, stdout, stderr = client.exec_command("cd ~")
            stdin, stdout, stderr = client.exec_command("ls -a")

            def exec(command: str):
                stdin, stdout, stderr = client.exec_command(command)
                if stdout.channel.recv_exit_status():
                    print(stdout.read().decode())
                    print(stderr.read().decode())
                    print("Error: Command \"" + command + "\" on client-side (Euler) failed!")
                    sys.exit(1)
                return stdin, stdout, stderr

            stdin, stdout, stderr = exec("cd dualfieldmfem && pwd && git pull --ff-only")
            print(stdout.read().decode())
            exec("cd dualfieldmfem/build && make")
            print(stderr.read().decode())

            EXECUTABLE = "./build/MEHCscheme"
            config_directory = "data/config/" + self.name +"/"
            out_directory = "out/data/" + self.name + "/"
            
            exec("cd dualfieldmfem && [ -d \"./" + config_directory + "\" ] && echo \"exists\" || mkdir "+config_directory)
            exec("cd dualfieldmfem && [ -d \"./out/data\" ] && echo \"exists\" || mkdir \"./out/data\"")
            exec("cd dualfieldmfem && [ -d \"./" + out_directory + "\" ] && echo \"exists\" || mkdir "+out_directory)

            files = os.listdir(config_directory)
            print(files)

            sftp = client.open_sftp()
            for file in files:
                local_path = "./" + config_directory + file
                remote_path = "/cluster/home/wtonnon/dualfieldmfem/" + config_directory + file
                sftp.put(local_path,remote_path)
                print("cd dualfieldmfem && sbatch --cpus-per-task="+cpuspertask+" --time="+time+" --mem-per-cpu="+mempercpu+" --wrap=\"./build/MEHCscheme -c /cluster/home/wtonnon/dualfieldmfem/" + config_directory + file +"\"")
                exec("cd dualfieldmfem && sbatch --cpus-per-task="+cpuspertask+" --time="+time+" --mem-per-cpu="+mempercpu+" --wrap=\"./build/MEHCscheme -c /cluster/home/wtonnon/dualfieldmfem/" + config_directory + file +"\"")
        finally:
            client.close()

class NavierStokesSimulationHelper(SimulationHelper):
    """
    Simulation helper specialized for Navier--Stokes IBVP data.

    This subclass converts symbolic Navier--Stokes problem data (initial/boundary
    conditions, forcing, and optional exact solutions) into a JSON configuration
    dictionary understood by the external solver.

    :param exact_equations: Navier--Stokes IBVP data container providing viscosity,
                            forcing, initial conditions, and optionally boundary/exact data.
    :type exact_equations: IBVPNavierStokes
    :param name: Simulation/benchmark name used for config and output folder names.
    :type name: str
    :param mesh: Path to the MFEM mesh file.
    :type mesh: str
    :param visualisation: Visualisation flag forwarded to the solver.
    :type visualisation: int
    :param printlevel: Verbosity level forwarded to the solver.
    :type printlevel: int
    """

    def __init__(self, exact_equations: IBVPNavierStokes, name: str, mesh = "./extern/mfem/data/ref-cube.mesh", visualisation = 1, printlevel=0):
        """
        Initialize the Navier--Stokes simulation helper.

        :param exact_equations: Navier--Stokes IBVP data container.
        :type exact_equations: IBVPNavierStokes
        :param name: Simulation/benchmark name.
        :type name: str
        :param mesh: Path to the MFEM mesh file.
        :type mesh: str
        :param visualisation: Visualisation flag.
        :type visualisation: int
        :param printlevel: Verbosity level.
        :type printlevel: int
        """
        super().__init__(name)
        self.exact_equations = exact_equations
        self.mesh = mesh
        self.visualisation = visualisation
        self.printlevel = printlevel

    def base_config_file(self):
        """
        Build the base JSON configuration dictionary for the external solver.

        The returned dictionary always includes:

        - mesh path, solver type, visualisation, printlevel
        - viscosity
        - forcing term
        - initial velocity and initial vorticity

        Additionally, the following fields are included when available:

        - ``boundary_data_u`` if boundary velocity is provided
        - ``exact_data_u`` and ``exact_data_w`` if exact solution fields are provided

        :returns: Configuration dictionary to be written as JSON.
        :rtype: dict
        """
        config = {
            "mesh": self.mesh,
            "solver": "GMRES",
            "visualisation": self.visualisation,
            "printlevel": self.printlevel,
            "viscosity": self.exact_equations.get_viscosity(),
            "force_data": self.sp_vector_to_str(self.exact_equations.get_force()),
            "initial_data_u": self.sp_vector_to_str(self.exact_equations.get_u_init()),
            "initial_data_w": self.sp_vector_to_str(self.exact_equations.get_vorticity_init()),
        }

        if isinstance(self.exact_equations,IBVPNavierStokes):
            config["boundary_data_u"] = self.sp_vector_to_str(self.exact_equations.get_u_boundary())
        else:
            print("Warning: boundary data for u is not given. We continue without!")

        if isinstance(self.exact_equations,IBVPNavierStokesSolution):
            config["exact_data_u"] = self.sp_vector_to_str(self.exact_equations.get_u())
            config["exact_data_w"] = self.sp_vector_to_str(self.exact_equations.get_vorticity())
        else:
            print("Warning: exact solutions for u and w are not given. We continue without!")

        return config

class StokesSimulationHelper(SimulationHelper):
    """
    Simulation helper specialized for manufactured Stokes problems.

    This subclass converts symbolic manufactured Stokes data into a JSON
    configuration dictionary understood by the external solver.

    :param exact_equations: Manufactured Stokes data container providing mass,
                            viscosity, forcing, and exact velocity.
    :type exact_equations: manufacturedStokes
    :param name: Simulation/benchmark name used for config and output folder names.
    :type name: str
    """

    def __init__(self, exact_equations: manufacturedStokes, name: str):
        """
        Initialize the Stokes simulation helper.

        :param exact_equations: Manufactured Stokes data container.
        :type exact_equations: manufacturedStokes
        :param name: Simulation/benchmark name.
        :type name: str
        """
        super().__init__(name)
        self.exact_equations = exact_equations

    def base_config_file(self):
        """
        Build the base JSON configuration dictionary for the external Stokes solver.

        The returned dictionary includes:

        - mesh path (default MFEM reference cube)
        - solver type, visualisation, and printlevel
        - mass and viscosity parameters
        - forcing term (right-hand side)
        - exact velocity solution

        :returns: Configuration dictionary to be written as JSON.
        :rtype: dict
        """
        config = {
            "mesh": "./extern/mfem/data/ref-cube.mesh",
            "solver": "GMRES",
            "visualisation": 0,
            "printlevel": 1,
            "mass": self.exact_equations.get_mass(),
            "viscosity": self.exact_equations.get_viscosity(),
            "force_data": self.sp_vector_to_str(self.exact_equations.get_rhs()),
            "exact_data_u": self.sp_vector_to_str(self.exact_equations.get_u()),
        }

        return config

class SimulationDataProcessor:
    """
    Collect and post-process simulation output data.

    This class scans solver output files in ``out/data/<name>/`` and produces
    convergence plots (error vs refinement) and optional conservation plots.
    It can also pull result files from the Euler cluster via SFTP.

    :param name: Simulation/benchmark name used to locate input/output folders.
    :type name: str
    """

    def __init__(self, name: str):
        """
        Initialize the data processor.

        :param name: Simulation/benchmark name.
        :type name: str
        """
        self.name = name
        self.data = None
        self.error_columns = ["u1_err_L2"]
        self.cons_columns = []

    def collect_data(self):
        """
        Collect error data from solver output CSV files.

        This scans files matching::

            out/data/<name>/<name>_conv_order*_ref*_vars.csv

        Each file is parsed and the *last row* is used (no time matching).
        Errors are read from columns listed in :attr:`error_columns`.

        The returned structure is grouped by polynomial order:

        - ``data[order]["refs"]``: 1D array of refinement levels
        - ``data[order]["errs"][col]``: 1D array of error values for column ``col``

        :returns: Collected data grouped by order.
        :rtype: dict[int, dict[str, object]]
        :raises ValueError: If any required error column is missing in a CSV file.
        """
        pattern = re.compile(self.name + r"_conv_order(\d+)_ref(\d+)_vars\.csv")

        self.data = {}

        for fname in glob.glob("out/data/" + self.name + "/" + self.name + "_conv_order*_ref*_vars.csv"):
            base = os.path.basename(fname)
            m = pattern.match(base)
            if not m:
                continue

            order = int(m.group(1))
            ref = int(m.group(2))

            df = pd.read_csv(fname)
            print("len(df) = " + str(len(df)))
            if len(df) == 0:
                print(f"Skipping empty file: {fname}")
                continue

            # Validate error columns exist
            for error_column in self.error_columns:
                if error_column not in df.columns:
                    raise ValueError(f"'{error_column}' column not found in {fname}")

            # Always take the last row
            row = df.iloc[-1]

            errs = {col: float(row[col]) for col in self.error_columns}

            if order not in self.data:
                self.data[order] = []
            self.data[order].append((ref, errs))

        # Convert lists to sorted arrays
        for order in list(self.data.keys()):
            if not self.data[order]:
                continue
            self.data[order].sort(key=lambda x: x[0])  # sort by refinement level
            refs = np.array([r for r, _ in self.data[order]], dtype=float)

            temp_dict = {"refs": refs, "errs": {}}
            for error_column in self.error_columns:
                temp_dict["errs"][error_column] = np.array(
                    [e[error_column] for _, e in self.data[order]],
                    dtype=float,
                )

            self.data[order] = temp_dict

        return self.data


    def add_reference_triangles(self, ax, order, x_anchor, y_anchor):
        """
        Add a dotted reference line indicating expected convergence order.

        The reference indicator is anchored at ``(x_anchor, y_anchor)`` and
        assumes that one refinement step corresponds to halving the mesh size.

        :param ax: Matplotlib axes to draw on.
        :type ax: matplotlib.axes.Axes
        :param order: Expected convergence order (used as exponent in :math:`O(h^p)`).
        :type order: int
        :param x_anchor: Anchor x-location (typically the last refinement index).
        :type x_anchor: float
        :param y_anchor: Anchor y-location (typically the last error value).
        :type y_anchor: float
        :returns: None
        :rtype: None
        """
        if x_anchor is None or y_anchor is None or y_anchor <= 0:
            return

        # One refinement step to the left
        x0 = x_anchor - 1.0
        x1 = x_anchor

        # First-order triangle (O(h)):
        # coarser mesh (x0) has error 2 * y_anchor
        y1 = y_anchor
        y0 = y_anchor * (2.0 ** order)

        ax.semilogy([x0, x1], [y0, y1], ":k")
    # ax.semilogy([x1, x1], [y0, y1], ":k")
        ax.text(
            x0 + 0.05,
            np.sqrt(y0 * y1),
            "O(h"+str(order)+")",
            verticalalignment="center",
            horizontalalignment="left",
        )

    def plot_convergence(self, show_plot = False, reference_order = lambda order: order):
        """
        Plot convergence curves (log-scale error vs refinement index).

        Requires :meth:`collect_data` to have been called first.

        One figure is generated per error column in :attr:`error_columns` and saved to::

            out/plots/<name>/convergence/<name>_<error_column>

        :param show_plot: If ``True``, display the plot window; otherwise close after saving.
        :type show_plot: bool
        :param reference_order: Function mapping polynomial order to reference slope order,
                                used for the :math:`O(h^p)` indicator.
                                Signature: ``reference_order(order: int) -> int``.
        :type reference_order: collections.abc.Callable[[int], int]
        :returns: None
        :rtype: None
        """
        figs = {}
        axs = {}
        for error_column in self.error_columns:
            figs[error_column], axs[error_column] = plt.subplots()

        markers = {
            1: "o-",
            2: "s-",
            3: "^-",
            4: "v-",
        }

        last_ref = None
        last_err = None

        if not self.data:
            print(self.data)
            print("Error: data array not initialized. Call collect_data() to initialize the data array!")
            exit(1)

        for order, d in sorted(self.data.items()):
            refs = d["refs"]
            errs = d["errs"]
            style = markers.get(order, "o-")
            for error_column in self.error_columns:
                last_ref = refs[-1]
                last_err = errs[error_column][-1]
                axs[error_column].semilogy(
                    refs,
                    errs[error_column],
                    style,
                    label=f"order {order}",
                    linewidth=1.5,
                    markersize=6,
                )
                self.add_reference_triangles(axs[error_column], reference_order(order), last_ref, last_err)

        for error_column in self.error_columns:
            axs[error_column].set_xlabel("Refinement level (ref index)")
            axs[error_column].set_ylabel(f"L2 error '{error_column}'")
            axs[error_column].set_title("Convergence vs mesh refinement")
            axs[error_column].grid(True, which="both", linestyle="--", linewidth=0.5)
            axs[error_column].xaxis.set_major_locator(MultipleLocator(1.0))
            axs[error_column].legend(loc="best")

            directory = "./out/plots/" +self.name + "/convergence"
            try:
                os.makedirs(directory)
            except FileExistsError:
                print("Warning: tried creating directory \"" + directory + "\", but it already exists.")

            # Add O(h) and O(h^2) reference triangles, anchored at last point
            # Save figure
            outname = (directory + "/" + self.name + "_" + error_column)
            figs[error_column].tight_layout()
            figs[error_column].savefig(outname, dpi=300)
            print(f"Saved figure to '{outname}'")

            if show_plot:
                plt.show()
            else:
                plt.close(figs[error_column])

    def plot_conserved_variables(self):
        """
        Plot conserved/diagnostic quantities over time from solver CSV output.

        This scans the same output files as :meth:`collect_data` and, for each
        column listed in :attr:`cons_columns`, produces a time series plot saved to::

            out/plots/<name>/conservation/<name>_<column>_order<k>_ref<r>.png

        Requires that the CSV files contain a ``time_full`` column.

        :returns: None
        :rtype: None
        :raises ValueError: If ``time_full`` or any column in :attr:`cons_columns`
                            is missing in a CSV file.
        """
        directory = "./out/plots/" +self.name + "/conservation"
        try:
            os.makedirs(directory)
        except FileExistsError:
            print("Warning: tried creating directory \"" + directory + "\", but it already exists.")

        pattern = re.compile(self.name + r"_conv_order(\d+)_ref(\d+)_vars\.csv")

        # data[order] will store list of (ref, err)
        self.data = {}

        for fname in glob.glob("out/data/" + self.name + "/" + self.name + "_conv_order*_ref*_vars.csv"):
            base = os.path.basename(fname)
            m = pattern.match(base)
            if not m:
                continue

            order = int(m.group(1))
            ref = int(m.group(2))

            # Read CSV
            df = pd.read_csv(fname)

            if "time_full" not in df.columns:
                raise ValueError(f"'time_full' column not found in {fname}")
            
            for cons_column in self.cons_columns:
                if cons_column not in df.columns:
                    raise ValueError(f"'{cons_column}' column not found in {fname}")
                
                fig, ax =plt.subplots()
                ax.plot(np.array(df["time_full"]),np.array(df[cons_column]))
                ax.set_xlabel("time_full [s]")
                ax.set_ylabel(cons_column)
                ax.grid()
                plt.savefig(directory + "/" + self.name+"_" + cons_column +"_order"+str(order)+"_ref"+str(ref)+".png" )
                plt.close(fig)

    def pull_data_from_euler(self):
        """
        Pull solver output CSV files from the Euler cluster via SFTP.

        For each config file found in::

            data/config/<name>/

        this method attempts to download the corresponding output file::

            out/data/<name>/<config_basename>_vars.csv

        from the remote path under ``/cluster/home/<user>/dualfieldmfem/`` into the
        local ``out/data/<name>/`` directory.

        Missing files are skipped with a warning.

        :returns: None
        :rtype: None
        """
        hostname = "euler.ethz.ch"
        username = "wtonnon"
        key_path = os.path.expanduser("~/.ssh/id_ed25519") 

        #key = paramiko.RSAKey.from_private_key_file(key_path)
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            client.connect(
                hostname=hostname,
                username=username,
                key_filename=key_path,
                port=22,
                timeout=10,
            )

            EXECUTABLE = "./build/MEHCscheme"
            config_directory = "data/config/" + self.name +"/"
            out_directory = "out/data/" + self.name + "/"

            try:
                files = os.listdir(config_directory)
            except:
                print("Error: directory " + config_directory + " not found! Cannot continue.")
                sys.exit(1)
            print(files)


            try:
                os.makedirs(out_directory)
            except FileExistsError:
                print("Warning: tried creating directory \"" + out_directory + "\", but it already exists.")

            sftp = client.open_sftp()
            for config_file in files:
                out_file, ext = os.path.splitext(config_file)
                out_file = out_file + "_vars.csv"
                local_path = "./" + out_directory + out_file
                remote_path = "/cluster/home/wtonnon/dualfieldmfem/" + out_directory + out_file
                try:
                    sftp.get(remote_path,local_path)
                except FileNotFoundError:
                    print("Warning: "+ remote_path + " or " + local_path  + " on remote or local machine, respectively, not found. Skipping file..")
        finally:
            client.close()


