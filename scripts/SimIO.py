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

class ExactEquations:
    def __init__(self, u: sp.Matrix, p: sp.Expr, nu: float, coords, t, initial_condition_only: bool):
        expected_rows, expected_cols = 3, 1
        if (u.rows, u.cols) != (expected_rows, expected_cols):
            raise ValueError(
                f"u must be {expected_rows}x{expected_cols} (a 3-dimensional vector), "
                f"but got {u.rows}x{u.cols}"
            )
        
        self.u = u
        self.p = p
        self.nu = nu
        self.coords = coords
        self.t = t
        self.initial_condition_only = initial_condition_only

        assert(sp.Eq(self.divergence(u),0))

    def grad(self, scalar_field):
        return sp.Matrix([sp.diff(scalar_field, c) for c in self.coords])

    def laplacian_vector(self, vec_field):
        n = vec_field.rows
        lap = sp.Matrix.zeros(n, 1)
        for i in range(n):
            lap_i = 0
            for c in self.coords:
                lap_i += sp.diff(vec_field[i], c, 2)
            lap[i] = lap_i
        return lap

    def convective_term(self, u):
        n = u.rows
        conv = sp.Matrix.zeros(n, 1)
        for i in range(n):
            term = 0
            for j, c in enumerate(self.coords):
                term += u[j] * sp.diff(u[i], c)
            conv[i] = term
        return conv

    def divergence(self, u):
        term = 0
        for j, c in enumerate(self.coords):
            term += sp.diff(u[j],c)
        return term

    def time_derivative(self, u, t):
        return sp.diff(u,t)

    def navier_stokes_rhs(self, u, p, rho, nu,t):
        """
        RHS = - (u · ∇)u - (1/rho) ∇p + nu ∇²u
        """
        dudt = self.time_derivative(u,t)
        conv = self.convective_term(u)
        grad_p = self.grad(p)
        lap_u = self.laplacian_vector(u)
        return dudt + conv - grad_p + nu * lap_u

    def curl(self, u):
        """
        3D curl: w = ∇ × u
        coords: [x, y, z]
        u: Matrix([u1, u2, u3])
        """
        x, y, z = coords
        u1, u2, u3 = u
        w1 = sp.diff(u3, y) - sp.diff(u2, z)
        w2 = sp.diff(u1, z) - sp.diff(u3, x)
        w3 = sp.diff(u2, x) - sp.diff(u1, y)
        return sp.Matrix([w1, w2, w3])
    
    def get_u(self):
        if(self.initial_condition_only):
            RuntimeError("ExactEquations: Ru was requested, but only the initial condition is defined.")
        return self.u
    
    def get_vorticity(self):
        if(self.initial_condition_only):
            RuntimeError("ExactEquations: vorticity was requested, but only the initial condition is defined.")
        return self.curl(self.u)

    def get_u_init(self):
        return self.u.subs(self.t, 0)
    
    def get_vorticity_init(self):
        return self.curl(self.u).subs(self.t, 0)
    
    def get_rhs(self):
        if(self.initial_condition_only):
            RuntimeError("ExactEquations: RHS was requested, but only the initial condition is defined.")
        rhs = self.navier_stokes_rhs(self.u, self.p, self.nu,self.nu,self.t)
        return rhs
    
    def get_vorticity(self):
        return self.curl(self.u)
    
    def get_initial_condition_only(self):
        return self.initial_condition_only
    
    def get_viscosity(self):
        return self.nu


class SimulationHelper:
    def __init__(self, exact_equations: ExactEquations, name: str):
        self.exact_equations = exact_equations
        self.name = name

    def expr_to_c(self,expr):
        s = ccode(expr)  # e.g. uses M_PI, pow, etc.
        s = s.replace('x0', 'x[0]').replace('x1', 'x[1]').replace('x2', 'x[2]')
        s = s.replace(' ', '')  # remove spaces to match your style
        return s

    def generate_config_files(self, T: float, dt: float, refinements: List[int], orders: List[int], tol=1e-5):
        
        
        directory = Path("./data/config/"+self.name)
        
        shutil.rmtree(directory, ignore_errors=True)

        # Recreate directory
        directory.mkdir(parents=True, exist_ok=True)
        
        u_init = self.exact_equations.get_u_init()
        w_init = self.exact_equations.get_vorticity_init()
        force = self.exact_equations.get_rhs()
        if(not self.exact_equations.get_initial_condition_only()):
            u = self.exact_equations.get_u()
            w = self.exact_equations.get_vorticity()

        # ---- Convert to C/JSON strings ----
        # force_data uses RHS (time-dependent)
        f0, f1, f2 = [self.expr_to_c(comp) for comp in force]
        force_data = (
            f"out[0] = {f0};out[1] = {f1};out[2] = {f2};"
        )

        # initial_data_u: u(x,0)
        iu0, iu1, iu2 = [self.expr_to_c(comp) for comp in u_init]
        initial_data_u = (
            f"out[0] = {iu0};out[1] = {iu1};out[2] = {iu2};"
        )

        # initial_data_w: w(x,0)
        iw0, iw1, iw2 = [self.expr_to_c(comp) for comp in w_init]
        initial_data_w = (
            f"out[0] = {iw0};out[1] = {iw1};out[2] = {iw2};"
        )


        if(not self.exact_equations.get_initial_condition_only()):
            # exact_data_u: u(x,t)
            eu0, eu1, eu2 = [self.expr_to_c(comp) for comp in u]
            exact_data_u = (
                f"out[0] = {eu0};out[1] = {eu1};out[2] = {eu2};"
            )

            # exact_data_w: w(x,t) = curl(u)
            ew0, ew1, ew2 = [self.expr_to_c(comp) for comp in w]
            exact_data_w = (
                f"out[0] = {ew0};out[1] = {ew1};out[2] = {ew2};"
            )

        for order in orders:
            for refinement in refinements:
                filename_no_extension = self.name + "_conv_order"+str(order)+"_ref"+str(refinement)
                config = {
                    "mesh": "./extern/mfem/data/periodic-cube.mesh",
                    "outputfile": self.name + "/" + filename_no_extension,
                    "solver": "GMRES",
                    "dt": dt,
                    "T": T,
                    "refinements": refinement,
                    "order": order,
                    "visualisation": 1,
                    "printlevel": 0,
                    "viscosity": self.exact_equations.get_viscosity(),
                    "tol": tol,
                    "boundary_data_u": "out[0] = 0.;out[1]=0.;out[2]=0.;",
                    "force_data": force_data,
                    "initial_data_u": initial_data_u,
                    "initial_data_w": initial_data_w,
                    "exact_data_u": (
                        "out[0] = cos(2*M_PI*x[1]);"
                        "out[1] = sin(2*M_PI*x[2]);"
                        "out[2] = sin(2*M_PI*x[0]);"
                    ),
                    "exact_data_w": (
                        "out[0] = -2*M_PI*cos(2*M_PI*x[2]);"
                        "out[1] = -2*M_PI*cos(2*M_PI*x[0]);"
                        "out[2] =  2*M_PI*sin(2*M_PI*x[1]);"
                    ),
                }

                if(not self.exact_equations.get_initial_condition_only()):
                    config["exact_data_u"] = exact_data_u
                    config["exact_data_w"] = exact_data_w

                filename = filename_no_extension + ".json"

                p = directory / Path(filename)

                with p.open("w") as f:
                    json.dump(config, f, indent=4)

    def run_convergence(self, T: float, dt: float, refinements: List[int], orders: List[int], tol=1e-5):
        self.generate_config_files(T,dt,refinements,orders,tol)
        self.run_all_configs()

    def run_all_configs(self):
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
        """Run a command on an already-connected SSHClient and handle errors."""
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

    def run_all_configs_euler(self, time="24:00:00", mempercpu="128G", cpuspertask="1"):
        hostname = "euler.ethz.ch"
        username = "wtonnon"
        key_path = "/home/wtonnon/.ssh/id_ed25519.pub"

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

    

x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
nu=0

# 3D velocity field u(x,t)
u1 = sp.cos(2 * sp.pi * x1)
u2 = sp.sin(2 * sp.pi * x2)
u3 = sp.sin(2 * sp.pi * x0)
u = sp.Matrix([u1, u2, u3])

# Pressure (here zero)
p = 0
coords = [x0, x1, x2]

testSimulationHelper = SimulationHelper(ExactEquations(u,p,nu,coords,t,False),"initial_test2")
#testSimulationHelper.run_convergence(1.,0.02,[0,1],[1,2])
testSimulationHelper.run_all_configs_euler()