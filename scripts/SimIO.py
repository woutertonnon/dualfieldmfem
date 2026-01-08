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

class ExactManipulationsSpace:
    
    def __init__(self, coords):
        self.coords = coords


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
        conv = -u.cross(self.curl(u))#
        return conv

    def divergence(self, u):
        term = 0
        for j, c in enumerate(self.coords):
            term += sp.diff(u[j],c)
        return term


    def curl(self, u):
        """
        3D curl: w = ∇ × u
        coords: [x, y, z]
        u: Matrix([u1, u2, u3])
        """
        x, y, z = self.coords
        u1, u2, u3 = u
        w1 = sp.diff(u3, y) - sp.diff(u2, z)
        w2 = sp.diff(u1, z) - sp.diff(u3, x)
        w3 = sp.diff(u2, x) - sp.diff(u1, y)
        return sp.Matrix([w1, w2, w3])


class manufacturedStokes(ExactManipulationsSpace):
    def __init__(self, u, p, mass, nu, coords):
        super().__init__(coords)
        self.u = u
        self.p = p
        self.mass = mass
        self.nu = nu

    def get_mass(self):
        return self.mass

    def get_viscosity(self):
        return self.nu

    def get_u(self):
        return self.u
    
    def get_vorticity(self):
        return self.curl(self.u)

    def get_rhs(self):
        return  self.mass*self.u + self.nu*self.curl(self.curl(self.u)) + self.grad(self.p)

class ExactManipulationsSpaceTime(ExactManipulationsSpace):
    
    def __init__(self, coords, t):
        super().__init__(coords)
        self.t = t


    def time_derivative(self, u, t):
        return sp.diff(u,t)


class IBVPNavierStokes(ExactManipulationsSpace):
    def __init__(self, u_init: sp.Matrix, nu: float, coords, force = sp.Matrix([0, 0, 0])):        
        expected_rows, expected_cols = 3, 1
        if (u_init.rows, u_init.cols) != (expected_rows, expected_cols):
            raise ValueError(
                f"u\_init must be {expected_rows}x{expected_cols} (a 3-dimensional vector), "
                f"but got {u.rows}x{u.cols}"
            )
        super().__init__(coords)
        self.nu = nu
        self.u_init = u_init
        self.coords = coords
        self.force = force
        self.vorticity_init = self.curl(u_init)

    def get_u_init(self):
        return self.u_init

    def get_vorticity_init(self):
        return self.vorticity_init
    
    def get_viscosity(self):
        return self.nu
    
    def get_rhs(self):
        return self.force
    
class IBVPNavierStokesSolution(IBVPNavierStokes):
    def __init__(self, u: sp.Matrix, p: sp.Expr, nu: float, coords, t, force = sp.Matrix([0,0,0])):
        self.t = t
        self.u = u
        self.p = p
        super().__init__(u.subs(self.t,0), nu, coords, force)

    def get_p(self):
        return self.p

    def get_u(self):
        return self.u
    
    def get_vorticity(self):
        return self.curl(self.u)
    
class manufacturedNavierStokes(IBVPNavierStokesSolution):
    def __init__(self, u: sp.Matrix, p: sp.Expr, nu: float, coords, t, force = sp.Matrix([0,0,0])):
        super().__init__(u.subs(self.t,0), p, nu, coords, t, self.navier_stokes_rhs(u,p,nu,t))

    def navier_stokes_rhs(self, u, p, nu,t):
        """
        RHS = - (u · ∇)u - (1/rho) ∇p + nu ∇²u
        """
        dudt = self.time_derivative(u,t)
        conv = self.convective_term(u)
        grad_p = self.grad(p)
        lap_u = self.laplacian_vector(u)
        return dudt + conv + grad_p - nu * lap_u



class SimulationHelper:
    def __init__(self, name: str):
        self.name = name

    def expr_to_c(self,expr):
        s = ccode(expr)  # e.g. uses M_PI, pow, etc.
        s = s.replace('x0', 'x[0]').replace('x1', 'x[1]').replace('x2', 'x[2]')
        s = s.replace(' ', '')  # remove spaces to match your style
        return s

    def sp_vector_to_str(self,expr):
        vec = [self.expr_to_c(comp) for comp in expr]
        out_str = ""
        for idx, comp in enumerate(vec):
            out_str = out_str + "out["+str(idx) +"] = " + str(comp) + ";"
        return out_str

    def generate_config_files(self, T: float, dt: float, refinements: Callable[int,Iterable[int]], orders: List[int], tol=1e-5):
        directory = Path("./data/config/"+self.name)
        
        shutil.rmtree(directory, ignore_errors=True)

        # Recreate directory
        directory.mkdir(parents=True, exist_ok=True)
  
        for order in orders:
            for refinement in refinements(order):
                filename_no_extension = self.name + "_conv_order"+str(order)+"_ref"+str(refinement)
                config = self.base_config_file()
                config["outputfile"] = self.name + "/" + filename_no_extension
                config["dt"] = dt
                config["T"] = T
                config["refinements"] = refinement 
                config["order"] = order
                config["tol"] = tol

                filename = filename_no_extension + ".json"
                p = directory / Path(filename)

                with p.open("w") as f:
                    json.dump(config, f, indent=4)

    def base_config_file(self) -> dict:
        raise NotImplementedError("Error: base_config_file() was not implemented for the class SimulationHelper!")


    def run_convergence(self, T: float, dt: float, refinements: Callable[int,Iterable[int]], orders: List[int], tol=1e-5):
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

class NavierStokesSimulationHelper(SimulationHelper):
    def __init__(self, exact_equations: IBVPNavierStokes, name: str):
        super().__init__(name)
        self.exact_equations = exact_equations

    def base_config_file(self):
        config = {
            "mesh": "./extern/mfem/data/periodic-cube.mesh",
            "solver": "GMRES",
            "visualisation": 0,
            "printlevel": 0,
            "viscosity": self.exact_equations.get_viscosity(),
            "boundary_data_u": "out[0] = 0.;out[1]=0.;out[2]=0.;",
            "force_data": self.sp_vector_to_str(self.exact_equations.get_force()),
            "initial_data_u": self.sp_vector_to_str(self.exact_equations.get_u_init()),
            "initial_data_w": self.sp_vector_to_str(self.exact_equations.get_vorticity_init()),
        }

        if isinstance(config,IBVPNavierStokesSolution):
            config["exact_data_u"] = self.sp_vector_to_str(self.exact_equations.get_u())
            config["exact_data_w"] = self.sp_vector_to_str(self.exact_equations.get_vorticity())
        else:
            print("Warning: exact solutions for u and w are not given. We continue without!")

        return config

class StokesSimulationHelper(SimulationHelper):
    def __init__(self, exact_equations: manufacturedStokes, name: str):
        super().__init__(name)
        self.exact_equations = exact_equations

    def base_config_file(self):
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
    def __init__(self, name: str):
        self.name = name
        self.data = None
        self.error_columns = ["u1_err_L2"]
        self.cons_columns = []

    def collect_data(self):
        """
        Scan conv_order*_ref*_vars.csv files and collect error vs refinement level
        using the LAST row of each file (no time matching).

        Returns:
            data: dict[order] = dict with keys 'refs' (np.array) and 'errs' (dict of np.array)
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
        Add small dotted triangles indicating O(h) and O(h^2) behavior.

        The triangles are aligned so that their right vertex is exactly at
        the last point of the convergence curve: (x_anchor, y_anchor).

        We assume that going one refinement level to the left corresponds
        to doubling h, so the error increases by 2^p for order p.
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
        Make a log-linear plot of L2 error vs refinement level for each order.
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


