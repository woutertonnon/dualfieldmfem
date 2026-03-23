"""Benchmark helper classes for MFEM Navier--Stokes and Stokes workflows.

These classes bridge existing benchmark scripts to the new canonical schema
and adapter translation pipeline.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import socket
from pathlib import Path
from typing import Callable, Iterable

from sympy.printing import ccode
import paramiko
from paramiko.ssh_exception import SSHException, ChannelException

from .adapter import MfemNavierStokesAdapter, MfemStokesAdapter


class _SympyCodegen:
    """Utility mixin for converting SymPy expressions to C snippets."""

    def expr_to_c(self, expr):
        """Convert a SymPy expression to compact C syntax with `x[i]` indexing."""
        s = ccode(expr)
        s = s.replace("x0", "x[0]").replace("x1", "x[1]").replace("x2", "x[2]")
        s = s.replace(" ", "")
        return s

    def sp_vector_to_str(self, expr):
        """Convert a SymPy vector to `out[i] = ...;` assignment code."""
        vec = [self.expr_to_c(comp) for comp in expr]
        out = ""
        for idx, comp in enumerate(vec):
            out += f"out[{idx}] = {comp};"
        return out


class NavierStokesBenchmarkHelper(_SympyCodegen):
    """Benchmark helper using the new canonical/adaptor framework.

    This class preserves the public methods used by existing benchmark scripts:
    - generate_config_files(...)
    - run_all_configs()
    - run_convergence(...)
    """

    def __init__(
        self,
        exact_equations,
        name: str,
        mesh: str = "./extern/mfem/data/ref-cube.mesh",
        visualisation: int = 1,
        printlevel: int = 0,
        executable: str = "./build/dualfieldnavierstokes_nitsche",
        linear_solver: str = "GMRES",
        legacy_overrides: dict | Callable[[int, int], dict] | None = None,
        legacy_remote_helper=None,
    ) -> None:
        """Create a helper for Navier--Stokes benchmark case generation and runs."""
        self.exact_equations = exact_equations
        self.name = name
        self.mesh = mesh
        self.visualisation = visualisation
        self.printlevel = printlevel
        self.linear_solver = linear_solver
        self.legacy_overrides = legacy_overrides
        self.adapter = MfemNavierStokesAdapter(executable=executable)
        self.legacy_remote_helper = legacy_remote_helper

    def _functions(self) -> dict[str, str]:
        """Build legacy function-code payload from symbolic equation data."""
        funcs = {
            "force_data": self.sp_vector_to_str(self.exact_equations.get_force()),
            "initial_data_u": self.sp_vector_to_str(self.exact_equations.get_u_init()),
            "initial_data_w": self.sp_vector_to_str(self.exact_equations.get_vorticity_init()),
            "boundary_data_u": self.sp_vector_to_str(self.exact_equations.get_u_boundary()),
        }
        if hasattr(self.exact_equations, "get_u"):
            funcs["exact_data_u"] = self.sp_vector_to_str(self.exact_equations.get_u())
        if hasattr(self.exact_equations, "get_vorticity"):
            funcs["exact_data_w"] = self.sp_vector_to_str(self.exact_equations.get_vorticity())
        return funcs

    def generate_config_files(
        self,
        T: float,
        dt: Callable[[int, int], float],
        refinements: Callable[[int], Iterable[int]],
        orders: list[int],
        tol: float = 1e-10,
    ) -> None:
        """Generate legacy-compatible JSON config files for a sweep."""
        config_dir = Path("./data/config") / self.name
        shutil.rmtree(config_dir, ignore_errors=True)
        config_dir.mkdir(parents=True, exist_ok=True)

        for order in orders:
            for refinement in refinements(order):
                stem = f"{self.name}_conv_order{order}_ref{refinement}"
                overrides = None
                if self.legacy_overrides is not None:
                    if callable(self.legacy_overrides):
                        overrides = self.legacy_overrides(int(order), int(refinement))
                    else:
                        overrides = self.legacy_overrides
                case = self.adapter.build_case(
                    case_id=stem,
                    mesh=self.mesh,
                    outputfile=f"{self.name}/{stem}",
                    viscosity=float(self.exact_equations.get_viscosity()),
                    functions=self._functions(),
                    order=int(order),
                    refinements=int(refinement),
                    dt=float(dt(order, refinement)),
                    T=float(T),
                    tol=float(tol),
                    visualisation=int(self.visualisation),
                    printlevel=int(self.printlevel),
                    linear_solver=self.linear_solver,
                    lid_attributes=getattr(self.exact_equations, "lid_attributes", None),
                    legacy_overrides=overrides,
                )
                self.adapter.write_config(case, config_dir / f"{stem}.json")

    def run_all_configs(self) -> None:
        """Run all generated configs locally with the configured executable."""
        config_dir = Path("./data/config") / self.name
        out_dir = Path("./out/data") / self.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for file_name in sorted(os.listdir(config_dir)):
            config_path = config_dir / file_name
            cmd = self.adapter.command_for(config_path)
            subprocess.run(cmd, text=True, check=False)

    def run_convergence(
        self,
        T: float,
        dt: Callable[[int, int], float],
        refinements: Callable[[int], Iterable[int]],
        orders: list[int],
        tol: float = 1e-10,
    ) -> None:
        """Convenience wrapper for config generation followed by local runs."""
        self.generate_config_files(T, dt, refinements, orders, tol)
        self.run_all_configs()

    def run_all_configs_euler(self, time: str = "24:00:00", mempercpu: str = "2048G", cpuspertask: str = "1"):
        """Submit all generated configs to Euler via SSH/SLURM."""
        if self.legacy_remote_helper is not None:
            self.legacy_remote_helper.run_all_configs_euler(
                time=time,
                mempercpu=mempercpu,
                cpuspertask=cpuspertask,
            )
            return

        hostname = "euler.ethz.ch"
        username = "wtonnon"
        key_path = os.path.expanduser("~/.ssh/id_ed25519")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        def run_remote_command(command: str):
            try:
                stdin, stdout, stderr = client.exec_command(command)
            except SSHException as exc:
                raise RuntimeError(f"Failed to start remote command {command!r}: {exc}") from exc

            try:
                exit_status = stdout.channel.recv_exit_status()
                _ = stdout.read().decode(errors="replace")
                err = stderr.read().decode(errors="replace")
            except (SSHException, ChannelException, socket.error, socket.timeout) as exc:
                raise RuntimeError(f"Connection error while running {command!r}: {exc}") from exc

            if exit_status != 0:
                raise RuntimeError(
                    f"Remote command {command!r} failed with exit status {exit_status}.\n"
                    f"stderr: {err.strip()}"
                )

        try:
            client.connect(
                hostname=hostname,
                username=username,
                key_filename=key_path,
                port=22,
                timeout=10,
            )

            run_remote_command("cd dualfieldmfem && git pull --ff-only")
            run_remote_command("cd dualfieldmfem/build && make")

            config_directory = "data/config/" + self.name + "/"
            out_directory = "out/data/" + self.name + "/"

            run_remote_command("cd dualfieldmfem && mkdir -p " + config_directory)
            run_remote_command("cd dualfieldmfem && mkdir -p out/data")
            run_remote_command("cd dualfieldmfem && mkdir -p " + out_directory)

            if self.mesh and os.path.isfile(self.mesh):
                run_remote_command("cd dualfieldmfem && mkdir -p " + os.path.dirname(self.mesh))
                sftp_mesh = client.open_sftp()
                remote_mesh_path = "/cluster/home/wtonnon/dualfieldmfem/" + self.mesh
                sftp_mesh.put(self.mesh, remote_mesh_path)
                sftp_mesh.close()

            files = os.listdir(config_directory)
            sftp = client.open_sftp()
            for file in files:
                local_path = "./" + config_directory + file
                remote_path = "/cluster/home/wtonnon/dualfieldmfem/" + config_directory + file
                sftp.put(local_path, remote_path)
                sbatch_cmd = (
                    "cd dualfieldmfem && sbatch --cpus-per-task="
                    + cpuspertask
                    + " --time="
                    + time
                    + " --mem-per-cpu="
                    + mempercpu
                    + " --wrap=\""
                    + self.adapter.executable
                    + " -c /cluster/home/wtonnon/dualfieldmfem/"
                    + config_directory
                    + file
                    + "\""
                )
                run_remote_command(sbatch_cmd)
            sftp.close()
        finally:
            client.close()


class StokesBenchmarkHelper(_SympyCodegen):
    """Benchmark helper for steady Stokes runs using canonical-to-legacy translation."""

    def __init__(
        self,
        exact_equations,
        name: str,
        mesh: str = "./extern/mfem/data/ref-cube.mesh",
        visualisation: int = 0,
        printlevel: int = 1,
        executable: str = "./build/stokes_nitsche",
        linear_solver: str = "GMRES",
        legacy_remote_helper=None,
    ) -> None:
        """Create a helper for Stokes benchmark case generation and runs."""
        self.exact_equations = exact_equations
        self.name = name
        self.mesh = mesh
        self.visualisation = visualisation
        self.printlevel = printlevel
        self.linear_solver = linear_solver
        self.adapter = MfemStokesAdapter(executable=executable)
        self.legacy_remote_helper = legacy_remote_helper

    def _functions(self) -> dict[str, str]:
        """Build legacy function-code payload for Stokes runs."""
        return {
            "force_data": self.sp_vector_to_str(self.exact_equations.get_rhs()),
            "exact_data_u": self.sp_vector_to_str(self.exact_equations.get_u()),
        }

    def generate_config_files(
        self,
        T: float,
        dt: Callable[[int, int], float],
        refinements: Callable[[int], Iterable[int]],
        orders: list[int],
        tol: float = 1e-10,
    ) -> None:
        """Generate legacy-compatible JSON config files for a Stokes sweep."""
        del T
        del dt
        config_dir = Path("./data/config") / self.name
        shutil.rmtree(config_dir, ignore_errors=True)
        config_dir.mkdir(parents=True, exist_ok=True)

        for order in orders:
            for refinement in refinements(order):
                stem = f"{self.name}_conv_order{order}_ref{refinement}"
                case = self.adapter.build_case(
                    case_id=stem,
                    mesh=self.mesh,
                    outputfile=f"{self.name}/{stem}",
                    mass=float(self.exact_equations.get_mass()),
                    viscosity=float(self.exact_equations.get_viscosity()),
                    functions=self._functions(),
                    order=int(order),
                    refinements=int(refinement),
                    tol=float(tol),
                    visualisation=int(self.visualisation),
                    printlevel=int(self.printlevel),
                    linear_solver=self.linear_solver,
                )
                self.adapter.write_config(case, config_dir / f"{stem}.json")

    def run_all_configs(self) -> None:
        """Run all generated Stokes configs locally."""
        config_dir = Path("./data/config") / self.name
        out_dir = Path("./out/data") / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for file_name in sorted(os.listdir(config_dir)):
            config_path = config_dir / file_name
            cmd = self.adapter.command_for(config_path)
            subprocess.run(cmd, text=True, check=False)

    def run_convergence(
        self,
        T: float,
        dt: Callable[[int, int], float],
        refinements: Callable[[int], Iterable[int]],
        orders: list[int],
        tol: float = 1e-10,
    ) -> None:
        """Convenience wrapper for config generation followed by local runs."""
        self.generate_config_files(T, dt, refinements, orders, tol)
        self.run_all_configs()

    def run_all_configs_euler(self, time: str = "24:00:00", mempercpu: str = "2048G", cpuspertask: str = "1"):
        """Submit all generated Stokes configs to Euler.

        Currently delegates to a supplied legacy remote helper.
        """
        if self.legacy_remote_helper is None:
            raise RuntimeError("Stokes Euler backend is not yet implemented in new helper")
        self.legacy_remote_helper.run_all_configs_euler(
            time=time,
            mempercpu=mempercpu,
            cpuspertask=cpuspertask,
        )
