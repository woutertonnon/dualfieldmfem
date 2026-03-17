"""MFEM Navier--Stokes/Stokes adapter implementations.

Adapters build canonical cases, translate them to legacy JSON, and provide
solver command lines for execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from simbench_core import apply_defaults_v1, to_legacy_flat, validate_case_v1, write_legacy_config


class _BaseMfemAdapter:
    """Shared helper methods for MFEM solver adapters."""

    def __init__(self, executable: str) -> None:
        self.executable = executable

    def command_for(self, config_path: str | Path) -> list[str]:
        """Return the solver command for a given config file path."""
        return [self.executable, "-c", str(config_path)]

    def to_legacy(self, case: dict[str, Any]) -> dict[str, Any]:
        """Translate canonical case to legacy flat config dictionary."""
        return to_legacy_flat(case)

    def write_config(self, case: dict[str, Any], out_path: str | Path) -> Path:
        """Translate and write legacy config JSON to disk."""
        return write_legacy_config(case, out_path)


class MfemNavierStokesAdapter(_BaseMfemAdapter):
    """Adapter for MFEM Navier--Stokes executables using DualFieldConfig contract."""

    def __init__(self, executable: str = "./build/hcurl_dualfieldnavierstokes_nitsche") -> None:
        super().__init__(executable)

    def build_case(
        self,
        *,
        case_id: str,
        mesh: str,
        outputfile: str,
        viscosity: float,
        functions: dict[str, str],
        order: int,
        refinements: int,
        dt: float,
        T: float,
        tol: float = 1e-8,
        visualisation: int = 0,
        printlevel: int = 0,
        linear_solver: str = "GMRES",
        variant: str = "hcurl_dualfield_nitsche",
        lid_attributes: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a validated canonical v1 Navier--Stokes case dictionary."""
        case: dict[str, Any] = {
            "schema_version": 1,
            "case_id": case_id,
            "solver": {
                "family": "mfem_navier_stokes",
                "variant": variant,
                "linear_solver": linear_solver,
                "parameters": {
                    "viscosity": viscosity,
                },
                "functions": functions,
                "boundary": {},
            },
            "discretization": {
                "order": order,
                "refinements": refinements,
            },
            "time": {
                "dt": dt,
                "T": T,
            },
            "runtime": {
                "tol": tol,
                "visualisation": visualisation,
                "printlevel": printlevel,
            },
            "output": {
                "mesh": mesh,
                "outputfile": outputfile,
            },
            "metadata": metadata or {},
        }
        if lid_attributes is not None:
            case["solver"]["boundary"]["lid_attributes"] = lid_attributes
        validate_case_v1(case)
        return apply_defaults_v1(case)


class MfemStokesAdapter(_BaseMfemAdapter):
    """Adapter for MFEM Stokes executable using NitscheStokesConfig contract."""

    def __init__(self, executable: str = "./build/stokes_nitsche") -> None:
        super().__init__(executable)

    def build_case(
        self,
        *,
        case_id: str,
        mesh: str,
        outputfile: str,
        mass: float,
        viscosity: float,
        functions: dict[str, str],
        order: int,
        refinements: int,
        tol: float = 1e-8,
        visualisation: int = 0,
        printlevel: int = 0,
        linear_solver: str = "GMRES",
        variant: str = "hcurl_nitsche",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a validated canonical v1 Stokes case dictionary."""
        case: dict[str, Any] = {
            "schema_version": 1,
            "case_id": case_id,
            "solver": {
                "family": "mfem_stokes",
                "variant": variant,
                "linear_solver": linear_solver,
                "parameters": {
                    "mass": mass,
                    "viscosity": viscosity,
                },
                "functions": functions,
                "boundary": {},
            },
            "discretization": {
                "order": order,
                "refinements": refinements,
            },
            "time": {
                "dt": 0.0,
                "T": 0.0,
            },
            "runtime": {
                "tol": tol,
                "visualisation": visualisation,
                "printlevel": printlevel,
            },
            "output": {
                "mesh": mesh,
                "outputfile": outputfile,
            },
            "metadata": metadata or {},
        }
        validate_case_v1(case)
        return apply_defaults_v1(case)
