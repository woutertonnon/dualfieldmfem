"""Translation utilities from canonical schema to legacy solver JSON.

Existing C++ executables consume a flat JSON contract. This module maps the
canonical v1 schema to that legacy format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import apply_defaults_v1, validate_case_v1


def to_legacy_flat(case: dict[str, Any]) -> dict[str, Any]:
    """Translate canonical v1 case into legacy flat JSON format.

    The legacy format matches the current C++ executable contract.

    Parameters
    ----------
    case
        Canonical v1 case dictionary.

    Returns
    -------
    dict[str, Any]
        Legacy flat config dictionary.
    """
    validate_case_v1(case)
    c = apply_defaults_v1(case)

    solver = c["solver"]
    params = solver["parameters"]
    funcs = solver["functions"]
    runtime = c["runtime"]
    time = c["time"]
    discr = c["discretization"]
    output = c["output"]

    legacy: dict[str, Any] = {
        "mesh": output["mesh"],
        "solver": solver["linear_solver"],
        "visualisation": runtime["visualisation"],
        "printlevel": runtime["printlevel"],
        "outputfile": output["outputfile"],
        "order": discr["order"],
        "refinements": discr["refinements"],
        "tol": runtime["tol"],
    }

    family = str(solver.get("family", "")).lower()
    if "stokes" in family and "navier" not in family:
        legacy["mass"] = params["mass"]
        legacy["viscosity"] = params["viscosity"]
        legacy["force_data"] = funcs["force_data"]
        if "exact_data_u" in funcs:
            legacy["exact_data_u"] = funcs["exact_data_u"]
    else:
        legacy["dt"] = time["dt"]
        legacy["T"] = time["T"]
        legacy["viscosity"] = params["viscosity"]
        legacy["force_data"] = funcs["force_data"]
        legacy["initial_data_u"] = funcs["initial_data_u"]
        legacy["initial_data_w"] = funcs["initial_data_w"]
        legacy["boundary_data_u"] = funcs["boundary_data_u"]
        if "exact_data_u" in funcs:
            legacy["exact_data_u"] = funcs["exact_data_u"]
        if "exact_data_w" in funcs:
            legacy["exact_data_w"] = funcs["exact_data_w"]

        boundary = solver.get("boundary", {})
        if "lid_attributes" in boundary:
            legacy["lid_attributes"] = boundary["lid_attributes"]

    return legacy


def write_legacy_config(case: dict[str, Any], path: str | Path) -> Path:
    """Translate and write a legacy config JSON file.

    Parameters
    ----------
    case
        Canonical v1 case dictionary.
    path
        Output JSON path.

    Returns
    -------
    pathlib.Path
        Path to the written config file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    legacy = to_legacy_flat(case)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(legacy, f, indent=4)
    return out_path
