"""Schema defaults and validation for canonical simbench cases.

The helpers in this module operate on the v1 canonical case structure used by
`simbench_core` and adapter packages.
"""

from __future__ import annotations

import copy
import math
from typing import Any


def _require_dict(obj: Any, name: str) -> dict[str, Any]:
    """Return *obj* if it is a dict, otherwise raise ValueError."""
    if not isinstance(obj, dict):
        raise ValueError(f"{name} must be an object/dict")
    return obj


def _require_type(obj: Any, name: str, expected: type | tuple[type, ...]) -> None:
    """Ensure *obj* is an instance of *expected*.

    Parameters
    ----------
    obj
        Value to validate.
    name
        Field name used in error messages.
    expected
        Expected Python type or tuple of types.
    """
    if not isinstance(obj, expected):
        expected_name = (
            ", ".join(t.__name__ for t in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise ValueError(f"{name} must be of type {expected_name}")


def _require_finite_nonnegative_number(value: Any, name: str) -> None:
    """Ensure *value* is a finite numeric value >= 0."""
    _require_type(value, name, (int, float))
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative")


def apply_defaults_v1(case: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied canonical v1 case with defaults applied.

    Defaults mirror the current C++ config defaults where applicable.

    Parameters
    ----------
    case
        Canonical v1 case dictionary.

    Returns
    -------
    dict[str, Any]
        Deep copy of ``case`` with missing default fields populated.
    """
    c = copy.deepcopy(case)

    c.setdefault("schema_version", 1)
    c.setdefault("runner", {})
    c.setdefault("metadata", {})

    runtime = c.setdefault("runtime", {})
    runtime.setdefault("tol", 1e-8)
    runtime.setdefault("visualisation", 0)
    runtime.setdefault("printlevel", 0)

    discretization = c.setdefault("discretization", {})
    discretization.setdefault("order", 1)
    discretization.setdefault("refinements", 10)

    time = c.setdefault("time", {})
    time.setdefault("dt", 0.02)
    time.setdefault("T", 1.0)

    output = c.setdefault("output", {})
    output.setdefault("mesh", "")
    output.setdefault("outputfile", "")

    solver = c.setdefault("solver", {})
    solver.setdefault("family", "")
    solver.setdefault("variant", "")
    solver.setdefault("linear_solver", "GMRES")
    solver.setdefault("parameters", {})
    solver.setdefault("functions", {})
    solver.setdefault("boundary", {})

    parameters = solver["parameters"]
    parameters.setdefault("viscosity", 0.0)
    if "stokes" in str(solver.get("family", "")).lower():
        parameters.setdefault("mass", 0.0)

    return c


def validate_case_v1(case: dict[str, Any]) -> None:
    """Validate canonical case schema v1.

    Raises ValueError on invalid input.

    Parameters
    ----------
    case
        Canonical v1 case dictionary.
    """
    c = apply_defaults_v1(case)

    _require_type(c.get("schema_version"), "schema_version", int)
    if c["schema_version"] != 1:
        raise ValueError("schema_version must be 1")

    _require_type(c.get("case_id"), "case_id", str)
    if not c["case_id"].strip():
        raise ValueError("case_id must be a non-empty string")

    solver = _require_dict(c.get("solver"), "solver")
    _require_type(solver.get("family"), "solver.family", str)
    _require_type(solver.get("variant"), "solver.variant", str)
    _require_type(solver.get("linear_solver"), "solver.linear_solver", str)

    params = _require_dict(solver.get("parameters"), "solver.parameters")
    funcs = _require_dict(solver.get("functions"), "solver.functions")
    boundary = _require_dict(solver.get("boundary"), "solver.boundary")

    discr = _require_dict(c.get("discretization"), "discretization")
    _require_type(discr.get("order"), "discretization.order", int)
    _require_type(discr.get("refinements"), "discretization.refinements", int)
    if discr["order"] < 0 or discr["refinements"] < 0:
        raise ValueError("discretization.order and discretization.refinements must be >= 0")

    time = _require_dict(c.get("time"), "time")
    _require_finite_nonnegative_number(time.get("dt"), "time.dt")
    _require_finite_nonnegative_number(time.get("T"), "time.T")

    runtime = _require_dict(c.get("runtime"), "runtime")
    _require_finite_nonnegative_number(runtime.get("tol"), "runtime.tol")
    _require_type(runtime.get("visualisation"), "runtime.visualisation", int)
    _require_type(runtime.get("printlevel"), "runtime.printlevel", int)

    output = _require_dict(c.get("output"), "output")
    _require_type(output.get("mesh"), "output.mesh", str)
    _require_type(output.get("outputfile"), "output.outputfile", str)
    if not output["mesh"].strip():
        raise ValueError("output.mesh must be a non-empty string")
    if not output["outputfile"].strip():
        raise ValueError("output.outputfile must be a non-empty string")

    family = solver["family"].lower()
    if "stokes" in family and "navier" not in family:
        required_funcs = {"force_data"}
        if "exact_data_u" in funcs:
            _require_type(funcs["exact_data_u"], "solver.functions.exact_data_u", str)
        if "mass" not in params:
            raise ValueError("solver.parameters.mass is required for stokes family")
        _require_finite_nonnegative_number(params.get("mass"), "solver.parameters.mass")
        _require_finite_nonnegative_number(params.get("viscosity"), "solver.parameters.viscosity")
    else:
        required_funcs = {
            "force_data",
            "initial_data_u",
            "initial_data_w",
            "boundary_data_u",
        }
        _require_finite_nonnegative_number(params.get("viscosity"), "solver.parameters.viscosity")

    for key in required_funcs:
        if key not in funcs:
            raise ValueError(f"solver.functions.{key} is required for family '{solver['family']}'")

    for key, val in funcs.items():
        _require_type(val, f"solver.functions.{key}", str)

    if "lid_attributes" in boundary:
        attrs = boundary["lid_attributes"]
        if not isinstance(attrs, list):
            raise ValueError("solver.boundary.lid_attributes must be a list of positive integers")
        for i, attr in enumerate(attrs):
            if not isinstance(attr, int) or attr <= 0:
                raise ValueError(
                    "solver.boundary.lid_attributes must contain only positive integers "
                    f"(bad value at index {i}: {attr!r})"
                )
