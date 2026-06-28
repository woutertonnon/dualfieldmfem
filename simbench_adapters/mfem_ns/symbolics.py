"""Lightweight symbolic containers used by benchmark helpers.

This module provides a minimal subset of the symbolic API previously exposed by
`scripts/SimIO.py`, enough for benchmark definitions and config generation.
"""

from __future__ import annotations

import sympy as sp


class _ExactManipulationsSpace:
    """Basic vector-calculus operations on symbolic fields."""

    def __init__(self, coords):
        """Create operator space from coordinate symbols, e.g. `[x0, x1]` (2D)
        or `[x0, x1, x2]` (3D). The spatial dimension is inferred from the
        number of coordinate symbols, so the same helpers serve both cases."""
        self.coords = coords

    @property
    def dim(self):
        """Spatial dimension inferred from the number of coordinate symbols."""
        return len(self.coords)

    def grad(self, scalar_field):
        """Return symbolic gradient of a scalar field."""
        return sp.Matrix([sp.diff(scalar_field, c) for c in self.coords])

    def curl(self, u):
        """Return the symbolic curl of a vector field.

        In 3D this is the usual 3-component vector. In 2D the curl of a planar
        field ``(u1, u2)`` is the scalar ``d(u2)/dx - d(u1)/dy``; we return it
        as a 1-component matrix so it can be emitted as a (scalar) vorticity
        snippet. The single-field semi-Lagrangian solver does not consume the
        vorticity, but the config schema still expects the field to be present.
        """
        if self.dim == 2:
            x, y = self.coords
            u1, u2 = u[0], u[1]
            return sp.Matrix([sp.diff(u2, x) - sp.diff(u1, y)])
        x, y, z = self.coords
        u1, u2, u3 = u
        w1 = sp.Add(sp.diff(u3, y), sp.Mul(-1, sp.diff(u2, z)))
        w2 = sp.Add(sp.diff(u1, z), sp.Mul(-1, sp.diff(u3, x)))
        w3 = sp.Add(sp.diff(u2, x), sp.Mul(-1, sp.diff(u1, y)))
        return sp.Matrix([w1, w2, w3])

    def laplacian_vector(self, vec_field):
        """Return component-wise Laplacian of a vector field."""
        n = vec_field.rows
        lap = sp.Matrix.zeros(n, 1)
        for i in range(n):
            terms = [sp.diff(vec_field[i], c, 2) for c in self.coords]
            lap[i] = sp.Add(*terms)
        return lap

    def convective_term(self, u):
        """Return rotational-form convection ``omega x u`` (``= -u x curl(u)``).

        The C++ Navier--Stokes operators discretize convection in rotational
        form via ``(curl(u_prev) x u, v)``. Manufactured forcing must use the
        same identity to remain consistent with that operator.

        In 2D the vorticity is the scalar ``omega = d(u2)/dx - d(u1)/dy`` and
        ``omega x u = (-omega*u2, omega*u1)``; in 3D we use the vector cross
        product directly.
        """
        if self.dim == 2:
            x, y = self.coords
            u1, u2 = u[0], u[1]
            omega = sp.diff(u2, x) - sp.diff(u1, y)
            return sp.Matrix([-omega * u2, omega * u1])
        return -u.cross(self.curl(u))

    def time_derivative(self, u, t):
        """Return time derivative of a symbolic expression or vector field."""
        return sp.diff(u, t)


class IBVPNavierStokes(_ExactManipulationsSpace):
    """Container for Navier--Stokes initial/boundary/forcing data."""

    def __init__(
        self,
        u_init: sp.Matrix,
        nu: float,
        coords,
        t,
        u_boundary=sp.Matrix([0, 0, 0]),
        force=sp.Matrix([0, 0, 0]),
        lid_attributes=None,
    ):
        """Initialize Navier--Stokes problem data used for config generation."""
        super().__init__(coords)
        self.t = t
        self.nu = nu
        self.u_init = u_init
        self.u_boundary = u_boundary
        self.force = force
        self.vorticity_init = self.curl(u_init)
        self.lid_attributes = lid_attributes

    def get_u_init(self):
        """Return initial velocity field."""
        return self.u_init

    def get_u_boundary(self):
        """Return prescribed boundary velocity field."""
        return self.u_boundary

    def get_vorticity_init(self):
        """Return initial vorticity field `curl(u_init)`."""
        return self.vorticity_init

    def get_viscosity(self):
        """Return kinematic viscosity."""
        return self.nu

    def get_force(self):
        """Return forcing term."""
        return self.force


class IBVPNavierStokesSolution(IBVPNavierStokes):
    """Navier--Stokes data container with exact solution fields `(u, p)`."""

    def __init__(
        self,
        u: sp.Matrix,
        p: sp.Expr,
        nu: float,
        coords,
        t,
        u_init=sp.Matrix([0, 0, 0]),
        u_boundary=sp.Matrix([0, 0, 0]),
        force=sp.Matrix([0, 0, 0]),
    ):
        """Initialize solution container and inherited problem data."""
        self.u = u
        self.p = p
        super().__init__(u_init, nu, coords, t, u_boundary=u_boundary, force=force)

    def get_p(self):
        """Return exact pressure field."""
        return self.p

    def get_u(self):
        """Return exact velocity field."""
        return self.u

    def get_vorticity(self):
        """Return exact vorticity field `curl(u)`."""
        return self.curl(self.u)


class ManufacturedNavierStokes(IBVPNavierStokesSolution):
    """Manufactured exact Navier--Stokes solution with auto-generated force.

    Given symbolic ``u(x,t)`` and ``p(x,t)``, this class computes

    ``f = du/dt + (u dot grad)u + grad(p) - nu*Delta(u)``

    so that ``(u, p)`` satisfies the incompressible Navier--Stokes equations.
    """

    def __init__(self, u: sp.Matrix, p: sp.Expr, nu: float, coords, t):
        force = self.navier_stokes_rhs(u, p, nu, coords, t)
        super().__init__(
            u=u,
            p=p,
            nu=nu,
            coords=coords,
            t=t,
            u_init=u.subs(t, 0),
            u_boundary=u,
            force=force,
        )

    @staticmethod
    def navier_stokes_rhs(u: sp.Matrix, p: sp.Expr, nu: float, coords, t) -> sp.Matrix:
        """Build forcing term for a manufactured Navier--Stokes pair ``(u, p)``."""
        ops = _ExactManipulationsSpace(coords)
        dudt = ops.time_derivative(u, t)
        conv = ops.convective_term(u)
        grad_p = ops.grad(p)
        lap_u = ops.laplacian_vector(u)
        return dudt + conv + grad_p - nu * lap_u
