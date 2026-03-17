
#!/usr/bin/env python3
import sympy as sp
import numpy as np
from numpy.polynomial.legendre import leggauss

# ----------------------------
# Symbolic setup (only to build integrand)
# ----------------------------
x, y, z = sp.symbols("x y z", real=True)

def curl3(w):
    w1, w2, w3 = w
    return sp.Matrix([
        sp.diff(w3, y) - sp.diff(w2, z),
        sp.diff(w1, z) - sp.diff(w3, x),
        sp.diff(w2, x) - sp.diff(w1, y),
    ])

def boundary_integral_unit_cube_gauss(u, v, theta=1.0, Cw=10.0, h=1.0, order=10):
    """
    Float64 Gauss–Legendre tensor-product quadrature on each face of [0,1]^3.

    Computes:
      ∫_{∂Ω} (n×curl(u))·v ds + theta ∫_{∂Ω} (n×curl(v))·u ds + ∫_{∂Ω} (Cw/h) (u×n)·(v×n) ds
    """

    u = sp.Matrix(u)
    v = sp.Matrix(v)
    cu = curl3(u)
    cv = curl3(v)

    n1, n2, n3 = sp.symbols("n1 n2 n3", real=True)
    n = sp.Matrix([n1, n2, n3])

    integrand_sym = (n.cross(cu)).dot(v) + sp.Float(theta) * (n.cross(cv)).dot(u) + sp.Float(Cw / h) * (u.cross(n)).dot(v.cross(n))
    integrand_sym = sp.simplify(integrand_sym)

    # Lambdify to NumPy for fast float64 evaluation
    f = sp.lambdify((x, y, z, n1, n2, n3), integrand_sym, modules="numpy")

    # 1D Gauss–Legendre on [0,1]: map from [-1,1] -> [0,1]
    t, w = leggauss(order)                # t in [-1,1]
    a = 0.5 * (t + 1.0)                   # a in [0,1]
    wa = 0.5 * w                          # scale weights by (b-a)/2 = 1/2

    # 2D tensor grid on [0,1]^2
    A, B = np.meshgrid(a, a, indexing="ij")         # shape (order, order)
    W2 = np.outer(wa, wa)                           # shape (order, order)

    def integrate_face_x(x0, normal):
        nx, ny, nz = normal
        vals = f(x0, A, B, nx, ny, nz)              # y=A, z=B
        return float(np.sum(vals * W2))

    def integrate_face_y(y0, normal):
        nx, ny, nz = normal
        vals = f(A, y0, B, nx, ny, nz)              # x=A, z=B
        return float(np.sum(vals * W2))

    def integrate_face_z(z0, normal):
        nx, ny, nz = normal
        vals = f(A, B, z0, nx, ny, nz)              # x=A, y=B
        return float(np.sum(vals * W2))

    total = 0.0
    total += integrate_face_x(0.0, (-1.0, 0.0, 0.0))
    total += integrate_face_x(1.0, ( 1.0, 0.0, 0.0))
    total += integrate_face_y(0.0, ( 0.0,-1.0, 0.0))
    total += integrate_face_y(1.0, ( 0.0, 1.0, 0.0))
    total += integrate_face_z(0.0, ( 0.0, 0.0,-1.0))
    total += integrate_face_z(1.0, ( 0.0, 0.0, 1.0))

    return total


if __name__ == "__main__":
    u = sp.Matrix([
        sp.exp(x - 2*y + z) + sp.sin(2*sp.pi*x)*sp.cos(sp.pi*z) + x*y*(1 - z),
        x**2*sp.sin(sp.pi*y) + sp.cos(2*sp.pi*z)*(y - z) + sp.exp(-x*z),
        sp.sin(sp.pi*x*y) + z**2*sp.cos(2*sp.pi*y) + (x - y)*sp.exp(z)
    ])

    v = sp.Matrix([
        sp.cos(sp.pi*x)*sp.exp(y - z) + x*(1 - x)*y + sp.sin(2*sp.pi*z),
        sp.sin(2*sp.pi*x*z) + sp.exp(-y) + (y - sp.Rational(1,2))**3,
        sp.cos(2*sp.pi*y*z) + sp.exp(x*y) - z*(1 - z)
    ])

    for refinements in range(0, 6):
        h = 0.5**refinements
        val = boundary_integral_unit_cube_gauss(u, v, theta=-1.0, Cw=100.0, h=h, order=12)
        print(f"refinement = {refinements}, h = {h:.6g}, Boundary integral value = {val:.16g}")
