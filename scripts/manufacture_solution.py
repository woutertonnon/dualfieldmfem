import sympy as sp
from sympy.printing import ccode

def grad(scalar_field, coords):
    return sp.Matrix([sp.diff(scalar_field, c) for c in coords])

def laplacian_vector(vec_field, coords):
    n = vec_field.rows
    lap = sp.Matrix.zeros(n, 1)
    for i in range(n):
        lap_i = 0
        for c in coords:
            lap_i += sp.diff(vec_field[i], c, 2)
        lap[i] = lap_i
    return lap

def convective_term(u, coords):
    n = u.rows
    conv = sp.Matrix.zeros(n, 1)
    for i in range(n):
        term = 0
        for j, c in enumerate(coords):
            term += u[j] * sp.diff(u[i], c)
        conv[i] = term
    return conv

def divergence(u, coords):
    term = 0
    for j, c in enumerate(coords):
        term += sp.diff(u[j],c)
    return term

def time_derivative(u, t):
    return sp.diff(u,t)

def navier_stokes_rhs(u, p, rho, nu, coords,t):
    """
    RHS = - (u · ∇)u - (1/rho) ∇p + nu ∇²u
    """
    dudt = time_derivative(u,t)
    conv = convective_term(u, coords)
    grad_p = grad(p, coords)
    lap_u = laplacian_vector(u, coords)
    return dudt + conv - grad_p + nu * lap_u

def curl(u, coords):
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

def expr_to_c(expr):
    """
    Convert SymPy expr to compact C expression using:
    - M_PI for pi
    - x[0], x[1], x[2] for space coords
    """
    s = ccode(expr)  # e.g. uses M_PI, pow, etc.
    s = s.replace('x0', 'x[0]').replace('x1', 'x[1]').replace('x2', 'x[2]')
    s = s.replace(' ', '')  # remove spaces to match your style
    return s

if __name__ == "__main__":
    # Coordinates and parameters
    x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
    rho, nu = sp.symbols('rho nu', positive=True)
    nu=0

    # 3D velocity field u(x,t)
    u1 = (2 - t) * sp.cos(2 * sp.pi * x1)
    u2 = (1 + t) * sp.sin(2 * sp.pi * x2)
    u3 = (1 - t) * sp.sin(2 * sp.pi * x0)
    u = sp.Matrix([u1, u2, u3])

    # Pressure (here zero)
    p = 0
    coords = [x0, x1, x2]

    assert(sp.Eq(divergence(u, coords),0))

    # RHS of Navier–Stokes (force term)
    rhs = navier_stokes_rhs(u, p, rho, nu, coords, t)

    # Vorticity w = curl(u)
    w = curl(u, coords)

    # Initial data (t = 0)
    u_init = u.subs(t, 0)
    w_init = w.subs(t, 0)



    # ---- Convert to C/JSON strings ----
    # force_data uses RHS (time-dependent)
    f0, f1, f2 = [expr_to_c(comp) for comp in rhs]
    force_data = (
        "\"force_data\": "
        f"\"out[0] = {f0};out[1] = {f1};out[2] = {f2};\","
    )

    # initial_data_u: u(x,0)
    iu0, iu1, iu2 = [expr_to_c(comp) for comp in u_init]
    initial_data_u = (
        "\"initial_data_u\": "
        f"\"out[0] = {iu0};out[1] = {iu1};out[2] = {iu2};\","
    )

    # initial_data_w: w(x,0)
    iw0, iw1, iw2 = [expr_to_c(comp) for comp in w_init]
    initial_data_w = (
        "\"initial_data_w\": "
        f"\"out[0] = {iw0};out[1] = {iw1};out[2] = {iw2};\","
    )

    # exact_data_u: u(x,t)
    eu0, eu1, eu2 = [expr_to_c(comp) for comp in u]
    exact_data_u = (
        "\"exact_data_u\": "
        f"\"out[0] = {eu0};out[1] = {eu1};out[2] = {eu2};\","
    )

    # exact_data_w: w(x,t) = curl(u)
    ew0, ew1, ew2 = [expr_to_c(comp) for comp in w]
    exact_data_w = (
        "\"exact_data_w\": "
        f"\"out[0] = {ew0};out[1] = {ew1};out[2] = {ew2};\","
    )

    # Print everything so you can copy-paste into the JSON file
    print(force_data)
    print(initial_data_u)
    print(initial_data_w)
    print(exact_data_u)
    print(exact_data_w)
