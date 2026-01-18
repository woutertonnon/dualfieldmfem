import SimIO
import sympy as sp
import sys 

x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
mass = 1.0
nu = .02
name = "StokesTest_boundary_nonperiodic"

# 3D velocity field u(x,t)
k = 2*sp.pi
psi = sp.sin(x0)*sp.cosh(x1)#x0*x0*x0*x1 -x0*x1*x1*x1
Fx = x0**2*(1-x0)**2
Fy = x1**2*(1-x1)**2
Fz = x2**2*(1-x2)**2
u1 = sp.sin(2.*sp.pi*x0)*sp.cos(2.*sp.pi*x1)#-2*x0**2*(x0-1)**2 * x1*(2*x1-1)*(x1-1) 
u2 = -sp.cos(2.*sp.pi*x0)*sp.sin(2.*sp.pi*x1)# 2*x1**2*(x1-1)**2 * x0*(2*x0-1)*(x0-1) 
u3 = sp.Float(0.)

u = sp.Matrix([u1, u2, u3])
force = sp.Matrix([0, 0, 0])

div_u = sp.diff(u1, x0) + sp.diff(u2, x1) + sp.diff(u3, x2)

assert sp.factor(div_u) == 0   # or: assert sp.simplify(div_u) == 0

# Pressure (here zero)
p = 0
coords = [x0, x1, x2]

if True:
    testSimulationHelper = SimIO.StokesSimulationHelper(SimIO.manufacturedStokes(u,p,mass,nu,coords),name)
    #testSimulationHelper.run_convergence(1.,0.02,lambda order : range(5-order),[1,2])
    testSimulationHelper.generate_config_files(1.,0.02,lambda order : range(6-order),[1,2],tol=1e-5)
    testSimulationHelper.run_all_configs_euler()

else:
    SDP = SimIO.SimulationDataProcessor(name)
    SDP.pull_data_from_euler()
    SDP.collect_data()
    SDP.plot_convergence()
