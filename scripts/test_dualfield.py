import SimIO
import sympy as sp
import sys 

run = True
euler = False

#x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
#nu = .01
#name = "SingleFieldVorticityPrevTimestep"
#
## 3D velocity field u(x,t)
#k = 1.
#psi = sp.sin(x0)*sp.cosh(x1)#x0*x0*x0*x1 -x0*x1*x1*x1
#Fx = x0**2*(1-x0)**2
#Fy = x1**2*(1-x1)**2
#Fz = x2**2*(1-x2)**2
#
#u1 = sp.sin(k*x0)*sp.cos(k*x1)*sp.exp(-2*nu*k*k*t)#-2*x0**2*(x0-1)**2 * x1*(2*x1-1)*(x1-1) * x2*(1-x2)
#u2 = -sp.cos(k*x0)*sp.sin(k*x1)*sp.exp(-2*nu*k*k*t)# 2*x1**2*(x1-1)**2 * x0*(2*x0-1)*(x0-1) * x2*(1-x2)
##u1 = -2*x0**2*(x0-1)**2 * x1*(2*x1-1)*(x1-1) * x2*(1-x2)
##u2 =  2*x1**2*(x1-1)**2 * x0*(2*x0-1)*(x0-1) * x2*(1-x2)
#u3 = sp.Integer(0)#-sp.cos(k*x0)*sp.cos(k*x1)*sp.sin(k*x2)
#
#u = sp.Matrix([u1, u2, u3])
#force = sp.Matrix([0, 0, 0])
#
#div_u = sp.diff(u1, x0) + sp.diff(u2, x1) + sp.diff(u3, x2)
#
#assert sp.factor(div_u) == 0   # or: assert sp.simplify(div_u) == 0
#
## Pressure (here zero)
#p = 0
#coords = [x0, x1, x2]
#
#if run:
#    testSimulationHelper = SimIO.NavierStokesSimulationHelper(SimIO.IBVPNavierStokesSolution(u,p,nu,coords,t),name)
#    if not euler:
#        testSimulationHelper.run_convergence(1.,lambda order, refinements: 0.04*(.5**(order*refinements)),lambda order: range(6-order-1),[1,2],tol=1e-7)
#    else:
#        testSimulationHelper.generate_config_files(1.,lambda order, refinements: 0.04*(.5**(order*refinements)),lambda order: range(6-order),[1,2],tol=1e-7)
#        testSimulationHelper.run_all_configs_euler()
#
#else:
#    SDP = SimIO.SimulationDataProcessor(name)
#    if euler:
#        SDP.pull_data_from_euler()
#    SDP.collect_data()
#    SDP.plot_convergence()
#
#x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
#nu = .01
#name = "3DTaylorGreen"
#
## 3D velocity field u(x,t)
#k = 1.
#psi = sp.sin(x0)*sp.cosh(x1)#x0*x0*x0*x1 -x0*x1*x1*x1
#Fx = x0**2*(1-x0)**2
#Fy = x1**2*(1-x1)**2
#Fz = x2**2*(1-x2)**2
#
#u1 = 2*sp.sin(k*x0)*sp.cos(k*x1)*sp.cos(k*x2)#-2*x0**2*(x0-1)**2 * x1*(2*x1-1)*(x1-1) * x2*(1-x2)
#u2 = -sp.cos(k*x0)*sp.sin(k*x1)*sp.cos(k*x2)# 2*x1**2*(x1-1)**2 * x0*(2*x0-1)*(x0-1) * x2*(1-x2)
#u3 = -sp.cos(k*x0)*sp.cos(k*x1)*sp.sin(k*x2)
#
#u = sp.Matrix([u1, u2, u3])
#force = sp.Matrix([0, 0, 0])
#
#div_u = sp.diff(u1, x0) + sp.diff(u2, x1) + sp.diff(u3, x2)
#
#assert sp.factor(div_u) == 0   # or: assert sp.simplify(div_u) == 0
#
## Pressure (here zero)
#p = 0
#coords = [x0, x1, x2]
#
#if run:
#    testSimulationHelper = SimIO.NavierStokesSimulationHelper(SimIO.IBVPNavierStokes(u,nu,coords,t, u),name)
#    if not euler:
#        testSimulationHelper.run_convergence(1.,lambda order, refinements: 0.04*(.5**(order*refinements)),lambda order: range(6-order-1),[1,2],tol=1e-7)
#    else:
#        testSimulationHelper.generate_config_files(1.,lambda order, refinements: 0.04*(.5**(order*refinements)),lambda order: range(6-order),[1,2],tol=1e-7)
#        testSimulationHelper.run_all_configs_euler()
#
#else:
#    SDP = SimIO.SimulationDataProcessor(name)
#    if euler:
#        SDP.pull_data_from_euler()
#    SDP.collect_data()
#    SDP.plot_convergence()
#
#

x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
nu = .01
name = "LidDrivenCavity3D"

# 3D velocity field u(x,t)
cut_off = lambda t: 2 - sp.cosh(((t-0.5)*2.)**10*1.317)
cut_off_z = lambda t: t
ub1 = cut_off(x0)*cut_off(x1)*cut_off_z(x2)
ub2 = sp.Integer(0)
ub3 = sp.Integer(0)

ub = sp.Matrix([ub1, ub2, ub3])
ui = sp.Matrix([0, 0, 0])
force = sp.Matrix([0, 0, 0])



# Pressure (here zero)
p = 0
coords = [x0, x1, x2]

if run:
    testSimulationHelper = SimIO.NavierStokesSimulationHelper(SimIO.IBVPNavierStokes(u_init=ui,nu=nu,coords=coords,t=t, u_boundary=ub),name)
    if not euler:
        testSimulationHelper.run_convergence(1.,lambda order, refinements: 0.01,lambda order: [4],[2],tol=1e-7)
    else:
        testSimulationHelper.generate_config_files(1.,lambda order, refinements: 0.04*(.5**(order*refinements)),lambda order: range(6-order),[1,2],tol=1e-7)
        testSimulationHelper.run_all_configs_euler()

else:
    SDP = SimIO.SimulationDataProcessor(name)
    if euler:
        SDP.pull_data_from_euler()
    SDP.collect_data()
    SDP.plot_convergence()
