import SimIO
import sympy as sp
import sys

if len(sys.argv) > 1:
    value = sys.argv[1]
    print("Argument:", value)

x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
nu=0

# 3D velocity field u(x,t)
u1 = sp.cos(2 * sp.pi * x1)
u2 = sp.sin(2 * sp.pi * x2)
u3 = sp.sin(2 * sp.pi * x0)
u = sp.Matrix([u1, u2, u3])

# Pressure (here zero)
p = 0
coords = [x0, x1, x2]

if value == "run":
    testSimulationHelper = SimIO.SimulationHelper(SimIO.ExactEquations(u,p,nu,coords,t,False),"test")
    #testSimulationHelper.run_convergence(1.,0.02,[0,1,2],[1])
    testSimulationHelper.generate_config_files(1.,0.02,[0,1,2], [1])
    testSimulationHelper.run_all_configs_euler()
else:
    SDP = SimIO.SimulationDataProcessor("test")
    SDP.pull_data_from_euler()
    SDP.collect_data(1.,0.1)
    SDP.plot_convergence(1.)
    SDP.plot_conserved_variables()