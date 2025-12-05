import SimIO
import sympy as sp
import sys

if len(sys.argv) > 1:
    value = sys.argv[1]
    print("Argument:", value)

if False:
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
        testSimulationHelper = SimIO.SimulationHelper(SimIO.ManufacturedEquations(u,p,nu,coords,t),"test")
        #testSimulationHelper.run_convergence(1.,0.02,[0,1,2],[1])
        testSimulationHelper.generate_config_files(1.,0.02,[0,1,2,3], [1])
        testSimulationHelper.run_all_configs_euler()
    else:
        SDP = SimIO.SimulationDataProcessor("test")
        SDP.pull_data_from_euler()
        SDP.collect_data(1.,0.1)
        SDP.plot_convergence(1.)
        SDP.plot_conserved_variables()


# Taylor Green
if False:

    x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
    nu=0
    name = "TaylorGreen"

    # 3D velocity field u(x,t)
    u1 = sp.sin(2*sp.pi*x0) * sp.cos(2*sp.pi*x1)
    u2 = -sp.cos(2 * sp.pi * x0)*sp.sin(2*sp.pi*x1)
    u3 = 0
    u = sp.Matrix([u1, u2, u3])

    # Pressure (here zero)
    p = 0
    coords = [x0, x1, x2]

    if value=="run":
        testSimulationHelper = SimIO.SimulationHelper(SimIO.ManufacturedEquations(u,p,nu,coords,t),name)
        #testSimulationHelper.run_convergence(1.,0.02,[0,1,2],[1])
        testSimulationHelper.generate_config_files(.5,0.02,[0,1,2,3,4,5], [1])
        testSimulationHelper.run_all_configs_euler()
    else:
        SDP = SimIO.SimulationDataProcessor(name)
        SDP.pull_data_from_euler()
        SDP.collect_data(.5,0.1)
        SDP.plot_convergence(1.)
        SDP.plot_conserved_variables()


# Taylor Green 3D
if False:

    x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
    nu=0
    name = "TaylorGreen3D"

    # 3D velocity field u(x,t)
    u1 = sp.sin(2*sp.pi*x0) * sp.cos(2*sp.pi*x1) * sp.cos(2*sp.pi*x2)
    u2 = sp.cos(2*sp.pi*x0) * sp.sin(2*sp.pi*x1) * sp.cos(2*sp.pi*x2)
    u3 = -2.*sp.cos(2*sp.pi*x0) * sp.cos(2*sp.pi*x1) * sp.sin(2*sp.pi*x2)
    u = sp.Matrix([u1, u2, u3])

    # Pressure (here zero)
    p = 0
    coords = [x0, x1, x2]

    if value=="run":
        testSimulationHelper = SimIO.SimulationHelper(SimIO.ManufacturedEquations(u,p,nu,coords,t),name)
        #testSimulationHelper.run_convergence(1.,0.02,[0,1,2],[1])
        testSimulationHelper.generate_config_files(.5,0.02,[0,1,2,3,4,5], [1])
        testSimulationHelper.run_all_configs_euler()
    else:
        SDP = SimIO.SimulationDataProcessor(name)
        SDP.pull_data_from_euler()
        SDP.collect_data(.5,0.1)
        SDP.plot_convergence(1.)
        SDP.plot_conserved_variables()

# Taylor Green 3D with no Force
if True:

    x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
    nu=0
    name = "TaylorGreen3DnoForce"

    # 3D velocity field u(x,t)
    u1 = sp.sin(2*sp.pi*x0) * sp.cos(2*sp.pi*x1) * sp.cos(2*sp.pi*x2)
    u2 = sp.cos(2*sp.pi*x0) * sp.sin(2*sp.pi*x1) * sp.cos(2*sp.pi*x2)
    u3 = -2.*sp.cos(2*sp.pi*x0) * sp.cos(2*sp.pi*x1) * sp.sin(2*sp.pi*x2)
    u = sp.Matrix([u1, u2, u3])
    force = sp.Matrix([0,0,0])

    # Pressure (here zero)
    p = 0
    coords = [x0, x1, x2]

    if value=="run":
        testSimulationHelper = SimIO.SimulationHelper(SimIO.InitialConditionAndForceAndSolution(u,p,force,nu,coords,t),name)
        #testSimulationHelper.run_convergence(1.,0.02,[0,1,2],[1])
        testSimulationHelper.generate_config_files(.5,0.02,[0,1,2,3,4,5], [1])
        testSimulationHelper.run_all_configs_euler()
    else:
        SDP = SimIO.SimulationDataProcessor(name)
        SDP.pull_data_from_euler()
        SDP.collect_data(.5,0.1)
        SDP.plot_convergence(1.)
        SDP.plot_conserved_variables()