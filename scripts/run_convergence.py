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
    nu=0.01
    name = "TaylorGreen"

    # 3D velocity field u(x,t)
    k = 2*sp.pi
    F = sp.exp(-2.*k*k*nu*t)
    u1 = sp.sin(k*x0) * sp.cos(k*x1)*F
    u2 = -sp.cos(k * x0)*sp.sin(k*x1)*F
    u3 = 0
    u = sp.Matrix([u1, u2, u3])
    force = sp.Matrix([0,0,0])

    # Pressure (here zero)
    p = 0
    coords = [x0, x1, x2]

    if value=="run":
        testSimulationHelper = SimIO.SimulationHelper(SimIO.InitialConditionAndForceAndSolution(u,p,force,nu,coords,t),name)
        #testSimulationHelper.run_convergence(1.,0.02,[0,1,2,3],[1])
        testSimulationHelper.generate_config_files(.5,0.02,[0,1,2,3], [1,2])
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
    nu=0.1
    name = "TaylorGreen3Dnew"

    # 3D velocity field u(x,t)
    k = 2.*sp.pi
    u1 = (sp.sin(k*x0-5.*sp.pi/6.)*sp.cos(k*x1-sp.pi/6.)*sp.sin(k*x2) - sp.cos(k*x2-5.*sp.pi/6.)*sp.sin(k*x0-sp.pi/6.)*sp.sin(k*x1))*sp.exp(-3*nu*k*k*t)
    u2 = (sp.sin(k*x1-5.*sp.pi/6.)*sp.cos(k*x2-sp.pi/6.)*sp.sin(k*x0) - sp.cos(k*x0-5.*sp.pi/6.)*sp.sin(k*x1-sp.pi/6.)*sp.sin(k*x2))*sp.exp(-3*nu*k*k*t)
    u3 = (sp.sin(k*x2-5.*sp.pi/6.)*sp.cos(k*x0-sp.pi/6.)*sp.sin(k*x1) - sp.cos(k*x1-5.*sp.pi/6.)*sp.sin(k*x2-sp.pi/6.)*sp.sin(k*x0))*sp.exp(-3*nu*k*k*t)
    u = sp.Matrix([u1, u2, u3])
    force = sp.Matrix([0,0,0])

    # Pressure (here zero)
    p = 0
    coords = [x0, x1, x2]

    if value=="run":
        testSimulationHelper = SimIO.SimulationHelper(SimIO.InitialConditionAndForceAndSolution(u,p,force,nu,coords,t),name)
        #testSimulationHelper.run_convergence(1.,0.02,[0,1,2,3],[1,2])
        testSimulationHelper.generate_config_files(.5,0.02,[0,1,2,3], [1,2])
        testSimulationHelper.run_all_configs_euler()
    else:
        SDP = SimIO.SimulationDataProcessor(name)
        SDP.pull_data_from_euler()
        SDP.collect_data(.5,0.1)
        SDP.plot_convergence(1.)
        SDP.plot_conserved_variables()

# Convergence tests dual field paper
if True:
    x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
    nu=1.
    name = "DualFieldPaperConvergenceTest"

    # 3D velocity field u(x,t)
    k = 2.*sp.pi
    u1 = (2-t)*sp.cos(2.*sp.pi*x2)
    u2 = (1+t)*sp.sin(2.*sp.pi*x2)
    u3 = (1-t)*sp.sin(2.*sp.pi*x0)
    u = sp.Matrix([u1, u2, u3])

    # Pressure (here zero)
    p = 0
    coords = [x0, x1, x2]

    if value=="run":
        testSimulationHelper = SimIO.SimulationHelper(SimIO.ManufacturedEquations(u,p,nu,coords,t),name)
        #testSimulationHelper.run_convergence(1.,0.02,[0,1,2],[1])
        testSimulationHelper.generate_config_files(.5,0.02,[0,1,2,3], [1,2])
        testSimulationHelper.run_all_configs_euler()
    else:
        SDP = SimIO.SimulationDataProcessor(name)
        SDP.pull_data_from_euler()
        SDP.collect_data(.5,0.1)
        SDP.plot_convergence(1.)
        SDP.plot_conserved_variables()

# Taylor Green 3D with no Force
if False:

    x0, x1, x2, t = sp.symbols('x0 x1 x2 t', real=True)
    nu=0
    name = "TaylorGreen3DnoForce"

    # 3D velocity field u(x,t)
    u1 = sp.sin(2.*sp.pi*x2) + sp.cos(2.*sp.pi*x1)
    u2 = sp.sin(2.*sp.pi*x0) + sp.cos(2.*sp.pi*x2)
    u3 = sp.sin(2.*sp.pi*x1) + sp.cos(2.*sp.pi*x0)
    u = sp.Matrix([u1, u2, u3])
    force = sp.Matrix([0,0,0])

    # Pressure (here zero)
    p = 0
    coords = [x0, x1, x2]

    if value=="run":
        testSimulationHelper = SimIO.SimulationHelper(SimIO.InitialConditionAndForceAndSolution(u,p,force,nu,coords,t),name)
        #testSimulationHelper.run_convergence(1.,0.02,[0,1,2],[1])
        testSimulationHelper.generate_config_files(.5,0.02,[0,1,2,3,4], [1])
        testSimulationHelper.run_all_configs_euler()
    else:
        SDP = SimIO.SimulationDataProcessor(name)
        SDP.pull_data_from_euler()
        SDP.collect_data(.5,0.1)
        SDP.plot_convergence(.5)
        SDP.plot_conserved_variables()