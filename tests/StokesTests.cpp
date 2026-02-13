#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>

#include "StokesOperators.h"
#include "mfem.hpp"

using namespace mfem;

TEST(StokesSystem, PeriodicConstantField_TransposeGivesZero)
{
    // Applies Stokes operator transpose to a constant velocity field on a periodic mesh.
    // Uses ND for velocity and CG for pressure with zero mass contribution in this configuration.
    // Expects the resulting vector to be (numerically) zero.

    const double viscosity = 1.0;
    const int refinements = 2;
    const int order = 1;
    const double tol = 1e-10;
    const std::string mesh_string = "../extern/mfem/data/periodic-cube.mesh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    GridFunction u(&ND);
    GridFunction p(&CG);

    mfem::VectorFunctionCoefficient vec1_coef(
        3, [](mfem::Vector, mfem::Vector &y) -> void
        {
            y.SetSize(3);
            y.Elem(0) = 1.0;
            y.Elem(1) = 1.0;
            y.Elem(2) = 1.0;
        });
    u.ProjectCoefficient(vec1_coef);
    p = 0.0;

    const int size_1 = u.Size() + p.Size();
    Vector x(size_1);
    x = 0.0;

    Array<int> u_dofs(u.Size()), p_dofs(p.Size());
    std::iota(u_dofs.begin(), u_dofs.end(), 0);
    std::iota(p_dofs.begin(), p_dofs.end(), u.Size());

    x.SetSubVector(u_dofs, u);
    x.SetSubVector(p_dofs, p);

    StokesSystem sys(ND, CG, 0.0, viscosity, 1.0, 100.0);

    mfem::Vector y(x.Size());
    sys.MultTranspose(x, y);

    y.Abs();
    EXPECT_NEAR(y.Max(), 0.0, tol);

    delete fec_ND;
    delete fec_CG;
}

TEST(StokesSystem, PeriodicConstantField_TransposeMatchesRHS)
{
    // Compares Stokes operator transpose applied to a constant field against an assembled RHS.
    // Uses periodic mesh with ND/CG spaces and constant forcing/traction-like data.
    // Expects componentwise equality between operator result and RHS within tolerance.

    const double viscosity = 1.0;
    const int refinements = 2;
    const int order = 1;
    const double tol = 1e-10;
    const std::string mesh_string = "../extern/mfem/data/periodic-cube.mesh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    GridFunction u(&ND);
    GridFunction p(&CG);

    mfem::VectorFunctionCoefficient vec1_coef(
        3, [](mfem::Vector, mfem::Vector &y) -> void
        {
            y.SetSize(3);
            y.Elem(0) = 1.0;
            y.Elem(1) = 1.0;
            y.Elem(2) = 1.0;
        });
    u.ProjectCoefficient(vec1_coef);
    p = 0.0;

    const int size_1 = u.Size() + p.Size();
    Vector x(size_1);
    x = 0.0;

    Array<int> u_dofs(u.Size()), p_dofs(p.Size());
    std::iota(u_dofs.begin(), u_dofs.end(), 0);
    std::iota(p_dofs.begin(), p_dofs.end(), u.Size());

    x.SetSubVector(u_dofs, u);
    x.SetSubVector(p_dofs, p);

    auto f = [](const mfem::Vector &, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = 1.0;
        y.Elem(1) = 1.0;
        y.Elem(2) = 1.0;
    };

    StokesRHS rhs(ND, CG, f, f);
    StokesSystem sys(ND, CG, 1.0, viscosity, 1.0, 100.0);

    mfem::Vector y(x.Size());
    sys.MultTranspose(x, y);

    ASSERT_EQ(y.Size(), rhs.Size());
    for (int i = 0; i < y.Size(); ++i) { EXPECT_NEAR(y[i], rhs[i], tol); }

    delete fec_ND;
    delete fec_CG;
}

TEST(StokesSystem, PeriodicConstantField_SolvesToExactConstant)
{
    // Solves the Stokes system on a periodic mesh with a constant exact velocity solution.
    // Uses GMRES on the coupled operator with RHS assembled from matching constant data.
    // Expects the computed velocity to match the exact constant field with near-zero L2 error.

    const double viscosity = 1.0;
    const int refinements = 1;
    const int order = 1;
    const double tol = 1e-10;
    const std::string mesh_string = "../extern/mfem/data/periodic-cube.mesh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    auto f = [](const mfem::Vector &, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = 1.0;
        y.Elem(1) = 1.0;
        y.Elem(2) = 1.0;
    };

    StokesSystem sys(ND, CG, 1.0, viscosity, 1.0, 100.0);
    StokesRHS rhs(ND, CG, f, f);
    StokesSolution x(ND, CG);

    auto solver = std::make_unique<mfem::GMRESSolver>();
    solver->SetAbsTol(tol);
    solver->SetKDim(3000);
    solver->SetRelTol(0.0);
    solver->SetMaxIter(10000);
    solver->SetPrintLevel(1);
    solver->SetOperator(sys);
    solver->Mult(rhs, x);

    mfem::VectorFunctionCoefficient exact_u(3, f);
    EXPECT_NEAR(x.get_u().ComputeL2Error(exact_u), 0.0, tol);

    delete fec_ND;
    delete fec_CG;
}

TEST(StokesSystem, CubeConstantField_TransposeGivesZeroVelocityBlock)
{
    // Applies Stokes operator transpose to a constant velocity field on a non-periodic cube mesh.
    // Uses ND/CG spaces with zero mass and no stabilization terms enabled in this configuration.
    // Expects the velocity block of the result to be (numerically) zero for this setup.

    const double viscosity = 100.0;
    const int refinements = 0;
    const int order = 1;
    const double tol = 1e-10;
    const std::string mesh_string = "../extern/mfem/data/ref-cube.mesh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    StokesSolution x(ND, CG);
    mfem::VectorFunctionCoefficient vec1_coef(
        3, [](mfem::Vector, mfem::Vector &y) -> void
        {
            y.SetSize(3);
            y.Elem(0) = 1.0;
            y.Elem(1) = 1.0;
            y.Elem(2) = 1.0;
        });
    x.get_u().ProjectCoefficient(vec1_coef);

    StokesSystem sys(ND, CG, 0.0, viscosity, 0.0, 0.0);

    StokesSolution y(ND, CG);
    sys.MultTranspose(x, y);

    for (auto com : y.get_u()) { EXPECT_NEAR(com, 0.0, tol); }

    delete fec_ND;
    delete fec_CG;
}

TEST(StokesSystem, CubeConstantField_TransposeMatchesRHS)
{
    // Compares Stokes operator transpose on a constant field against an assembled RHS on a cube mesh.
    // Uses ND/CG spaces with viscosity and stabilization parameters passed consistently to RHS/system.
    // Expects componentwise agreement between operator result and RHS within tolerance.

    const double viscosity = 10.0;
    const int refinements = 2;
    const int order = 1;
    const double tol = 1e-10;
    const std::string mesh_string = "../extern/mfem/data/ref-cube.mesh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    GridFunction u(&ND);
    GridFunction p(&CG);

    mfem::VectorFunctionCoefficient vec1_coef(
        3, [](mfem::Vector, mfem::Vector &y) -> void
        {
            y.SetSize(3);
            y.Elem(0) = 1.0;
            y.Elem(1) = 1.0;
            y.Elem(2) = 1.0;
        });
    u.ProjectCoefficient(vec1_coef);
    p = 0.0;

    const int size_1 = u.Size() + p.Size();
    Vector x(size_1);
    x = 0.0;

    Array<int> u_dofs(u.Size()), p_dofs(p.Size());
    std::iota(u_dofs.begin(), u_dofs.end(), 0);
    std::iota(p_dofs.begin(), p_dofs.end(), u.Size());

    x.SetSubVector(u_dofs, u);
    x.SetSubVector(p_dofs, p);

    auto f = [](const mfem::Vector &, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = 1.0;
        y.Elem(1) = 1.0;
        y.Elem(2) = 1.0;
    };

    StokesRHS rhs(ND, CG, f, f, 1.0, 100.0, viscosity);
    StokesSystem sys(ND, CG, 1.0, viscosity, 1.0, 100.0);

    StokesSolution y(ND, CG);
    sys.MultTranspose(x, y);

    ASSERT_EQ(y.Size(), rhs.Size());
    for (int i = 0; i < y.Size(); ++i) { EXPECT_NEAR(y[i], rhs[i], tol); }

    delete fec_ND;
    delete fec_CG;
}

TEST(StokesSystem, CubeConstantField_SolvesToExactConstant)
{
    // Solves the Stokes system on a cube mesh where the exact velocity is constant.
    // Uses GMRES on the coupled system with RHS assembled from the same constant field.
    // Expects the numerical velocity solution to match the exact field in L2 norm.

    const double viscosity = 1.0;
    const int refinements = 1;
    const int order = 1;
    const double tol = 1e-10;
    const std::string mesh_string = "../extern/mfem/data/ref-cube.mesh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    auto f = [](const mfem::Vector &, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = 1.0;
        y.Elem(1) = 1.0;
        y.Elem(2) = 1.0;
    };

    StokesSystem sys(ND, CG, 1.0, viscosity, 1.0, 100.0);
    StokesRHS rhs(ND, CG, f, f);
    StokesSolution x(ND, CG);

    auto solver = std::make_unique<mfem::GMRESSolver>();
    solver->SetAbsTol(tol);
    solver->SetKDim(3000);
    solver->SetRelTol(0.0);
    solver->SetMaxIter(10000);
    solver->SetPrintLevel(1);
    solver->SetOperator(sys);
    solver->Mult(rhs, x);

    mfem::VectorFunctionCoefficient exact_u(3, f);
    EXPECT_NEAR(x.get_u().ComputeL2Error(exact_u), 0.0, tol);

    delete fec_ND;
    delete fec_CG;
}

TEST(StokesSystem, CubeVortexTrace_SolvesToExactVortex)
{
    // Solves a Stokes problem on a cube mesh with a vortex-like exact velocity trace.
    // Uses zero body force and imposes a rotational trace field via the RHS construction.
    // Expects the recovered velocity to match the prescribed trace field in L2 norm.

    const double mass = 0.0;
    const double viscosity = 0.1;
    const int refinements = 1;
    const int order = 1;
    const double tol = 1e-10;
    const std::string mesh_string = "../extern/mfem/data/ref-cube.mesh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    auto f = [](const mfem::Vector &, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = 0.0;
        y.Elem(1) = 0.0;
        y.Elem(2) = 0.0;
    };

    auto tr_u = [](const mfem::Vector &x, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = -x.Elem(1);
        y.Elem(1) = x.Elem(0);
        y.Elem(2) = 0.0;
    };

    StokesSystem sys(ND, CG, mass, viscosity, 1.0, 100.0);
    StokesRHS rhs(ND, CG, f, tr_u, 1.0, 100.0, viscosity);
    StokesSolution x(ND, CG);

    auto solver = std::make_unique<mfem::GMRESSolver>();
    solver->SetAbsTol(tol);
    solver->SetKDim(3000);
    solver->SetRelTol(0.0);
    solver->SetMaxIter(10000);
    solver->SetPrintLevel(1);
    solver->SetOperator(sys);
    solver->Mult(rhs, x);

    mfem::VectorFunctionCoefficient exact_u(3, tr_u);
    EXPECT_NEAR(x.get_u().ComputeL2Error(exact_u), 0.0, tol);

    delete fec_ND;
    delete fec_CG;
}

TEST(SchurPreconditioner, CubeSmoothRHS_GMRESWithSchurPreconditionerRuns)
{
    // Runs GMRES on the coupled Stokes system using a Schur-complement preconditioner.
    // Assembles RHS from a smooth, non-constant vector field on a refined cube mesh.
    // Confirms the solve completes with the provided tolerance and configuration.

    const double viscosity = 0.02;
    const double mass = 1.0;
    const int refinements = 2;
    const int order = 1;
    const double tol = 1e-5;
    const std::string mesh_string = "../extern/mfem/data/ref-cube.mesh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    GridFunction u(&ND);
    GridFunction p(&CG);

    mfem::VectorFunctionCoefficient vec1_coef(
        3, [](mfem::Vector, mfem::Vector &y) -> void
        {
            y.SetSize(3);
            y.Elem(0) = 1.0;
            y.Elem(1) = 1.0;
            y.Elem(2) = 1.0;
        });
    u.ProjectCoefficient(vec1_coef);
    p = 0.0;

    const int size_1 = u.Size() + p.Size();
    Vector x(size_1);
    x = 0.0;

    Array<int> u_dofs(u.Size()), p_dofs(p.Size());
    std::iota(u_dofs.begin(), u_dofs.end(), 0);
    std::iota(p_dofs.begin(), p_dofs.end(), u.Size());

    x.SetSubVector(u_dofs, u);
    x.SetSubVector(p_dofs, p);

    auto f = [](const mfem::Vector &x, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = x.Elem(0) * std::sin(x.Elem(1));
        y.Elem(1) = x.Elem(1) * x.Elem(2);
        y.Elem(2) = x.Elem(0);
    };

    StokesRHS rhs(ND, CG, f, f, -1.0, 0.0);
    SchurPreconditioner pre(ND, CG, mass, viscosity);

    StokesSystem sys(ND, CG, mass, viscosity, -1.0, 0.0);
    pre.SetOperator(sys);

    auto solver = std::make_unique<mfem::GMRESSolver>();
    solver->SetAbsTol(tol);
    solver->SetKDim(3000);
    solver->SetRelTol(0.0);
    solver->SetMaxIter(10000);
    solver->SetPrintLevel(1);
    solver->SetOperator(sys);
    solver->SetPreconditioner(pre);
    solver->Mult(rhs, x);

    delete fec_ND;
    delete fec_CG;
}

TEST(SchurSolver, CubeSmoothRHS_SchurSolverReproducesRHSUnderOperator)
{
    // Solves the Stokes system using a custom Schur solver and verifies operator consistency.
    // Builds RHS with a smooth forcing and zero trace, then solves on a refined cube mesh.
    // Applies the system operator to the computed solution and compares to RHS entrywise.

    const double viscosity = 1.0;
    const double mass = 1.0;
    const int refinements = 2;
    const int order = 1;
    const double theta = -1.0;
    const double Cw = 0.0;
    const std::string mesh_string = "../extern/mfem/data/ref-cube.mesh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    GridFunction u(&ND);
    GridFunction p(&CG);

    mfem::VectorFunctionCoefficient vec1_coef(
        3, [](mfem::Vector, mfem::Vector &y) -> void
        {
            y.SetSize(3);
            y.Elem(0) = 1.0;
            y.Elem(1) = 1.0;
            y.Elem(2) = 1.0;
        });
    u.ProjectCoefficient(vec1_coef);
    p = 0.0;

    const int size_1 = u.Size() + p.Size();
    Vector x(size_1);
    x = 0.0;

    Array<int> u_dofs(u.Size()), p_dofs(p.Size());
    std::iota(u_dofs.begin(), u_dofs.end(), 0);
    std::iota(p_dofs.begin(), p_dofs.end(), u.Size());

    x.SetSubVector(u_dofs, u);
    x.SetSubVector(p_dofs, p);

    auto f = [](const mfem::Vector &x, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = x.Elem(0) * std::sin(x.Elem(1));
        y.Elem(1) = x.Elem(1) * x.Elem(2);
        y.Elem(2) = x.Elem(0);
    };

    auto tr_u = [](const mfem::Vector &, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = 0.0;
        y.Elem(1) = 0.0;
        y.Elem(2) = 0.0;
    };

    StokesRHS rhs(ND, CG, f, tr_u, theta, Cw, viscosity);
    StokesSystem sys(ND, CG, mass, viscosity, theta, Cw);

    int iterations = 0;
    SchurSolver solver(ND, CG, mass, viscosity, iterations, 1e-12);
    solver.SetOperator(sys);
    solver.Mult(rhs, x);

    mfem::Vector sys_x(sys.NumRows());
    sys.Mult(x, sys_x);

    for (int i = 0; i < rhs.GetBlock(0).Size(); ++i) { EXPECT_NEAR(sys_x[i], rhs[i], 1e-6); }
    for (int i = rhs.GetBlock(0).Size(); i < rhs.Size(); ++i) { EXPECT_NEAR(sys_x[i], rhs[i], 1e-6); }

    delete fec_ND;
    delete fec_CG;
}

TEST(StokesSystemAndRHS, CubeVortexTrace_VelocityBlockMatchesRHSVelocityBlock)
{
    // Checks consistency between the velocity-block operator application and assembled RHS velocity block.
    // Uses a vortex-like trace field with zero forcing on a refined cube mesh and strong theta penalty.
    // Expects the operator-applied velocity to match the RHS velocity block within a relative tolerance.

    const double viscosity = 1.0;
    const double mass = 0.0;
    const int refinements = 1;
    const int order = 1;
    const double theta = -1e8;
    const double Cw = 0.0;
    const std::string mesh_string = "../extern/mfem/data/ref-cube.mesh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    auto f = [](const mfem::Vector &, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = 0.0;
        y.Elem(1) = 0.0;
        y.Elem(2) = 0.0;
    };

    auto tr_u = [](const mfem::Vector &x, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = -x.Elem(1);
        y.Elem(1) = x.Elem(0);
        y.Elem(2) = 0.0;
    };

    mfem::VectorFunctionCoefficient u_coef(3, tr_u);
    mfem::GridFunction u(&ND);
    u.ProjectCoefficient(u_coef);

    StokesRHS rhs(ND, CG, f, tr_u, theta, Cw, viscosity);
    StokesSystem sys(ND, CG, mass, viscosity, theta, Cw);

    mfem::Vector sys_u(sys.GetBlock(0, 0).Height());
    sys.GetBlock(0, 0).Mult(u, sys_u);

    for (int i = 0; i < rhs.GetBlock(0).Size(); ++i)
    {
        EXPECT_NEAR(sys_u[i], rhs[i], rhs.Norml2() * 1e-7);
    }

    delete fec_ND;
    delete fec_CG;
}
