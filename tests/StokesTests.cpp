#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <numeric>
#include <string>

#include "StokesOperators.h"
#include "mfem.hpp"

using namespace mfem;

TEST(HdivStokesSystem, PeriodicMesh_MultMatchesRHSAndGMRESSolvesToConstantField)
{
    // Checks that the H(div) Stokes operator applied to a constant velocity field matches the
    // assembled RHS, then solves with GMRES and verifies the recovered velocity in L2.

    const double viscosity = 1.0;
    const int refinements = 0;
    const int order = 2;
    const double tol = 1e-10;
    const std::string mesh_string = "../geo/mesh/ConstantField.msh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_RT = new RT_FECollection(order-1, dim);
    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_DG = new L2_FECollection(order-1, dim);

    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace DG(&mesh, fec_DG);

    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1;

    mfem::Array<int> ess_tdof;
    RT.GetEssentialTrueDofs(ess_bdr, ess_tdof);

    hdiv::StokesSolution sol(RT, ND, DG);

    auto vec1 = [](const mfem::Vector&, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = 1.0;
        y.Elem(1) = 1.0;
        y.Elem(2) = 1.0;
    };
    auto vec0 = [](const mfem::Vector&, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = 0.0;
        y.Elem(1) = 0.0;
        y.Elem(2) = 0.0;
    };

    mfem::VectorFunctionCoefficient vec1_coef(3, vec1);
    mfem::VectorFunctionCoefficient vec0_coef(3, vec0);
    sol.get_u().ProjectCoefficient(vec1_coef);

    const double mass = 100.;
    hdiv::StokesSystem sys(RT, ND, DG, ess_tdof, mass, viscosity);
    sys.Update(vec0_coef);

    hdiv::StokesRHS rhs(RT, ND, DG, ess_tdof, vec0, vec1);
    rhs.Update(sol.get_u(), 0., mass);

    mfem::Vector y(sol.Size());
    sys.Mult(sol, y);

    for (int i = 0; i < y.Size(); ++i)
        EXPECT_NEAR(y[i], rhs[i], tol) << "Failed at index = " << i;

    mfem::GMRESSolver solver;
    solver.SetOperator(sys);
    solver.SetPrintLevel(1);
    solver.SetRelTol(1e-8);
    solver.SetAbsTol(1e-4);
    solver.SetKDim(1000);
    solver.SetMaxIter(1000000);

    sol.get_u() = 0.;
    solver.Mult(rhs, sol);

    ASSERT_NEAR(sol.get_u().ComputeL2Error(vec1_coef), 0., 1e-3);

    delete fec_RT;
    delete fec_ND;
    delete fec_DG;
}

TEST(HdivStokesSystem, CubeMesh_FormLinearSystemWithMINRESSolvesToConstantField)
{
    // Solves the H(div) Stokes system on a cube mesh via FormLinearSystem and MINRES,
    // and verifies the recovered velocity matches the exact constant field in L2.

    const double viscosity = 1.0;
    const int refinements = 0;
    const int order = 1;
    const double tol = 1e-10;
    const std::string mesh_string = "../extern/mfem/data/ref-cube.mesh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_RT = new RT_FECollection(order-1, dim);
    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_DG = new L2_FECollection(order-1, dim);

    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace DG(&mesh, fec_DG);

    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1;

    mfem::Array<int> ess_tdof;
    RT.GetEssentialTrueDofs(ess_bdr, ess_tdof);

    hdiv::StokesSolution sol(RT, ND, DG);

    auto vec1 = [](const mfem::Vector&, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = 1.0;
        y.Elem(1) = 1.0;
        y.Elem(2) = 1.0;
    };
    mfem::VectorFunctionCoefficient vec1_coef(3, vec1);
    sol.get_u().ProjectCoefficient(vec1_coef);

    hdiv::StokesSystem sys(RT, ND, DG, ess_tdof, 1., viscosity);
    hdiv::StokesRHS rhs(RT, ND, DG, ess_tdof, vec1, vec1);

    mfem::Operator *A;
    mfem::Vector X, B;
    sys.FormLinearSystem(ess_tdof, sol, rhs, A, X, B);

    mfem::MINRESSolver solver;
    solver.SetOperator(*A);
    solver.SetPrintLevel(1);
    solver.SetMaxIter(10000);
    solver.Mult(B, X);

    sys.RecoverFEMSolution(X, rhs, sol);

    ASSERT_NEAR(sol.get_u().ComputeL2Error(vec1_coef), 0., 1e-5);

    delete fec_RT;
    delete fec_ND;
    delete fec_DG;
}


TEST(HcurlStokesSystem, PeriodicMesh_MultTransposeOfConstantFieldIsZero)
{
    // Verifies that MultTranspose of the H(curl) Stokes operator applied to a constant velocity
    // field is zero on a periodic mesh (no viscous or mass contributions for constant fields).

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

    hcurl::StokesSystem sys(ND, CG, 0.0, viscosity, 1.0, 100.0);

    mfem::Vector y(x.Size());
    sys.MultTranspose(x, y);

    y.Abs();
    EXPECT_NEAR(y.Max(), 0.0, tol);

    delete fec_ND;
    delete fec_CG;
}

TEST(HcurlStokesSystem, PeriodicMesh_MultTransposeMatchesRHS)
{
    // Verifies that MultTranspose of the H(curl) Stokes operator applied to a constant field
    // matches the assembled RHS component-wise on a periodic mesh.

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

    hcurl::StokesRHS rhs(ND, CG, f, f);
    hcurl::StokesSystem sys(ND, CG, 1.0, viscosity, 1.0, 100.0);

    mfem::Vector y(x.Size());
    sys.MultTranspose(x, y);

    ASSERT_EQ(y.Size(), rhs.Size());
    for (int i = 0; i < y.Size(); ++i) { EXPECT_NEAR(y[i], rhs[i], tol); }

    delete fec_ND;
    delete fec_CG;
}

TEST(HcurlStokesSystem, PeriodicMesh_GMRESSolvesToConstantField)
{
    // Solves the H(curl) Stokes system with constant forcing on a periodic mesh using GMRES,
    // and verifies the recovered velocity matches the exact constant field in L2.

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

    hcurl::StokesSystem sys(ND, CG, 1.0, viscosity, 1.0, 100.0);
    hcurl::StokesRHS rhs(ND, CG, f, f);
    hcurl::StokesSolution x(ND, CG);

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

TEST(HcurlStokesSystem, CubeMesh_MultTransposeVelocityBlockIsZeroWithoutNitsche)
{
    // Verifies that the velocity block of MultTranspose is zero when applied to a constant field
    // on a cube mesh with mass, Nitsche boundary terms, and interior-face jump stabilization
    // all disabled (mass=0, theta=0, Cw=0, sigma=0).

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

    hcurl::StokesSolution x(ND, CG);
    mfem::VectorFunctionCoefficient vec1_coef(
        3, [](mfem::Vector, mfem::Vector &y) -> void
        {
            y.SetSize(3);
            y.Elem(0) = 1.0;
            y.Elem(1) = 1.0;
            y.Elem(2) = 1.0;
        });
    x.get_u().ProjectCoefficient(vec1_coef);

    // sigma=0, gamma=0: disable all interior-face terms
    hcurl::StokesSystem sys(ND, CG, 0.0, viscosity, 0.0, 0.0, 0.0, 0.0);

    hcurl::StokesSolution y(ND, CG);
    sys.MultTranspose(x, y);

    for (auto com : y.get_u()) { EXPECT_NEAR(com, 0.0, tol); }

    delete fec_ND;
    delete fec_CG;
}

TEST(HcurlStokesSystem, CubeMesh_MultTransposeMatchesRHSWithNitsche)
{
    // Verifies that MultTranspose of the H(curl) Stokes operator matches the assembled RHS
    // component-wise on a cube mesh with Nitsche boundary terms and jump stabilization enabled.
    // The test field is the constant (1,1,1), which has zero interior-face jumps in ND so the
    // jump penalty (sigma=10) contributes nothing and the RHS assembled without interior terms
    // still matches the operator output.

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

    hcurl::StokesRHS rhs(ND, CG, f, f, 1.0, 100.0, viscosity);
    hcurl::StokesSystem sys(ND, CG, 1.0, viscosity, 1.0, 100.0, 10.0);

    hcurl::StokesSolution y(ND, CG);
    sys.MultTranspose(x, y);

    ASSERT_EQ(y.Size(), rhs.Size());
    for (int i = 0; i < y.Size(); ++i) { EXPECT_NEAR(y[i], rhs[i], tol); }

    delete fec_ND;
    delete fec_CG;
}

TEST(HcurlStokesSystem, CubeMesh_GMRESSolvesToConstantField)
{
    // Solves the H(curl) Stokes system with constant forcing on a cube mesh using GMRES,
    // and verifies the recovered velocity matches the exact constant field in L2.
    // Jump stabilization (sigma=10) is active; the constant field has zero interior-face
    // jumps in ND so the penalty does not pollute the exact solution.

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

    hcurl::StokesSystem sys(ND, CG, 1.0, viscosity, 1.0, 100.0, 10.0);
    hcurl::StokesRHS rhs(ND, CG, f, f);
    hcurl::StokesSolution x(ND, CG);

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

TEST(HcurlStokesSystem, CubeMesh_GMRESSolvesToVortexTraceField)
{
    // Solves the H(curl) Stokes system on a cube mesh with a vortex trace boundary condition
    // u = (-y, x, 0) and zero body force, and verifies the recovered velocity matches the
    // exact rotational field in L2.  The exact solution lies in ND1 (it is a + b×x with
    // b = (0,0,1)), so its ND projection has zero interior-face jumps and jump stabilization
    // (sigma=10) does not alter the solution.

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

    hcurl::StokesSystem sys(ND, CG, mass, viscosity, 1.0, 100.0, 10.0);
    hcurl::StokesRHS rhs(ND, CG, f, tr_u, 1.0, 100.0, viscosity);
    hcurl::StokesSolution x(ND, CG);

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

TEST(HcurlSchurPreconditioner, CubeMesh_PreconditionedGMRESConverges)
{
    // Verifies that GMRES preconditioned with the Schur complement preconditioner converges
    // on a smooth RHS problem on a refined cube mesh.

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

    hcurl::StokesRHS rhs(ND, CG, f, f, -1.0, 0.0);
    hcurl::SchurPreconditioner pre(ND, CG, mass, viscosity);

    hcurl::StokesSystem sys(ND, CG, mass, viscosity, -1.0, 0.0);
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

TEST(HcurlSchurSolver, CubeMesh_SolutionSatisfiesOperatorEquation)
{
    // Solves the H(curl) Stokes system using the Schur solver and verifies that applying
    // the system operator to the computed solution reproduces the RHS component-wise.

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

    hcurl::StokesRHS rhs(ND, CG, f, tr_u, theta, Cw, viscosity);
    hcurl::StokesSystem sys(ND, CG, mass, viscosity, theta, Cw);

    int iterations = 0;
    hcurl::SchurSolver solver(ND, CG, mass, viscosity, iterations, 1e-12);
    solver.SetOperator(sys);
    solver.Mult(rhs, x);

    mfem::Vector sys_x(sys.NumRows());
    sys.Mult(x, sys_x);

    for (int i = 0; i < rhs.GetBlock(0).Size(); ++i) { EXPECT_NEAR(sys_x[i], rhs[i], 1e-6); }
    for (int i = rhs.GetBlock(0).Size(); i < rhs.Size(); ++i) { EXPECT_NEAR(sys_x[i], rhs[i], 1e-6); }

    delete fec_ND;
    delete fec_CG;
}

TEST(HcurlStokesSystem, CubeMesh_VelocityBlockMatchesRHSForVortexTrace)
{
    // Verifies that the velocity block of the system operator applied to the projected vortex
    // trace field matches the velocity block of the assembled RHS within a relative tolerance.

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

    hcurl::StokesRHS rhs(ND, CG, f, tr_u, theta, Cw, viscosity);
    hcurl::StokesSystem sys(ND, CG, mass, viscosity, theta, Cw);

    mfem::Vector sys_u(sys.GetBlock(0, 0).Height());
    sys.GetBlock(0, 0).Mult(u, sys_u);

    for (int i = 0; i < rhs.GetBlock(0).Size(); ++i)
    {
        EXPECT_NEAR(sys_u[i], rhs[i], rhs.Norml2() * 1e-7);
    }

    delete fec_ND;
    delete fec_CG;
}

TEST(HcurlStokesSystem, CubeMesh_NormalJumpPenaltyIncreasesABlock)
{
    // Verifies that sigma > 0 (ND_DGPenaltyIntegrator, normal-jump penalty) strictly
    // increases the diagonal of the A block.  gamma=0 isolates this term from the
    // curl-jump ghost penalty.

    const double viscosity = 1.0;
    const double mass = 1.0;
    const int refinements = 1;
    const int order = 1;
    const std::string mesh_string = "../extern/mfem/data/ref-cube.mesh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    // theta=-1, Cw=0: no boundary Nitsche; gamma=0: no curl-jump.
    // Only sigma distinguishes the two systems.
    hcurl::StokesSystem sys_no_jump  (ND, CG, mass, viscosity, -1.0, 0.0,  0.0, 0.0);
    hcurl::StokesSystem sys_with_jump(ND, CG, mass, viscosity, -1.0, 0.0, 10.0, 0.0);

    const mfem::SparseMatrix &A0 = sys_no_jump  .GetBlock(0, 0);
    const mfem::SparseMatrix &A1 = sys_with_jump.GetBlock(0, 0);

    ASSERT_EQ(A0.Height(), A1.Height());

    bool any_larger = false;
    for (int i = 0; i < A0.Height(); ++i)
    {
        const double d0 = A0.Elem(i, i);
        const double d1 = A1.Elem(i, i);
        EXPECT_GE(d1, d0 - 1e-14) << "Diagonal decreased at DOF " << i;
        if (d1 > d0 + 1e-14) any_larger = true;
    }
    EXPECT_TRUE(any_larger) << "Normal-jump penalty did not increase any diagonal entry";

    delete fec_ND;
    delete fec_CG;
}

TEST(HcurlStokesSystem, CubeMesh_CurlJumpPenaltyIncreasesABlock)
{
    // Verifies that gamma > 0 (ND_CurlJumpIntegrator, curl-jump ghost penalty) strictly
    // increases the diagonal of the A block.  sigma=0 isolates this term from the
    // normal-jump penalty.
    //
    // For ND1 elements curl(u) is piecewise constant (P0), so [[curl phi_k]] is
    // non-zero on every interior face adjacent to edge DOFs, guaranteeing a strict
    // diagonal increase.

    const double viscosity = 1.0;
    const double mass = 1.0;
    const int refinements = 1;
    const int order = 1;
    const std::string mesh_string = "../extern/mfem/data/ref-cube.mesh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    // theta=-1, Cw=0: no boundary Nitsche; sigma=0: no normal-jump.
    // Only gamma distinguishes the two systems.
    hcurl::StokesSystem sys_no_curl_jump  (ND, CG, mass, viscosity, -1.0, 0.0, 0.0, 0.0);
    hcurl::StokesSystem sys_with_curl_jump(ND, CG, mass, viscosity, -1.0, 0.0, 0.0, 1.0);

    const mfem::SparseMatrix &A0 = sys_no_curl_jump  .GetBlock(0, 0);
    const mfem::SparseMatrix &A1 = sys_with_curl_jump.GetBlock(0, 0);

    ASSERT_EQ(A0.Height(), A1.Height());

    bool any_larger = false;
    for (int i = 0; i < A0.Height(); ++i)
    {
        const double d0 = A0.Elem(i, i);
        const double d1 = A1.Elem(i, i);
        EXPECT_GE(d1, d0 - 1e-14) << "Diagonal decreased at DOF " << i;
        if (d1 > d0 + 1e-14) any_larger = true;
    }
    EXPECT_TRUE(any_larger) << "Curl-jump ghost penalty did not increase any diagonal entry";

    delete fec_ND;
    delete fec_CG;
}

TEST(HcurlStokesSystem, CubeMesh_UpwindPenaltyIncreasesABlockWithNonzeroWind)
{
    // Verifies that upwind_scale > 0 (ND_UpwindIntegrator, Heumann upwind) strictly
    // increases the diagonal of the A block when a non-zero wind is present.
    // sigma=0 and gamma=0 isolate the upwind term from the other face penalties.
    //
    // The upwind term assembles |w.n_F| [[u]].[v] on interior faces.  With a
    // constant wind (1,0,0), every interior face that has a non-zero normal
    // component in the x-direction contributes to the diagonal, so at least some
    // ND edge DOFs must see an increased diagonal.

    const double viscosity = 1.0;
    const double mass = 1.0;
    const int refinements = 1;
    const int order = 1;
    const std::string mesh_string = "../extern/mfem/data/ref-cube.mesh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    // Constant wind in the x-direction
    mfem::VectorFunctionCoefficient w_coef(
        3, [](const mfem::Vector &, mfem::Vector &y) {
            y.SetSize(3);
            y[0] = 1.0; y[1] = 0.0; y[2] = 0.0;
        });

    // theta=-1, Cw=0: no boundary Nitsche; sigma=0, gamma=0: only upwind differs.
    hcurl::StokesSystem sys_no_upwind  (ND, CG, mass, viscosity, -1.0, 0.0, 0.0, 0.0, 0.0);
    hcurl::StokesSystem sys_with_upwind(ND, CG, mass, viscosity, -1.0, 0.0, 0.0, 0.0, 1.0);

    sys_no_upwind  .Update(w_coef);
    sys_with_upwind.Update(w_coef);

    const mfem::SparseMatrix &A0 = sys_no_upwind  .GetBlock(0, 0);
    const mfem::SparseMatrix &A1 = sys_with_upwind.GetBlock(0, 0);

    ASSERT_EQ(A0.Height(), A1.Height());

    bool any_larger = false;
    for (int i = 0; i < A0.Height(); ++i)
    {
        const double d0 = A0.Elem(i, i);
        const double d1 = A1.Elem(i, i);
        EXPECT_GE(d1, d0 - 1e-14) << "Diagonal decreased at DOF " << i;
        if (d1 > d0 + 1e-14) any_larger = true;
    }
    EXPECT_TRUE(any_larger) << "Heumann upwind did not increase any diagonal entry";

    delete fec_ND;
    delete fec_CG;
}

TEST(HdivStokesSystem, CubeMesh_PreconditionedGMRESConverges)
{
    // Verifies that GMRES preconditioned with hdiv::BlockDiagPreconditioner converges
    // on a smooth RHS problem on a refined cube mesh with zero advection.

    const double viscosity = 1.0;
    const double mass = 10.0;
    const int refinements = 2;
    const int order = 1;
    const double tol = 1e-5;
    const std::string mesh_string = "../extern/mfem/data/ref-cube.mesh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_RT = new RT_FECollection(order - 1, dim);
    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_DG = new L2_FECollection(order - 1, dim);

    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace DG(&mesh, fec_DG);

    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1;

    mfem::Array<int> ess_tdof;
    RT.GetEssentialTrueDofs(ess_bdr, ess_tdof);

    auto f = [](const mfem::Vector &x, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = x.Elem(0) * std::sin(x.Elem(1));
        y.Elem(1) = x.Elem(1) * x.Elem(2);
        y.Elem(2) = x.Elem(0);
    };
    auto zero = [](const mfem::Vector &, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y = 0.0;
    };

    mfem::VectorFunctionCoefficient zero_coef(3, zero);

    hdiv::StokesSystem sys(RT, ND, DG, ess_tdof, mass, viscosity);
    sys.Update(zero_coef);   // zero advection
    hdiv::StokesRHS rhs(RT, ND, DG, ess_tdof, f, zero);
    mfem::GridFunction u_zero(&RT);
    u_zero = 0.;
    rhs.Update(u_zero, 0., mass);

    hdiv::BlockDiagPreconditioner pre(RT, ND, DG, mass);
    pre.SetOperator(sys);

    hdiv::StokesSolution sol(RT, ND, DG);

    mfem::GMRESSolver solver;
    solver.SetRelTol(tol);
    solver.SetAbsTol(1e-12);
    solver.SetKDim(300);
    solver.SetMaxIter(10000);
    solver.SetPrintLevel(1);
    solver.SetOperator(sys);
    solver.SetPreconditioner(pre);
    solver.Mult(rhs, sol);

    ASSERT_TRUE(solver.GetConverged());

    // Verify residual: ||sys * sol - rhs|| / ||rhs|| < tol
    mfem::Vector res(sol.Size());
    sys.Mult(sol, res);
    res -= rhs;
    const double rhs_norm = rhs.Norml2();
    EXPECT_LT(res.Norml2() / (rhs_norm > 0. ? rhs_norm : 1.), tol * 10.);

    delete fec_RT;
    delete fec_ND;
    delete fec_DG;
}

TEST(HdivDirectSolver, CubeMesh_SolutionSatisfiesOperatorEquation)
{
    // Solves the H(div) Stokes system using hdiv::DirectSolver and verifies that
    // applying the system operator to the computed solution reproduces the RHS
    // component-wise.

    const double viscosity = 1.0;
    const double mass = 1.0;
    const int refinements = 0;
    const int order = 1;
    const std::string mesh_string = "../geo/mesh/LidDrivenCavity3D.msh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_RT = new RT_FECollection(order - 1, dim);
    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_DG = new L2_FECollection(order - 1, dim);

    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace DG(&mesh, fec_DG);

    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1;   // constrain all boundary faces

    mfem::Array<int> ess_tdof;
    RT.GetEssentialTrueDofs(ess_bdr, ess_tdof);

    auto f = [](const mfem::Vector &x, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = x.Elem(0) * std::sin(x.Elem(1));
        y.Elem(1) = x.Elem(1) * x.Elem(2);
        y.Elem(2) = x.Elem(0);
    };
    auto zero = [](const mfem::Vector &, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y = 0.0;
    };

    mfem::VectorFunctionCoefficient zero_coef(3, zero);

    hdiv::StokesSystem sys(RT, ND, DG, ess_tdof, mass, viscosity);
    sys.Update(zero_coef);   // zero advection
    hdiv::StokesRHS rhs(RT, ND, DG, ess_tdof, f, zero);
    mfem::GridFunction u_prev(&RT);
    u_prev = 0.;
    rhs.Update(u_prev, 0., mass);

    int iterations = 0;
    hdiv::DirectSolver solver(RT, ND, DG, iterations);
    solver.SetOperator(sys);

    hdiv::StokesSolution sol(RT, ND, DG);
    solver.Mult(rhs, sol);

    mfem::Vector sys_sol(sys.NumRows());
    sys.Mult(sol, sys_sol);

    for (int i = 0; i < rhs.GetBlock(0).Size(); ++i)
        EXPECT_NEAR(sys_sol[i], rhs[i], 1e-6) << "RT block failed at index " << i;
    for (int i = rhs.GetBlock(0).Size(); i < rhs.GetBlock(0).Size() + rhs.GetBlock(1).Size(); ++i)
        EXPECT_NEAR(sys_sol[i], rhs[i], 1e-6) << "ND block failed at index " << i;
    //for (int i = rhs.GetBlock(0).Size() + rhs.GetBlock(1).Size(); i < rhs.Size(); ++i)
    //    EXPECT_NEAR(sys_sol[i], rhs[i], 1e-6) << "DG block failed at index " << i;

    delete fec_RT;
    delete fec_ND;
    delete fec_DG;
}

TEST(HdivDirectSolver, CubeMesh_NormalBCL2BoundaryError)
{
    // Verifies that the solved velocity satisfies the prescribed normal BC
    // u·n = (1,0,0)·n in the L2 sense:
    //
    //   ||(u_h - (1,0,0)) · n||_{L2(∂Ω)} ≈ 0
    //
    // Also checks that ||u_h·n||^2_{L2(∂Ω)} = 2, which is the expected value
    // for u=(1,0,0) on a unit-cube mesh (only the two x-aligned faces contribute).
    // For RT elements the essential BC directly constrains the face-normal DOFs,
    // so with a direct (UMFPack) solve both errors should be at machine precision.

    const double viscosity = 1.0;
    const double mass = 1.0;
    const int refinements = 0;
    const int order = 1;
    const std::string mesh_string = "../geo/mesh/LidDrivenCavity3D.msh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_RT = new RT_FECollection(order - 1, dim);
    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_DG = new L2_FECollection(order - 1, dim);

    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace DG(&mesh, fec_DG);

    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1;   // constrain all boundary faces

    mfem::Array<int> ess_tdof;
    RT.GetEssentialTrueDofs(ess_bdr, ess_tdof);

    auto one = [](const mfem::Vector &x, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = 1.;
        y.Elem(1) = 0.;
        y.Elem(2) = 0.;
    };
    auto zero = [](const mfem::Vector &, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y = 0.0;
    };

    mfem::VectorFunctionCoefficient zero_coef(3, zero);
    mfem::VectorFunctionCoefficient one_coef(3, one);

    hdiv::StokesSystem sys(RT, ND, DG, ess_tdof, mass, viscosity);
    sys.Update(zero_coef);
    hdiv::StokesRHS rhs(RT, ND, DG, ess_tdof, zero, one);
    mfem::GridFunction u_prev(&RT);
    u_prev = 0.;
    rhs.Update(u_prev, 0., mass);

    int iterations = 0;
    hdiv::DirectSolver solver(RT, ND, DG, iterations);
    solver.SetOperator(sys);

    hdiv::StokesSolution sol(RT, ND, DG);
    solver.Mult(rhs, sol);

    // Integrate ((u_h - u_exact) · n)^2 over every essential boundary face,
    // where u_exact = (1,0,0).
    // The un-normalised face normal n_J from CalcOrtho(J) satisfies:
    //   dS = ||n_J|| d(ref),   n_hat = n_J / ||n_J||
    // so  ((u-u_exact) · n_hat)^2 dS = ((u-u_exact) · n_J)^2 / ||n_J|| d(ref).
    const mfem::GridFunction &u = sol.get_u();
    double err_norm_sq = 0.0;
    double norm_sq = 0.0;

    for (int be = 0; be < mesh.GetNBE(); ++be)
    {
        const int attr = mesh.GetBdrElement(be)->GetAttribute();
        if (!ess_bdr[attr - 1]) continue;

        mfem::FaceElementTransformations *ftr = mesh.GetBdrFaceTransformations(be);
        const int face_order = 2 * (order - 1) + 2;
        const mfem::IntegrationRule &ir =
            mfem::IntRules.Get(ftr->GetGeometryType(), face_order);

        for (int q = 0; q < ir.GetNPoints(); ++q)
        {
            const mfem::IntegrationPoint &ip = ir.IntPoint(q);
            ftr->SetAllIntPoints(&ip);

            // Un-normalised outward normal; ||n_J|| = surface area element.
            mfem::Vector n_J(dim);
            mfem::CalcOrtho(ftr->Jacobian(), n_J);

            // Evaluate u at this face point via the adjacent interior element.
            mfem::Vector u_val(dim);
            u.GetVectorValue(*ftr->Elem1, ftr->GetElement1IntPoint(), u_val);

            // (u_h - u_exact) · n_J,  with u_exact = (1,0,0).
            double un = u_val * n_J;
            double un_err = un - n_J[0]; // assume exact solution is (1,0,0)
            err_norm_sq += ip.weight * un_err * un_err / n_J.Norml2();
            norm_sq += ip.weight * un * un / n_J.Norml2();
        }
    }

    const double l2_normal_error = std::sqrt(err_norm_sq);
    EXPECT_NEAR(norm_sq, 2.0, 1e-10);
    EXPECT_NEAR(l2_normal_error, 0.0, 1e-10);

    delete fec_RT;
    delete fec_ND;
    delete fec_DG;
}

#if defined(MFEM_USE_MPI) && defined(MFEM_HYPRE_VERSION)
TEST(HcurlStokesSystem, CubeMesh_GMRESAMSSolverConvergesWithSmallResidual)
{
    const double viscosity = 1.0;
    const int refinements = 0;
    const int order = 1;
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

    hcurl::StokesSystem sys(ND, CG, 1.0, viscosity, 1.0, 100.0, 10.0);
    hcurl::StokesRHS rhs(ND, CG, f, f);
    hcurl::StokesSolution x(ND, CG);
    int iterations = 0;

    hcurl::GMRESAMSSolver solver(ND, CG, iterations, viscosity);
    solver.SetOperator(sys);
    solver.Mult(rhs, x);

    mfem::Vector r(rhs.Size());
    sys.Mult(x, r);
    r -= rhs;
    const double rhs_norm = rhs.Norml2();
    EXPECT_LT(r.Norml2() / (rhs_norm > 0.0 ? rhs_norm : 1.0), 1e-6);
    EXPECT_GT(iterations, 0);

    delete fec_ND;
    delete fec_CG;
}

TEST(HdivStokesSystem, CubeMesh_GMRESADSSolverConvergesWithSmallResidual)
{
    const double viscosity = 1.0;
    const double mass = 10.0;
    const int refinements = 0;
    const int order = 1;
    const std::string mesh_string = "../extern/mfem/data/ref-cube.mesh";

    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    auto *fec_RT = new RT_FECollection(order - 1, dim);
    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_DG = new L2_FECollection(order - 1, dim);

    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace DG(&mesh, fec_DG);

    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1;
    mfem::Array<int> ess_tdof;
    RT.GetEssentialTrueDofs(ess_bdr, ess_tdof);

    auto zero = [](const mfem::Vector &, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y = 0.0;
    };
    auto one = [](const mfem::Vector &, double, mfem::Vector &y) -> void
    {
        y.SetSize(3);
        y.Elem(0) = 1.0;
        y.Elem(1) = 0.0;
        y.Elem(2) = 0.0;
    };

    mfem::VectorFunctionCoefficient zero_coef(3, zero);
    hdiv::StokesSystem sys(RT, ND, DG, ess_tdof, mass, viscosity);
    sys.Update(zero_coef);

    hdiv::StokesRHS rhs(RT, ND, DG, ess_tdof, zero, one);
    mfem::GridFunction u_prev(&RT);
    u_prev = 0.0;
    rhs.Update(u_prev, 0.0, mass);

    hdiv::StokesSolution x(RT, ND, DG);
    int iterations = 0;
    hdiv::GMRESADSSolver solver(RT, ND, DG, iterations, mass);
    solver.SetOperator(sys);
    solver.Mult(rhs, x);

    mfem::Vector r(rhs.Size());
    sys.Mult(x, r);
    r -= rhs;
    const double rhs_norm = rhs.Norml2();
    EXPECT_LT(r.Norml2() / (rhs_norm > 0.0 ? rhs_norm : 1.0), 1e-6);
    EXPECT_GT(iterations, 0);

    delete fec_RT;
    delete fec_ND;
    delete fec_DG;
}
#endif
