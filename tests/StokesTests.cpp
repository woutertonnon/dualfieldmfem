#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>

#include "StokesOperators.h"
#include "mfem.hpp"

using namespace mfem;

TEST(HdivStokesSystem, PeriodicConstantField_TransposeGivesZero)
{
    // Applies Stokes operator transpose to a constant velocity field on a periodic mesh.
    // Uses ND for velocity and CG for pressure with zero mass contribution in this configuration.
    // Expects the resulting vector to be (numerically) zero.

    const double viscosity = 1.0;
    const int refinements = 6;
    const int order = 2;
    const double tol = 1e-10;
    const std::string mesh_string = "../extern/mfem/data/ref-cube.mesh";

    //std::cout << "test\n";
    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();

    //std::cout << "test\n";
    auto *fec_RT = new RT_FECollection(order-1,dim);
    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_DG = new L2_FECollection(order-1, dim);

    //std::cout << "test\n";
    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace DG(&mesh, fec_DG);

    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1; // attribute 1 is essential

    mfem::Array<int> ess_tdof;
    RT.GetEssentialTrueDofs(ess_bdr, ess_tdof);



    //std::cout << "test\n";
    hdiv::StokesSolution sol(RT,ND,DG);

    // 1. Get the total number of True DOFs
    int n_tdofs = sol.Size();

    // 2. Initialize an array of size N with all zeros
    mfem::Array<int> ess_marker(n_tdofs);
    ess_marker = 0; 

    // 3. Set values to 1 for every index present in ess_tdof
    for (int i = 0; i < ess_tdof.Size(); i++) {
        int index = ess_tdof[i];
        ess_marker[index] = 1;
    }


    //std::cout << "test\n";
    auto vec1 = [](const mfem::Vector&, double, mfem::Vector &y) -> void
        {
            y.SetSize(3);
            y.Elem(0) = 1.0;
            y.Elem(1) = 1.0;
            y.Elem(2) = 1.0;
        };
    mfem::VectorFunctionCoefficient vec1_coef(
        3, vec1);
    sol.get_u().ProjectCoefficient(vec1_coef);
        auto vec0 = [](const mfem::Vector&, double, mfem::Vector &y) -> void
        {
            y.SetSize(3);
            y.Elem(0) = 0.0;
            y.Elem(1) = 0.0;
            y.Elem(2) = 0.0;
        };
    mfem::VectorFunctionCoefficient vec0_coef(3,vec0);

    //std::cout << "test7\n";
    double mass = 100.;
    hdiv::StokesSystem sys(RT, ND, DG, ess_tdof, mass, viscosity);
    sys.Update(vec0_coef);
    //sys.Print(std::cout);
    //ess_tdof.Print(std::cout);
    //std::cout << "test9\n";
    //std::cout << "test10\n";
    hdiv::StokesRHS rhs(RT,ND,DG, ess_tdof, vec0, vec1);
    rhs.Update(sol.get_u(),0.,mass);

    //std::cout << "test11\n";
    //rhs.Print(std::cout);
;
    mfem::Vector y(sol.Size());
    sys.Mult(sol, y);
    //    y.Print(std::cout);
    //    std::cout << std::endl;
    //    rhs.Print(std::cout);

    for(int i = 0; i<y.Size(); ++i){
   //     if(ess_marker[i]==0 && std::abs(y[i]-rhs[i])>1e-5) std::cout << "WARNING!!!\n";
    //    std::cout << i << "," << ess_marker[i] << ", " << y[i] << ", " << rhs[i] << std::endl;
    }
   //     y.Print(std::cout);
    

    std::cout << "RT Size: " << RT.GetNDofs() << "\nND Size: " << ND.GetNDofs() << "\nDG Size: " << DG.GetNDofs() << std::endl;

    for(int i=0; i<y.Size(); ++i)
        EXPECT_NEAR(y[i], rhs[i], tol) << "Failed at index = " << i << std::endl;

    int iters;
    mfem::GMRESSolver solver;
    //std::cout << "test17\n";
    //mfem::SparseMatrix *mono_mat = sys.CreateMonolithic();
    //std::cout << "test18\n";
    solver.SetOperator(sys);
    solver.SetPrintLevel(1);
    solver.SetRelTol(1e-8);
    solver.SetAbsTol(1e-4);
    solver.SetKDim(1000);
    solver.SetMaxIter(1000000);

    //X.Print(std::cout);
    //B.Print(std::cout);
    sol.get_u() = 0.;
    solver.Mult(rhs,sol);
    //sol.Print(std::cout);


    y=0.;
    sys.Mult(sol, y);
      //  sol.Print(std::cout);
   //     y.Print(std::cout);
    y -= rhs;
    //y.Print(std::cout);
    //X.Print(std::cout);
    //sol.get_u().Print(std::cout);
    ASSERT_NEAR(sol.get_u().ComputeL2Error(vec1_coef),0.,1e-3);
    
    delete fec_RT;
    delete fec_ND;
    delete fec_DG;
}

TEST(HdivStokesSystem, PeriodicConstantField_TransposeGivesZero2)
{
    // Applies Stokes operator transpose to a constant velocity field on a reference cube.
    // Uses ND for velocity and CG for pressure with zero mass contribution in this configuration.
    // Expects the resulting vector to be (numerically) zero.

    const double viscosity = 1.0;
    const int refinements = 0;
    const int order = 1;
    const double tol = 1e-10;
    const std::string mesh_string = "../extern/mfem/data/ref-cube.mesh";

    std::cout << "test\n";
    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; ++l) { mesh.UniformRefinement(); }
    const int dim = mesh.Dimension();




    std::cout << "test\n";
    auto *fec_RT = new RT_FECollection(order-1,dim);
    auto *fec_ND = new ND_FECollection(order, dim);
    auto *fec_DG = new L2_FECollection(order-1, dim);

    std::cout << "test\n";
    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace DG(&mesh, fec_DG);
    std::cout << "RT Size: " << RT.GetNDofs() << "\nND Size: " << ND.GetNDofs() << "\nDG Size: " << DG.GetNDofs() << ", n elem = " << mesh.GetNE() << std::endl;
    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1; // attribute 1 is essential

    mfem::Array<int> ess_tdof;
    RT.GetEssentialTrueDofs(ess_bdr, ess_tdof);


    std::cout << "test\n";
    hdiv::StokesSolution sol(RT,ND,DG);

    std::cout << "test\n";
    auto vec1 = [](const mfem::Vector&, double, mfem::Vector &y) -> void
        {
            y.SetSize(3);
            y.Elem(0) = 1.0;
            y.Elem(1) = 1.0;
            y.Elem(2) = 1.0;
        };
    mfem::VectorFunctionCoefficient vec1_coef(
        3, vec1);
    sol.get_u().ProjectCoefficient(vec1_coef);


    std::cout << "test\n";
    hdiv::StokesSystem sys(RT, ND, DG, ess_tdof, 1., viscosity);
    hdiv::StokesRHS rhs(RT,ND,DG, ess_tdof,vec1, vec1);


    //ess_tdof.Append(sol.Size()-1);
    //std::cout << "ess_tdof with size: " << ess_tdof.Size() << "\n";
      //  ess_tdof.Print(std::cout);

    //sol.get_u().ProjectBdrCoefficientNormal(vec1_coef, ess_bdr);

    // assemble a,b then:
    mfem::Operator *A;
    mfem::Vector X, B;
    sys.FormLinearSystem(ess_tdof, sol, rhs, A, X, B);
        rhs.Print(std::cout);
    //sys.Print(std::cout);
    //rhs.Print(std::cout);
    //sol.Print(std::cout);
    // solve reduced system
    int iters;
    mfem::MINRESSolver solver;
    solver.SetOperator(*A);
    solver.SetPrintLevel(1);
    solver.SetMaxIter(10000);

    //X.Print(std::cout);
    //B.Print(std::cout);
    solver.Mult(B, X);

    //X.Print(std::cout);
    // recover full x (including essential values)
    sys.RecoverFEMSolution(X, rhs, sol);

    //X.Print(std::cout);
    //sol.get_u().Print(std::cout);
    ASSERT_NEAR(sol.get_u().ComputeL2Error(vec1_coef),0.,1e-5);

    delete fec_RT;
    delete fec_ND;
    delete fec_DG;
}


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

    hcurl::StokesSystem sys(ND, CG, 0.0, viscosity, 1.0, 100.0);

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

    hcurl::StokesRHS rhs(ND, CG, f, f);
    hcurl::StokesSystem sys(ND, CG, 1.0, viscosity, 1.0, 100.0);

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

    hcurl::StokesSystem sys(ND, CG, 0.0, viscosity, 0.0, 0.0);

    hcurl::StokesSolution y(ND, CG);
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

    hcurl::StokesRHS rhs(ND, CG, f, f, 1.0, 100.0, viscosity);
    hcurl::StokesSystem sys(ND, CG, 1.0, viscosity, 1.0, 100.0);

    hcurl::StokesSolution y(ND, CG);
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

    hcurl::StokesSystem sys(ND, CG, mass, viscosity, 1.0, 100.0);
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
