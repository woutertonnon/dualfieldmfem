#include <gtest/gtest.h>

#include "mfem.hpp"
#include "StokesOperators.h"

using namespace mfem;

namespace
{
    Mesh MakeTestMesh()
    {
        return Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, true);
    }
} // namespace

TEST(StokesSystem, ConstantVectorFieldPeriodic)
{
    // ------------------------------------------------------------------
    // 0. Configuration
    // ------------------------------------------------------------------
    double viscosity = 1.;
    int refinements = 2;
    int order = 1;
    int printlevel = 0;
    double tol = 1e-10;
    std::string mesh_string = std::string("../extern/mfem/data/periodic-cube.mesh");

    // ------------------------------------------------------------------
    // 1. Mesh and FE spaces (PARALLEL)
    // ------------------------------------------------------------------
    Mesh mesh(mesh_string.c_str(), 1, 1);
    std::cout << mesh.GetNE() << std::endl;
    for (int l = 0; l < refinements; l++)
    {
        mesh.UniformRefinement();
    }
    std::cout << mesh.GetNE() << std::endl;
    int dim = mesh.Dimension();

    // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
    FiniteElementCollection *fec_ND = new ND_FECollection(order, dim);
    FiniteElementCollection *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    // ------------------------------------------------------------------
    // 2. Unknowns and gridfunctions (PARALLEL)
    // ------------------------------------------------------------------
    GridFunction u(&ND);
    GridFunction p(&CG);

    // Initial data from user-provided functions
    {
        mfem::VectorFunctionCoefficient vec1_coef(3,[](mfem::Vector x, mfem::Vector &y) -> void
                                        {y.SetSize(3); y.Elem(0)=1.;y.Elem(1)=1.;y.Elem(2)=1.; });
        u.ProjectCoefficient(vec1_coef);
        p = 0.;
    }

    // ------------------------------------------------------------------
    // 3. System sizes and block layout
    //    NOTE: Sizes are local DOFs per rank.
    // ------------------------------------------------------------------
    int size_1 = u.Size() + p.Size();

    Vector x(size_1);
    x = 0.0;

    Array<int> u_dofs(u.Size()), p_dofs(p.Size());
    std::iota(u_dofs.begin(), u_dofs.end(), 0);
    std::iota(p_dofs.begin(), p_dofs.end(), u.Size());


    x.SetSubVector(u_dofs, u);
    x.SetSubVector(p_dofs, p);


    // A1 blocks:
    StokesSystem sys(ND, CG, 0., viscosity, 1., 100.);

    mfem::Vector y(x.Size());
    sys.MultTranspose(x,y);

    y.Abs();
    EXPECT_NEAR(y.Max(),0.,1e-10);

    delete fec_ND;
    delete fec_CG;
}

TEST(StokesSystem, ConstantVectorFieldPeriodicForce)
{
    // ------------------------------------------------------------------
    // 0. Configuration
    // ------------------------------------------------------------------
    double viscosity = 1.;
    int refinements = 2;
    int order = 1;
    int printlevel = 0;
    double tol = 1e-10;
    std::string mesh_string = std::string("../extern/mfem/data/periodic-cube.mesh");

    // ------------------------------------------------------------------
    // 1. Mesh and FE spaces (PARALLEL)
    // ------------------------------------------------------------------
    Mesh mesh(mesh_string.c_str(), 1, 1);
    std::cout << mesh.GetNE() << std::endl;
    for (int l = 0; l < refinements; l++)
    {
        mesh.UniformRefinement();
    }
    std::cout << mesh.GetNE() << std::endl;
    int dim = mesh.Dimension();

    // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
    FiniteElementCollection *fec_ND = new ND_FECollection(order, dim);
    FiniteElementCollection *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    // ------------------------------------------------------------------
    // 2. Unknowns and gridfunctions (PARALLEL)
    // ------------------------------------------------------------------
    GridFunction u(&ND);
    GridFunction p(&CG);

    // Initial data from user-provided functions
    {
        mfem::VectorFunctionCoefficient vec1_coef(3,[](mfem::Vector x, mfem::Vector &y) -> void
                                        {y.SetSize(3); y.Elem(0)=1.;y.Elem(1)=1.;y.Elem(2)=1.; });
        u.ProjectCoefficient(vec1_coef);
        p = 0.;
    }

    // ------------------------------------------------------------------
    // 3. System sizes and block layout
    //    NOTE: Sizes are local DOFs per rank.
    // ------------------------------------------------------------------
    int size_1 = u.Size() + p.Size();

    Vector x(size_1);
    x = 0.0;

    Array<int> u_dofs(u.Size()), p_dofs(p.Size());
    std::iota(u_dofs.begin(), u_dofs.end(), 0);
    std::iota(p_dofs.begin(), p_dofs.end(), u.Size());


    x.SetSubVector(u_dofs, u);
    x.SetSubVector(p_dofs, p);

    auto f = [](const mfem::Vector& x, double, mfem::Vector& y) -> void{
            y.SetSize(3);
            y.Elem(0) = 1.;
            y.Elem(1) = 1.;
            y.Elem(2) = 1.;
        };


    StokesRHS rhs(ND,CG,f, f);

    // A1 blocks:
    StokesSystem sys(ND, CG, 1., viscosity, 1., 100.);

    mfem::Vector y(x.Size());
    sys.MultTranspose(x,y);

    ASSERT_EQ(y.Size(), rhs.Size());
    for (int i = 0; i < y.Size(); ++i)
    {
        EXPECT_NEAR(y[i], rhs[i], tol);
    }

    delete fec_ND;
    delete fec_CG;
}


TEST(StokesSystem, ConstantVectorFieldPeriodicSolver)
{
    // ------------------------------------------------------------------
    // 0. Configuration
    // ------------------------------------------------------------------
    double viscosity = 1.;
    int refinements = 1;
    int order = 1;
    int printlevel = 0;
    double tol = 1e-10;
    std::string mesh_string = std::string("../extern/mfem/data/periodic-cube.mesh");

    // ------------------------------------------------------------------
    // 1. Mesh and FE spaces (PARALLEL)
    // ------------------------------------------------------------------
    Mesh mesh(mesh_string.c_str(), 1, 1);
    std::cout << mesh.GetNE() << std::endl;
    for (int l = 0; l < refinements; l++)
    {
        mesh.UniformRefinement();
    }
    std::cout << mesh.GetNE() << std::endl;
    int dim = mesh.Dimension();

    // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
    FiniteElementCollection *fec_ND = new ND_FECollection(order, dim);
    FiniteElementCollection *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);


    auto f =
        [](const mfem::Vector& x, double, mfem::Vector& y) -> void{
            y.SetSize(3);
            y.Elem(0) = 1.;
            y.Elem(1) = 1.;
            y.Elem(2) = 1.;
        };



    // A1 blocks:
    StokesSystem sys(ND, CG, 1., viscosity, 1., 100.);
    StokesRHS rhs(ND, CG, f, f);
    StokesSolution x(ND, CG);

    auto solver = std::make_unique<mfem::GMRESSolver>();
    solver->SetAbsTol(tol);
    solver->SetKDim(3000);
    solver->SetRelTol(0.);
    solver->SetMaxIter(10000);
    solver->SetPrintLevel(1);
    solver->SetOperator(sys);
    solver->Mult(rhs,x);

    

    mfem::VectorFunctionCoefficient exact_u(3,f);
    EXPECT_NEAR(x.get_u().ComputeL2Error(exact_u),0.,1e-10);

    delete fec_ND;
    delete fec_CG;
}


TEST(StokesSystem, ConstantVectorFieldCube)
{
    // ------------------------------------------------------------------
    // 0. Configuration
    // ------------------------------------------------------------------
    double viscosity = 100.;
    int refinements = 0;
    int order = 1;
    int printlevel = 0;
    double tol = 1e-10;
    std::string mesh_string = std::string("../extern/mfem/data/ref-cube.mesh");

    // ------------------------------------------------------------------
    // 1. Mesh and FE spaces (PARALLEL)
    // ------------------------------------------------------------------
    Mesh mesh(mesh_string.c_str(), 1, 1);
    std::cout << mesh.GetNE() << std::endl;
    for (int l = 0; l < refinements; l++)
    {
        mesh.UniformRefinement();
    }
    std::cout << mesh.GetNE() << std::endl;
    int dim = mesh.Dimension();

    // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
    FiniteElementCollection *fec_ND = new ND_FECollection(order, dim);
    FiniteElementCollection *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    // ------------------------------------------------------------------
    // 2. Unknowns and gridfunctions (PARALLEL)
    // ------------------------------------------------------------------
    StokesSolution x(ND,CG);
    mfem::VectorFunctionCoefficient vec1_coef(3,[](mfem::Vector x, mfem::Vector &y) -> void
                                {y.SetSize(3); y.Elem(0)=1.;y.Elem(1)=1.;y.Elem(2)=1.; });
    x.get_u().ProjectCoefficient(vec1_coef);

    // A1 blocks:
    StokesSystem sys(ND, CG, 0., viscosity, 0., 0.);

    StokesSolution y(ND,CG);
    sys.MultTranspose(x,y);
    
    y.get_u().Print(std::cout);
    for(auto com : y.get_u())
        EXPECT_NEAR(com,0.,1e-10);

    delete fec_ND;
    delete fec_CG;
}

TEST(StokesSystem, ConstantVectorFieldCubeForce)
{
    // ------------------------------------------------------------------
    // 0. Configuration
    // ------------------------------------------------------------------
    double viscosity = 10.;
    int refinements = 2;
    int order = 1;
    int printlevel = 0;
    double tol = 1e-10;
    std::string mesh_string = std::string("../extern/mfem/data/ref-cube.mesh");

    // ------------------------------------------------------------------
    // 1. Mesh and FE spaces (PARALLEL)
    // ------------------------------------------------------------------
    Mesh mesh(mesh_string.c_str(), 1, 1);
    std::cout << mesh.GetNE() << std::endl;
    for (int l = 0; l < refinements; l++)
    {
        mesh.UniformRefinement();
    }
    std::cout << mesh.GetNE() << std::endl;
    int dim = mesh.Dimension();

    // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
    FiniteElementCollection *fec_ND = new ND_FECollection(order, dim);
    FiniteElementCollection *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    // ------------------------------------------------------------------
    // 2. Unknowns and gridfunctions (PARALLEL)
    // ------------------------------------------------------------------
    GridFunction u(&ND);
    GridFunction p(&CG);

    // Initial data from user-provided functions
    {
        mfem::VectorFunctionCoefficient vec1_coef(3,[](mfem::Vector x, mfem::Vector &y) -> void
                                        {y.SetSize(3); y.Elem(0)=1.;y.Elem(1)=1.;y.Elem(2)=1.; });
        u.ProjectCoefficient(vec1_coef);
        p = 0.;
    }

    // ------------------------------------------------------------------
    // 3. System sizes and block layout
    //    NOTE: Sizes are local DOFs per rank.
    // ------------------------------------------------------------------
    int size_1 = u.Size() + p.Size();

    Vector x(size_1);
    x = 0.0;

    Array<int> u_dofs(u.Size()), p_dofs(p.Size());
    std::iota(u_dofs.begin(), u_dofs.end(), 0);
    std::iota(p_dofs.begin(), p_dofs.end(), u.Size());


    x.SetSubVector(u_dofs, u);
    x.SetSubVector(p_dofs, p);

    auto f = [](const mfem::Vector& x, double, mfem::Vector& y) -> void{
            y.SetSize(3);
            y.Elem(0) = 1.;
            y.Elem(1) = 1.;
            y.Elem(2) = 1.;
        };


    StokesRHS rhs(ND,CG,f, f,1.,100.,viscosity);

    // A1 blocks:
    StokesSystem sys(ND, CG, 1., viscosity, 1., 100.);

    StokesSolution y(ND,CG);
    sys.MultTranspose(x,y);

        y.get_p().Print(std::cout);

    ASSERT_EQ(y.Size(), rhs.Size());
    for (int i = 0; i < y.Size(); ++i)
    {
        EXPECT_NEAR(y[i], rhs[i], tol);
    }

    delete fec_ND;
    delete fec_CG;
}


TEST(StokesSystem, ConstantVectorFieldCubeSolver)
{
    // ------------------------------------------------------------------
    // 0. Configuration
    // ------------------------------------------------------------------
    double viscosity = 1.;
    int refinements = 1;
    int order = 1;
    int printlevel = 0;
    double tol = 1e-10;
    std::string mesh_string = std::string("../extern/mfem/data/ref-cube.mesh");

    // ------------------------------------------------------------------
    // 1. Mesh and FE spaces (PARALLEL)
    // ------------------------------------------------------------------
    Mesh mesh(mesh_string.c_str(), 1, 1);
    std::cout << mesh.GetNE() << std::endl;
    for (int l = 0; l < refinements; l++)
    {
        mesh.UniformRefinement();
    }
    std::cout << mesh.GetNE() << std::endl;
    int dim = mesh.Dimension();

    // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
    FiniteElementCollection *fec_ND = new ND_FECollection(order, dim);
    FiniteElementCollection *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);


    auto f =
        [](const mfem::Vector& x, double, mfem::Vector& y) -> void{
            y.SetSize(3);
            y.Elem(0) = 1.;
            y.Elem(1) = 1.;
            y.Elem(2) = 1.;
        };



    // A1 blocks:
    StokesSystem sys(ND, CG, 1., viscosity, 1., 100.);
    StokesRHS rhs(ND, CG, f, f);
    StokesSolution x(ND, CG);

    auto solver = std::make_unique<mfem::GMRESSolver>();
    solver->SetAbsTol(tol);
    solver->SetKDim(3000);
    solver->SetRelTol(0.);
    solver->SetMaxIter(10000);
    solver->SetPrintLevel(1);
    solver->SetOperator(sys);
    solver->Mult(rhs,x);

    

    mfem::VectorFunctionCoefficient exact_u(3,f);
    EXPECT_NEAR(x.get_u().ComputeL2Error(exact_u),0.,1e-10);

    delete fec_ND;
    delete fec_CG;
}


TEST(StokesSystem, VortexVectorFieldCubeSolver)
{
    // ------------------------------------------------------------------
    // 0. Configuration
    // ------------------------------------------------------------------
    double mass = 0.;
    double viscosity = .1;
    int refinements = 1;
    int order = 1;
    int printlevel = 0;
    double tol = 1e-10;
    std::string mesh_string = std::string("../extern/mfem/data/ref-cube.mesh");

    // ------------------------------------------------------------------
    // 1. Mesh and FE spaces (PARALLEL)
    // ------------------------------------------------------------------
    Mesh mesh(mesh_string.c_str(), 1, 1);
    std::cout << mesh.GetNE() << std::endl;
    for (int l = 0; l < refinements; l++)
    {
        mesh.UniformRefinement();
    }
    std::cout << mesh.GetNE() << std::endl;
    int dim = mesh.Dimension();

    // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
    FiniteElementCollection *fec_ND = new ND_FECollection(order, dim);
    FiniteElementCollection *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);


    auto f =
        [](const mfem::Vector& x, double, mfem::Vector& y) -> void{
            y.SetSize(3);
            y.Elem(0) = 0.;
            y.Elem(1) = 0.;
            y.Elem(2) = 0.;
        };    
    auto tr_u =
        [](const mfem::Vector& x, double, mfem::Vector& y) -> void{
            y.SetSize(3);
            y.Elem(0) = -x.Elem(1);
            y.Elem(1) = x.Elem(0);
            y.Elem(2) = 0.;
        };



    // A1 blocks:
    StokesSystem sys(ND, CG, mass, viscosity, 1., 100.);
    StokesRHS rhs(ND, CG, f, tr_u,1.,100.,viscosity);
    StokesSolution x(ND, CG);

    auto solver = std::make_unique<mfem::GMRESSolver>();
    solver->SetAbsTol(tol);
    solver->SetKDim(3000);
    solver->SetRelTol(0.);
    solver->SetMaxIter(10000);
    solver->SetPrintLevel(1);
    solver->SetOperator(sys);
    solver->Mult(rhs,x);

    

    mfem::VectorFunctionCoefficient exact_u(3,tr_u);
    EXPECT_NEAR(x.get_u().ComputeL2Error(exact_u),0.,1e-10);

    delete fec_ND;
    delete fec_CG;
}


TEST(SchurPreconditioner, test1){
       // ------------------------------------------------------------------
    // 0. Configuration
    // ------------------------------------------------------------------
    double viscosity = .02;
    double mass = 1.;
    int refinements = 2;
    int order = 1;
    int printlevel = 0;
    double tol = 1e-5;
    std::string mesh_string = std::string("../extern/mfem/data/ref-cube.mesh");

    // ------------------------------------------------------------------
    // 1. Mesh and FE spaces (PARALLEL)
    // ------------------------------------------------------------------
    Mesh mesh(mesh_string.c_str(), 1, 1);
    std::cout << mesh.GetNE() << std::endl;
    for (int l = 0; l < refinements; l++)
    {
        mesh.UniformRefinement();
    }
    std::cout << mesh.GetNE() << std::endl;
    int dim = mesh.Dimension();

    // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
    FiniteElementCollection *fec_ND = new ND_FECollection(order, dim);
    FiniteElementCollection *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    // ------------------------------------------------------------------
    // 2. Unknowns and gridfunctions (PARALLEL)
    // ------------------------------------------------------------------
    GridFunction u(&ND);
    GridFunction p(&CG);

    // Initial data from user-provided functions
    {
        mfem::VectorFunctionCoefficient vec1_coef(3,[](mfem::Vector x, mfem::Vector &y) -> void
                                        {y.SetSize(3); y.Elem(0)=1.;y.Elem(1)=1.;y.Elem(2)=1.; });
        u.ProjectCoefficient(vec1_coef);
        p = 0.;
    }

    // ------------------------------------------------------------------
    // 3. System sizes and block layout
    //    NOTE: Sizes are local DOFs per rank.
    // ------------------------------------------------------------------
    int size_1 = u.Size() + p.Size();

    Vector x(size_1);
    x = 0.0;

    Array<int> u_dofs(u.Size()), p_dofs(p.Size());
    std::iota(u_dofs.begin(), u_dofs.end(), 0);
    std::iota(p_dofs.begin(), p_dofs.end(), u.Size());


    x.SetSubVector(u_dofs, u);
    x.SetSubVector(p_dofs, p);

    auto f = [](const mfem::Vector& x, double, mfem::Vector& y) -> void{
            y.SetSize(3);
            y.Elem(0) = x.Elem(0)*sin(x.Elem(1));
            y.Elem(1) = x.Elem(1)*x.Elem(2);
            y.Elem(2) = x.Elem(0);
        };


    StokesRHS rhs(ND,CG,f, f, -1., 0.);
    SchurPreconditioner pre(ND,CG,mass,viscosity);

    // A1 blocks:
    StokesSystem sys(ND, CG, mass, viscosity, -1., 0.);
    pre.SetOperator(sys);

    //StokesSolution x(ND,CG);
    auto solver = std::make_unique<mfem::GMRESSolver>();
    solver->SetAbsTol(tol);
    solver->SetKDim(3000);
    solver->SetRelTol(0.);
    solver->SetMaxIter(10000);
    solver->SetPrintLevel(1);
    solver->SetOperator(sys);
    solver->SetPreconditioner(pre);
    solver->Mult(rhs,x);

    //sys.MultTranspose(x,y);

      //  y.get_p().Print(std::cout);

    //ASSERT_EQ(y.Size(), rhs.Size());
    //for (int i = 0; i < y.Size(); ++i)
   // {
    //    EXPECT_NEAR(y[i], rhs[i], tol);
   // }

    delete fec_ND;
    delete fec_CG;
}


TEST(SchurSolver, test1){
       // ------------------------------------------------------------------
    // 0. Configuration
    // ------------------------------------------------------------------
    double viscosity = 1.;
    double mass = 1.;
    int refinements = 2;
    int order = 1;
    int printlevel = 0;
    double tol = 1e-5;
    double Cw = 0.;
    double theta = -1.;
    std::string mesh_string = std::string("../extern/mfem/data/ref-cube.mesh");

    // ------------------------------------------------------------------
    // 1. Mesh and FE spaces (PARALLEL)
    // ------------------------------------------------------------------
    Mesh mesh(mesh_string.c_str(), 1, 1);
    std::cout << mesh.GetNE() << std::endl;
    for (int l = 0; l < refinements; l++)
    {
        mesh.UniformRefinement();
    }
    std::cout << mesh.GetNE() << std::endl;
    int dim = mesh.Dimension();

    // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
    FiniteElementCollection *fec_ND = new ND_FECollection(order, dim);
    FiniteElementCollection *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    // ------------------------------------------------------------------
    // 2. Unknowns and gridfunctions (PARALLEL)
    // ------------------------------------------------------------------
    GridFunction u(&ND);
    GridFunction p(&CG);

    // Initial data from user-provided functions
    {
        mfem::VectorFunctionCoefficient vec1_coef(3,[](mfem::Vector x, mfem::Vector &y) -> void
                                        {y.SetSize(3); y.Elem(0)=1.;y.Elem(1)=1.;y.Elem(2)=1.; });
        u.ProjectCoefficient(vec1_coef);
        p = 0.;
    }

    // ------------------------------------------------------------------
    // 3. System sizes and block layout
    //    NOTE: Sizes are local DOFs per rank.
    // ------------------------------------------------------------------
    int size_1 = u.Size() + p.Size();

    Vector x(size_1);
    x = 0.0;

    Array<int> u_dofs(u.Size()), p_dofs(p.Size());
    std::iota(u_dofs.begin(), u_dofs.end(), 0);
    std::iota(p_dofs.begin(), p_dofs.end(), u.Size());


    x.SetSubVector(u_dofs, u);
    x.SetSubVector(p_dofs, p);

    auto f = [](const mfem::Vector& x, double, mfem::Vector& y) -> void{
            y.SetSize(3);
            y.Elem(0) = x.Elem(0)*sin(x.Elem(1));
            y.Elem(1) = x.Elem(1)*x.Elem(2);
            y.Elem(2) = x.Elem(0);
        };

    auto tr_u = [](const mfem::Vector& x, double, mfem::Vector& y) -> void{
        y.SetSize(3);
        y.Elem(0) = 0.;
        y.Elem(1) = 0.;
        y.Elem(2) = 0.;
    };


    StokesRHS rhs(ND,CG,f, tr_u, theta, Cw,viscosity);

    // A1 blocks:
    StokesSystem sys(ND, CG, mass, viscosity, theta, Cw);
    int iterations;
    SchurSolver solver(ND, CG, mass, viscosity, iterations, 1e-12);
    solver.SetOperator(sys);

    solver.Mult(rhs,x);

    mfem::Vector sys_x(sys.NumRows());
    sys.Mult(x,sys_x);

    for(int i = 0; i < rhs.GetBlock(0).Size(); ++i)
        EXPECT_NEAR(sys_x[i], rhs[i], 1e-6);


    for(int i = rhs.GetBlock(0).Size(); i < rhs.Size(); ++i)
        EXPECT_NEAR(sys_x[i], rhs[i], 1e-6);

    delete fec_ND;
    delete fec_CG;
}



TEST(StokesSystemAndRHS, CompareLHSandRHS){
       // ------------------------------------------------------------------
    // 0. Configuration
    // ------------------------------------------------------------------
    double viscosity = 1.;
    double mass = 0.;
    int refinements = 1;
    int order = 1;
    int printlevel = 0;
    double tol = 1e-5;
    double Cw = 0.;
    double theta = -1e8;
    std::string mesh_string = std::string("../extern/mfem/data/ref-cube.mesh");

    // ------------------------------------------------------------------
    // 1. Mesh and FE spaces (PARALLEL)
    // ------------------------------------------------------------------
    Mesh mesh(mesh_string.c_str(), 1, 1);
    std::cout << mesh.GetNE() << std::endl;
    for (int l = 0; l < refinements; l++)
    {
        mesh.UniformRefinement();
    }
    std::cout << mesh.GetNE() << std::endl;
    int dim = mesh.Dimension();

    // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
    FiniteElementCollection *fec_ND = new ND_FECollection(order, dim);
    FiniteElementCollection *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);



    auto f = [](const mfem::Vector& x, double, mfem::Vector& y) -> void{
            y.SetSize(3);
            y.Elem(0) = 0.;
            y.Elem(1) = 0.;
            y.Elem(2) = 0.;
        };

    auto tr_u = [](const mfem::Vector& x, double, mfem::Vector& y) -> void{
        y.SetSize(3);
        y.Elem(0) = -x.Elem(1);
        y.Elem(1) = x.Elem(0);
        y.Elem(2) = 0.;
    };
    mfem::VectorFunctionCoefficient u_coef(3, tr_u);
    mfem::GridFunction u(&ND);
    u.ProjectCoefficient(u_coef);


    StokesRHS rhs(ND,CG,f, tr_u, theta, Cw,viscosity);


    // A1 blocks:
    StokesSystem sys(ND, CG, mass, viscosity, theta, Cw);
    mfem::Vector sys_u(sys.GetBlock(0,0).Height());
    sys.GetBlock(0,0).Mult(u,sys_u);

    for(int i = 0; i < rhs.GetBlock(0).Size(); ++i)
        EXPECT_NEAR(sys_u[i], rhs[i], rhs.Norml2()*1e-7);
    
    delete fec_ND;
    delete fec_CG;
}
