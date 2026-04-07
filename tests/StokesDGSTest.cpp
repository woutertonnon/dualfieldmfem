#include <gtest/gtest.h>
#include "mfem.hpp"
#include "StokesDGS.hpp"
#include "StokesOperator.hpp"

void testResidualComp(const unsigned n,
                      const unsigned p,
                      const mfem::Element::Type el_type,
                      const double penalty = 10.0,
                      const double tol = 1e-12)
{
    const double theta = 1.0,
                 factor = 1.0;

    auto mesh_ptr = std::make_shared<mfem::Mesh>(
        mfem::Mesh::MakeCartesian3D(n, n, n, el_type)
    );

    // Added tau = 0.0
    auto op_ptr = std::make_shared<StokesNitsche::StokesNitscheOperator>(
        mesh_ptr, 0.0, p, theta, penalty, factor
    );

    op_ptr->setOperatorMode(StokesNitsche::OperatorMode::DEC);
    ASSERT_EQ(op_ptr->getOperatorMode(), StokesNitsche::OperatorMode::DEC);

    StokesNitsche::StokesNitscheDGS dgs(op_ptr);

    std::unique_ptr<mfem::SparseMatrix> A = op_ptr->getFullSystem();

    const unsigned int nv = op_ptr->getH1Space().GetNDofs(),
                       ne = op_ptr->getHCurlSpace().GetNDofs();

    mfem::Vector rhs(nv + ne),
                 sol(nv + ne),
                 x(nv + ne),
                 residual_mat(nv + ne),
                 residual_op(nv + ne);

    sol.Randomize(1);
    op_ptr->Mult(sol, rhs);
    x.Randomize(2);

    mfem::Vector x_extended(nv + ne + 1),
                 residual_extended(nv + ne + 1);

    for (unsigned int k = 0; k < nv + ne; ++k)
    {
        x_extended(k) = x(k);
        residual_extended(k) = rhs(k);
    }
    x_extended(nv + ne) = 0;

    A->AddMult(x_extended, residual_extended, -1.0);
    residual_mat.MakeRef(residual_extended, 0, ne + nv);

    residual_op = rhs;
    op_ptr->AddMult(x, residual_op, -1.0);

    const double res_norm_mat = residual_mat.Norml2() / rhs.Norml2(),
                 res_norm_dgs = dgs.computeResidualNorm(rhs, x) / rhs.Norml2(),
                 res_norm_op  = residual_op.Norml2() / rhs.Norml2();

    ASSERT_NEAR(res_norm_dgs, res_norm_op, 1e-12)
        << "Failed at order " << p << std::endl;
    ASSERT_NEAR(res_norm_dgs, res_norm_mat, 1e-12)
        << "Failed at order " << p << std::endl;
}

void testConvergence(const unsigned n,
                     const unsigned p,
                     const mfem::Element::Type el_type,
                     const double penalty = 10.0,
                     const double tol = 1e-12)
{
    const double theta = 1.0,
                 factor = 1.0;

    auto mesh_ptr = std::make_shared<mfem::Mesh>(
        mfem::Mesh::MakeCartesian3D(n, n, n, el_type)
    );

    // Added tau = 0.0
    auto op_ptr = std::make_shared<StokesNitsche::StokesNitscheOperator>(
        mesh_ptr, 0.0, p, theta, penalty, factor
    );

    op_ptr->setOperatorMode(StokesNitsche::OperatorMode::DEC);
    ASSERT_EQ(op_ptr->getOperatorMode(), StokesNitsche::OperatorMode::DEC);

    std::unique_ptr<mfem::SparseMatrix> A = op_ptr->getFullDECSystem();
    auto solver = std::make_unique<mfem::UMFPackSolver>(*A);

    const unsigned int nv = op_ptr->getH1Space().GetNDofs(),
                       ne = op_ptr->getHCurlSpace().GetNDofs();

    mfem::Vector rhs(nv + ne),
                 sol(nv + ne),
                 sol_dgs(nv + ne),
                 sol_lu(nv + ne),
                 residual(nv + ne);

    sol.Randomize(1);
    op_ptr->Mult(sol, rhs);

    mfem::Vector rhs_extended(nv + ne + 1),
                 sol_extended(nv + ne + 1);

    for (unsigned int k = 0; k < nv + ne; ++k) rhs_extended(k) = rhs(k);
    rhs_extended(nv + ne) = 0;

    solver->Mult(rhs_extended, sol_extended);
    sol_lu.MakeRef(sol_extended, 0, ne + nv);

    mfem::Vector residual_lu = rhs;
    op_ptr->AddMult(sol_lu, residual_lu, -1.0);

    ASSERT_NEAR(sol_extended(nv + ne) / sol_extended.Norml2(), 0., tol);
    ASSERT_LT(residual_lu.Norml2() / rhs.Norml2(), tol);

    StokesNitsche::StokesNitscheDGS dgs(op_ptr);

    sol_dgs.Randomize(2);

    const unsigned int maxit = 10000;
    double err = tol + 1;
    unsigned int iter;
    mfem::Vector residual_dgs;

    for (iter = 0; iter < maxit && err > tol; ++iter)
    {
        residual_dgs = rhs;
        op_ptr->AddMult(sol_dgs, residual_dgs, -1.0);
        ASSERT_EQ(residual_dgs.CheckFinite(), 0) << "Failed at order " << p << std::endl;

        err = residual_dgs.Norml2() / rhs.Norml2();

        if (!(iter % 100) || !(err > tol))
        {
            std::cout << iter << "\t Rel. residual: " << err << std::endl;
        }

        dgs.Mult(rhs, sol_dgs);
        ASSERT_EQ(sol_dgs.CheckFinite(), 0);
    }

    ASSERT_LT(iter, maxit) << "Failed at order " << p << std::endl;
    ASSERT_LT(residual_dgs.Norml2() / rhs.Norml2(), tol) << "Failed at order " << p << std::endl;
}

TEST(StokesDGSTest, ResidualComputationHex)
{
    for(unsigned p = 1; p <= 3; ++p)
        testResidualComp(3, p, mfem::Element::HEXAHEDRON);
}

TEST(StokesDGSTest, ResidualComputationTetra)
{
    for(unsigned p = 1; p <= 3; ++p)
        testResidualComp(3, p, mfem::Element::TETRAHEDRON);
}


TEST(StokesDGSTest, ConvergenceTetra)
{
    for(unsigned p = 1; p <= 3; ++p)
    {
        std::cout << "Order " << p << std::endl;
        testConvergence(3, p, mfem::Element::TETRAHEDRON, 10 * p * p);
    }
}