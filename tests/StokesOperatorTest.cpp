#include <gtest/gtest.h>
#include "mfem.hpp"
#include "StokesOperator.h"

using namespace StokesNitsche;

TEST(StokesOperatorTest, MatrixRegularityGalerkinP1)
{
    const unsigned int n = 4,
                       p = 1;
    const double theta = 1.0,
                 penalty = 10.0,
                 factor = 1.0;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        n, n, n, mfem::Element::TETRAHEDRON
    );

    std::shared_ptr<mfem::Mesh> mesh_ptr =
        std::make_unique<mfem::Mesh>(
            std::move(mesh)
        );
    // Added tau = 0.0
    StokesNitscheOperator op(mesh_ptr, 0.0, 1, p,
                             theta, penalty, factor);

    ASSERT_EQ(op.getOperatorMode(), OperatorMode::Galerkin);
    std::unique_ptr<mfem::SparseMatrix> A = op.getFullGalerkinSystem();

    mfem::Vector ew;
    std::unique_ptr<mfem::DenseMatrix> Ad =
        std::unique_ptr<mfem::DenseMatrix>(
            A->ToDenseMatrix()
        );
    Ad->Eigenvalues(ew);

    ew.Abs();
    const double min_ew = ew.Min();

    // Should be large enough, so that it is not nearly singular
    ASSERT_GT(min_ew, 1e-12);
}

TEST(StokesOperatorTest, MatrixRegularityDECP1)
{
    const unsigned int n = 4,
                       p = 1;
    const double theta = 1.0,
                 penalty = 10.0,
                 factor = 1.0;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        n, n, n, mfem::Element::TETRAHEDRON
    );

    std::shared_ptr<mfem::Mesh> mesh_ptr =
        std::make_unique<mfem::Mesh>(
            std::move(mesh)
        );
    // Added tau = 0.0
    StokesNitscheOperator op(mesh_ptr, 0.0, 1, p,
                             theta, penalty, factor,
                             MassLumping::Diagonal);

    op.setOperatorMode(StokesNitsche::OperatorMode::DEC);

    ASSERT_EQ(op.getOperatorMode(), OperatorMode::DEC);
    std::unique_ptr<mfem::SparseMatrix> A = op.getFullDECSystem();

    mfem::Vector ew;
    std::unique_ptr<mfem::DenseMatrix> Ad =
        std::unique_ptr<mfem::DenseMatrix>(
            A->ToDenseMatrix()
        );
    Ad->Eigenvalues(ew);

    ew.Abs();
    const double min_ew = ew.Min();

    // Should be large enough, so that it is not nearly singular
    ASSERT_GT(min_ew, 1e-12);
}

TEST(StokesOperatorTest, MatrixRegularityGalerkinP2)
{
    const unsigned int n = 3,
                       p = 2;
    const double theta = 1.0,
                 penalty = 10.0,
                 factor = 1.0;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        n, n, n, mfem::Element::TETRAHEDRON
    );

    std::shared_ptr<mfem::Mesh> mesh_ptr =
    std::make_unique<mfem::Mesh>(
        std::move(mesh)
    );
    // Added tau = 0.0
    StokesNitscheOperator op(mesh_ptr, 0.0, 1, p,
                             theta, penalty, factor);

    ASSERT_EQ(op.getOperatorMode(), OperatorMode::Galerkin);
    std::unique_ptr<mfem::SparseMatrix> A = op.getFullGalerkinSystem();

    mfem::Vector ew;
    std::unique_ptr<mfem::DenseMatrix> Ad =
    std::unique_ptr<mfem::DenseMatrix>(
        A->ToDenseMatrix()
    );
    Ad->Eigenvalues(ew);

    ew.Abs();
    const double min_ew = ew.Min();

    // Should be large enough, so that it is not nearly singular
    ASSERT_GT(min_ew, 1e-12);
}

TEST(StokesOperatorTest, MatrixRegularityDECP2)
{
    const unsigned int n = 3,
                       p = 2;
    const double theta = 1.0,
                 penalty = 10.0,
                 factor = 1.0;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        n, n, n, mfem::Element::TETRAHEDRON
    );

    std::shared_ptr<mfem::Mesh> mesh_ptr =
    std::make_unique<mfem::Mesh>(
        std::move(mesh)
    );
    // Added tau = 0.0
    StokesNitscheOperator op(mesh_ptr, 0.0, 1, p,
                             theta, penalty, factor,
                             MassLumping::Diagonal);

    op.setOperatorMode(StokesNitsche::OperatorMode::DEC);

    ASSERT_EQ(op.getOperatorMode(), OperatorMode::DEC);
    std::unique_ptr<mfem::SparseMatrix> A = op.getFullDECSystem();

    mfem::Vector ew;
    std::unique_ptr<mfem::DenseMatrix> Ad =
    std::unique_ptr<mfem::DenseMatrix>(
        A->ToDenseMatrix()
    );
    Ad->Eigenvalues(ew);

    ew.Abs();
    const double min_ew = ew.Min();

    // Should be large enough, so that it is not nearly singular
    ASSERT_GT(min_ew, 1e-12);
}

TEST(StokesOperatorTest, OperatorGalerkinP1)
{
    const unsigned int n = 5, p = 1;
    const double theta = 1.0,
                 penalty = 3.0,
                 factor = 0.0;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        n, n, n, mfem::Element::TETRAHEDRON
    );

    std::shared_ptr<mfem::Mesh> mesh_ptr =
        std::make_unique<mfem::Mesh>(
            std::move(mesh)
        );
    // Added tau = 0.0
    StokesNitscheOperator op(mesh_ptr, 0.0, 1, p,
                             theta, penalty, factor);

    const int nv = op.getH1Space().GetNDofs(),
              ne = op.getHCurlSpace().GetNDofs();

    ASSERT_EQ(op.getOperatorMode(), OperatorMode::Galerkin);

    std::unique_ptr<mfem::SparseMatrix> A = op.getFullSystem();

    mfem::Vector x_(nv + ne + 1),
                 y_op(nv + ne),
                 y_mat(nv + ne);
    x_(nv + ne) = 0;
    mfem::Vector x(x_, 0, nv + ne);
    x.Randomize(1);

    op.Mult(x, y_op);

    mfem::Vector y_extended(ne + nv + 1);
    A->Mult(x_, y_extended);

    y_mat.MakeRef(y_extended, 0, ne + nv);

    mfem::Vector y_err(y_mat);
    y_err -= y_op;

    ASSERT_NEAR(y_err.Norml2() / y_op.Norml2(), 0., 1e-12);
}

TEST(StokesOperatorTest, OperatorDECP1)
{
    const unsigned int n = 5, p = 1;
    const double theta = 1.0,
                 penalty = 3.0,
                 factor = 1.0;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        n, n, n, mfem::Element::TETRAHEDRON
    );

    std::shared_ptr<mfem::Mesh> mesh_ptr =
        std::make_unique<mfem::Mesh>(
            std::move(mesh)
        );
    // Added tau = 0.0
    StokesNitscheOperator op(mesh_ptr, 0.0, 1, p,
                             theta, penalty, factor,
                             MassLumping::Diagonal);
    op.setOperatorMode(StokesNitsche::OperatorMode::DEC);

    const int nv = op.getH1Space().GetNDofs(),
              ne = op.getHCurlSpace().GetNDofs();

    ASSERT_EQ(op.getOperatorMode(), OperatorMode::DEC);

    std::unique_ptr<mfem::SparseMatrix> A = op.getFullSystem();

    mfem::Vector x_(nv + ne + 1),
                 y_op(nv + ne),
                 y_mat(nv + ne);
    x_(nv + ne) = 0;
    mfem::Vector x(x_, 0, nv + ne);
    x.Randomize(1);

    op.Mult(x, y_op);

    mfem::Vector y_extended(ne + nv + 1);
    A->Mult(x_, y_extended);

    y_mat.MakeRef(y_extended, 0, ne + nv);

    mfem::Vector y_err(y_mat);
    y_err -= y_op;

    ASSERT_NEAR(y_err.Norml2() / y_op.Norml2(), 0., 1e-12);
}

TEST(StokesOperatorTest, OperatorGalerkinP2)
{
    const unsigned int n = 5, p = 2;
    const double theta = 1.0,
    penalty = 3.0,
    factor = 0.0;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        n, n, n, mfem::Element::TETRAHEDRON
    );

    std::shared_ptr<mfem::Mesh> mesh_ptr =
    std::make_unique<mfem::Mesh>(
        std::move(mesh)
    );
    // Added tau = 0.0
    StokesNitscheOperator op(mesh_ptr, 0.0, 1, p,
                             theta, penalty, factor);

    const int nv = op.getH1Space().GetNDofs(),
              ne = op.getHCurlSpace().GetNDofs();

    ASSERT_EQ(op.getOperatorMode(), OperatorMode::Galerkin);

    std::unique_ptr<mfem::SparseMatrix> A = op.getFullSystem();

    mfem::Vector x_(nv + ne + 1),
                 y_op(nv + ne),
                 y_mat(nv + ne);
                 x_(nv + ne) = 0;
    mfem::Vector x(x_, 0, nv + ne);
    x.Randomize(1);

    op.Mult(x, y_op);

    mfem::Vector y_extended(ne + nv + 1);
    A->Mult(x_, y_extended);

    y_mat.MakeRef(y_extended, 0, ne + nv);

    mfem::Vector y_err(y_mat);
    y_err -= y_op;

    ASSERT_NEAR(y_err.Norml2() / y_op.Norml2(), 0., 1e-12);
}

TEST(StokesOperatorTest, OperatorDECP2)
{
    const unsigned int n = 5, p = 1;
    const double theta = 1.0,
    penalty = 3.0,
    factor = 1.0;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        n, n, n, mfem::Element::TETRAHEDRON
    );

    std::shared_ptr<mfem::Mesh> mesh_ptr =
    std::make_unique<mfem::Mesh>(
        std::move(mesh)
    );
    // Added tau = 0.0
    StokesNitscheOperator op(mesh_ptr, 0.0, 1, p,
                             theta, penalty, factor,
                             MassLumping::Diagonal);
    op.setOperatorMode(StokesNitsche::OperatorMode::DEC);

    const int nv = op.getH1Space().GetNDofs(),
              ne = op.getHCurlSpace().GetNDofs();

    ASSERT_EQ(op.getOperatorMode(), OperatorMode::DEC);

    std::unique_ptr<mfem::SparseMatrix> A = op.getFullSystem();

    mfem::Vector x_(nv + ne + 1),
                 y_op(nv + ne),
                 y_mat(nv + ne);
                 x_(nv + ne) = 0;
    mfem::Vector x(x_, 0, nv + ne);
    x.Randomize(1);

    op.Mult(x, y_op);

    mfem::Vector y_extended(ne + nv + 1);
    A->Mult(x_, y_extended);

    y_mat.MakeRef(y_extended, 0, ne + nv);

    mfem::Vector y_err(y_mat);
    y_err -= y_op;

    ASSERT_NEAR(y_err.Norml2() / y_op.Norml2(), 0., 1e-12);
}
