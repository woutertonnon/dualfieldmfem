#include "StokesOperator.h"

#include <iostream>
#include <utility>

#include "BoundaryOperators.h"
#include "incidence.h"

namespace StokesNitsche
{

void StokesNitscheOperator::initFESpaces()
{
    MFEM_VERIFY(order_ > 0, "StokesNitscheOperator: order == 0, use order > 0");
    const int dim = mesh_->Dimension();

    h1_fec_ = std::make_unique<mfem::H1_FECollection>(order_, dim);

    hcurl_fec_ = std::make_unique<mfem::ND_FECollection>(order_, dim);

    if (dim == 2)
    {
        // 2D De Rham: the scalar curl maps ND(order_) -> L2(order_-1).
        // For order_==1 this is L2 order 0 (the original lowest-order case);
        // for order_>=2 it must track the ND order so assembleDiscreteCurl's
        // order check (L2 order == ND order - 1) holds.
        hdiv_or_l2_fec_ = std::make_unique<mfem::L2_FECollection>(
            order_ - 1, dim, mfem::BasisType::GaussLegendre,
            mfem::FiniteElement::INTEGRAL);
    }
    else
    {
        hdiv_or_l2_fec_ =
            std::make_unique<mfem::RT_FECollection>(order_ - 1, dim);
    }

    h1_space_ =
        std::make_unique<mfem::FiniteElementSpace>(mesh_.get(), h1_fec_.get());
    hcurl_space_ = std::make_unique<mfem::FiniteElementSpace>(
        mesh_.get(), hcurl_fec_.get());
    hdiv_or_l2_space_ = std::make_unique<mfem::FiniteElementSpace>(
        mesh_.get(), hdiv_or_l2_fec_.get());

    // Need to do this, cannot (easily) be done in the constructor
    this->height = h1_space_->GetNDofs() + hcurl_space_->GetNDofs();
    this->width  = this->height;
}

void StokesNitscheOperator::initIncidence()
{
    d0_ = assembleDiscreteGradient(h1_space_.get(), hcurl_space_.get());
    d1_ = assembleDiscreteCurl(hcurl_space_.get(), hdiv_or_l2_space_.get());
}

void StokesNitscheOperator::initMass()
{
    mass_h1_    = std::make_unique<mfem::BilinearForm>(h1_space_.get());
    mass_hcurl_ = std::make_unique<mfem::BilinearForm>(hcurl_space_.get());
    mass_hdiv_or_l2_ =
        std::make_unique<mfem::BilinearForm>(hdiv_or_l2_space_.get());

    mfem::ConstantCoefficient one(1.0);

    // If we need RowSums, we CANNOT use partial assembly because we need the
    // explicit sparse matrix.
    mfem::AssemblyLevel assembly_level = mfem::AssemblyLevel::LEGACY;
    if (ml_ != MassLumping::RowSum && false)
        assembly_level = mfem::AssemblyLevel::PARTIAL;

    mass_h1_->SetAssemblyLevel(assembly_level);
    mass_hcurl_->SetAssemblyLevel(assembly_level);
    mass_hdiv_or_l2_->SetAssemblyLevel(assembly_level);

    mass_h1_->AddDomainIntegrator(new mfem::MassIntegrator(one));
    mass_hcurl_->AddDomainIntegrator(new mfem::VectorFEMassIntegrator(one));

    if (mesh_->Dimension() == 2)
        mass_hdiv_or_l2_->AddDomainIntegrator(new mfem::MassIntegrator(one));
    else
        mass_hdiv_or_l2_->AddDomainIntegrator(
            new mfem::VectorFEMassIntegrator(one));

    mass_h1_->Assemble();
    mass_hcurl_->Assemble();
    mass_hdiv_or_l2_->Assemble();

    mass_h1_->Finalize();
    mass_hcurl_->Finalize();
    mass_hdiv_or_l2_->Finalize();
}

void StokesNitscheOperator::initLumpedMass()
{
    mass_h1_lumped_.SetSize(h1_space_->GetNDofs());
    mass_hcurl_lumped_.SetSize(hcurl_space_->GetNDofs());
    mass_hdiv_or_l2_lumped_.SetSize(hdiv_or_l2_space_->GetNDofs());

    switch (ml_)
    {
        case MassLumping::None:
            break;
        case MassLumping::Diagonal:
            mass_h1_->AssembleDiagonal(mass_h1_lumped_);
            mass_hcurl_->AssembleDiagonal(mass_hcurl_lumped_);
            mass_hdiv_or_l2_->AssembleDiagonal(mass_hdiv_or_l2_lumped_);
            break;
        case MassLumping::Barycentric:
            throw std::logic_error(
                "Barycentric mass lumping not implemented (yet)");
            break;
        case MassLumping::RowSum:
        {
            auto sum_rows =
                [&](const mfem::SparseMatrix& A, mfem::Vector& abs_row_sums)
            {
                abs_row_sums = 0.0;
#pragma omp parallel for
                for (int i = 0; i < A.Height(); ++i)
                {
                    const int     nnz  = A.RowSize(i);
                    const double* vals = A.GetRowEntries(i);
                    double        s    = 0.0;
                    for (int k = 0; k < nnz; ++k)
                        s += std::abs(vals[k]);
                    abs_row_sums[i] = s;
                }
            };

            // This only works because we turned off PA for RowSum in initMass()
            sum_rows(mass_h1_->SpMat(), mass_h1_lumped_);
            sum_rows(mass_hcurl_->SpMat(), mass_hcurl_lumped_);
            sum_rows(mass_hdiv_or_l2_->SpMat(), mass_hdiv_or_l2_lumped_);
            break;
        }
        default:
            throw std::runtime_error("Unreachable!");
    }
}

void StokesNitscheOperator::initNitsche(
    const double theta,
    const double penalty,
    const double factor)
{
    nitsche_ = std::make_unique<mfem::BilinearForm>(hcurl_space_.get());

    nitsche_->AddBdrFaceIntegrator(
        new ND_NitscheIntegrator(theta, penalty, factor));

    nitsche_->Assemble();
    nitsche_->Finalize();
}

StokesNitscheOperator::StokesNitscheOperator(
    std::shared_ptr<mfem::Mesh> mesh_ptr,
    const double                tau,
    const unsigned              order,
    const double                theta,
    const double                penalty,
    const double                factor,
    const MassLumping           ml,
    const mfem::Array<int>*     outflow_marker,
    const double                outflow_penalty)
    : mfem::Operator(), tau_(tau), order_(order), theta_(theta),
      penalty_(penalty), mesh_(mesh_ptr), ml_(ml),
      outflow_penalty_(outflow_penalty)
{
    if (outflow_marker && outflow_marker->Size() > 0)
        outflow_marker_ = *outflow_marker;

    initFESpaces();
    initIncidence();
    initMass();
    initLumpedMass();
    initNitsche(theta, penalty, factor);
    initOutflow();

    offsets_.SetSize(3);
    offsets_[0] = 0;
    offsets_[1] = hcurl_space_->GetNDofs();
    offsets_[2] = hcurl_space_->GetNDofs() + h1_space_->GetNDofs();
}

void StokesNitscheOperator::initOutflow()
{
    if (outflow_marker_.Size() == 0)
        return;

    // b_out_ = <u.n, q> over the outflow faces, assembled as a MixedBilinearForm
    // with trial = ND (velocity) and test = H1 (pressure) -> (nv x ne) matrix.
    {
        auto B = std::make_unique<mfem::MixedBilinearForm>(
            hcurl_space_.get(), h1_space_.get());
        B->AddBdrFaceIntegrator(new ND_NormalScalarBdrIntegrator(),
                                outflow_marker_);
        B->Assemble();
        B->Finalize();
        b_out_ = std::unique_ptr<mfem::SparseMatrix>(B->LoseMat());
    }

    // m_out_ = gamma * <p, q> boundary mass over the outflow faces (nv x nv).
    {
        auto M = std::make_unique<mfem::BilinearForm>(h1_space_.get());
        mfem::ConstantCoefficient gamma(outflow_penalty_);
        M->AddBoundaryIntegrator(new mfem::BoundaryMassIntegrator(gamma),
                                 outflow_marker_);
        M->Assemble();
        M->Finalize();
        m_out_ = std::unique_ptr<mfem::SparseMatrix>(M->LoseMat());
    }
}

// -------------------------------------------------------------------------
// System Generation
// -------------------------------------------------------------------------

std::unique_ptr<mfem::SparseMatrix>
StokesNitscheOperator::getFullGalerkinSystem() const
{
    mfem::ConstantCoefficient one(1.0);
    const int                 nv = h1_space_->GetNDofs();
    const int                 ne = hcurl_space_->GetNDofs();

    const mfem::Array<int> offsets({0, ne, ne + nv, ne + nv + 1});
    mfem::BlockMatrix      block(offsets);

    std::unique_ptr<mfem::SparseMatrix> curlcurl, grad, gradT;

    // Gradient Block (G)
    {
        auto G = std::make_unique<mfem::MixedBilinearForm>(
            h1_space_.get(), hcurl_space_.get());
        G->AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator(one));
        G->Assemble();
        G->Finalize();
        grad = std::unique_ptr<mfem::SparseMatrix>(G->LoseMat());
    }

    // -Divergence Block (G^T)
    gradT = std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(*grad));

    // CurlCurl Block + Nitsche + Mass (Time-stepping)
    {
        auto cc = std::make_unique<mfem::BilinearForm>(hcurl_space_.get());
        cc->AddDomainIntegrator(new mfem::CurlCurlIntegrator(one));
        cc->Assemble();
        cc->Finalize();

        auto cc_nitsche = std::unique_ptr<mfem::SparseMatrix>(
            mfem::Add(cc->SpMat(), nitsche_->SpMat()));

        if (tau_ != 0.0)
        {
            // Add tau * Mass to the block
            curlcurl = std::unique_ptr<mfem::SparseMatrix>(
                mfem::Add(1.0, *cc_nitsche, tau_, mass_hcurl_->SpMat()));
        }
        else
        {
            curlcurl = std::move(cc_nitsche);
        }
    }

    // Consistent-Nitsche outflow (consistent weak form, no mass scaling):
    //   (0,1) = grad - b^T,  (1,0) = gradT - b,  (1,1) = -m_out.
    if (outflow_marker_.Size() > 0)
    {
        auto bt = std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(*b_out_));
        auto grad_o =
            std::unique_ptr<mfem::SparseMatrix>(mfem::Add(1.0, *grad, -1.0, *bt));
        auto gradT_o = std::unique_ptr<mfem::SparseMatrix>(
            mfem::Add(1.0, *gradT, -1.0, *b_out_));
        auto pblk = std::make_unique<mfem::SparseMatrix>(*m_out_);
        *pblk *= -1.0;

        const mfem::Array<int> offs2({0, ne, ne + nv});
        mfem::BlockMatrix       block2(offs2);
        block2.SetBlock(0, 0, curlcurl.get());
        block2.SetBlock(0, 1, grad_o.get());
        block2.SetBlock(1, 0, gradT_o.get());
        block2.SetBlock(1, 1, pblk.get());
        return std::unique_ptr<mfem::SparseMatrix>(block2.CreateMonolithic());
    }

    // Mean Constraint Block
    mfem::Array<int> cols(nv);
    for (int k = 0; k < nv; ++k)
        cols[k] = k;

    mfem::SparseMatrix mean(1, nv);

    mfem::GridFunction ones(h1_space_.get());
    ones.ProjectCoefficient(one);

    mfem::Vector mass_x_ones(nv);

    // Mult works perfectly with both Partial and Full assembly!
    mass_h1_->Mult(ones, mass_x_ones);

    mean.AddRow(0, cols, mass_x_ones);
    mean.Finalize();

    auto meanT = std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(mean));

    // Assemble Block Matrix
    block.SetBlock(0, 0, curlcurl.get());
    block.SetBlock(0, 1, grad.get());
    block.SetBlock(1, 0, gradT.get());
    block.SetBlock(2, 1, &mean);
    block.SetBlock(1, 2, meanT.get());

    return std::unique_ptr<mfem::SparseMatrix>(block.CreateMonolithic());
}

std::unique_ptr<mfem::SparseMatrix> StokesNitscheOperator::getFullDECSystem()
    const
{
    const int nv = h1_space_->GetNDofs();
    const int ne = hcurl_space_->GetNDofs();

    const mfem::Array<int> offsets({0, ne, ne + nv, ne + nv + 1});
    mfem::BlockMatrix      block(offsets);

    // Gradient (d0)
    auto grad = std::make_unique<mfem::SparseMatrix>(d0_);

    // Divergence (scaled d0^T)
    mfem::Vector inv_mass_h1(mass_h1_lumped_);
    inv_mass_h1.Reciprocal();

    auto gradT = std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(*grad));
    gradT->ScaleRows(inv_mass_h1);
    gradT->ScaleColumns(mass_hcurl_lumped_);

    // CurlCurl + Nitsche + Time-stepping ID
    std::unique_ptr<mfem::SparseMatrix> curlcurl;
    {
        auto tmp = std::make_unique<mfem::SparseMatrix>(d1_);
        tmp->ScaleRows(mass_hdiv_or_l2_lumped_);

        auto d1T = std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(d1_));

        auto product =
            std::unique_ptr<mfem::SparseMatrix>(mfem::Mult(*d1T, *tmp));

        curlcurl = std::unique_ptr<mfem::SparseMatrix>(
            mfem::Add(*product, nitsche_->SpMat()));

        mfem::Vector inv_mass_hcurl(mass_hcurl_lumped_);
        inv_mass_hcurl.Reciprocal();
        curlcurl->ScaleRows(inv_mass_hcurl);

        if (tau_ != 0.0)
        {
            // Build an explicit tau * I matrix and add it
            mfem::SparseMatrix id(curlcurl->Height());
            for (int i = 0; i < curlcurl->Height(); ++i)
                id.Add(i, i, tau_);
            id.Finalize();

            curlcurl =
                std::unique_ptr<mfem::SparseMatrix>(mfem::Add(*curlcurl, id));
        }
    }

    // Consistent-Nitsche outflow: replace the zero-mean Lagrange constraint by
    // the boundary terms (the penalty fixes the pressure datum). DEC scaling:
    //   (0,1) = d0 - M_hcurl^{-1} b^T,  (1,0) = gradT - M_h1^{-1} b,
    //   (1,1) = -M_h1^{-1} m_out.
    if (outflow_marker_.Size() > 0)
    {
        mfem::Vector inv_mass_hcurl(mass_hcurl_lumped_);
        inv_mass_hcurl.Reciprocal();

        auto bt = std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(*b_out_));
        bt->ScaleRows(inv_mass_hcurl);
        auto grad_o =
            std::unique_ptr<mfem::SparseMatrix>(mfem::Add(1.0, *grad, -1.0, *bt));

        auto b_sc = std::make_unique<mfem::SparseMatrix>(*b_out_);
        b_sc->ScaleRows(inv_mass_h1);
        auto gradT_o = std::unique_ptr<mfem::SparseMatrix>(
            mfem::Add(1.0, *gradT, -1.0, *b_sc));

        auto pblk = std::make_unique<mfem::SparseMatrix>(*m_out_);
        pblk->ScaleRows(inv_mass_h1);
        *pblk *= -1.0;

        const mfem::Array<int> offs2({0, ne, ne + nv});
        mfem::BlockMatrix       block2(offs2);
        block2.SetBlock(0, 0, curlcurl.get());
        block2.SetBlock(0, 1, grad_o.get());
        block2.SetBlock(1, 0, gradT_o.get());
        block2.SetBlock(1, 1, pblk.get());
        return std::unique_ptr<mfem::SparseMatrix>(block2.CreateMonolithic());
    }

    // Mean Constraint
    mfem::Array<int> cols(nv);
    for (int k = 0; k < nv; ++k)
        cols[k] = k;

    mfem::SparseMatrix mean(1, nv);
    mean.AddRow(0, cols, mass_h1_lumped_);
    mean.Finalize();

    auto meanT = std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(mean));

    // Assemble
    block.SetBlock(0, 0, curlcurl.get());
    block.SetBlock(0, 1, grad.get());
    block.SetBlock(1, 0, gradT.get());
    block.SetBlock(2, 1, &mean);
    block.SetBlock(1, 2, meanT.get());

    return std::unique_ptr<mfem::SparseMatrix>(block.CreateMonolithic());
}

std::unique_ptr<mfem::SparseMatrix> StokesNitscheOperator::getFullSystem() const
{
    if (opmode_ == OperatorMode::Galerkin)
        return getFullGalerkinSystem();
    else
        return getFullDECSystem();
}

void StokesNitscheOperator::eliminateConstants(mfem::Vector& x) const
{
    MFEM_ASSERT(x.Size() == this->NumCols(), "Vector size mismatch");

    const int ne = hcurl_space_->GetNDofs();
    const int nv = h1_space_->GetNDofs();

    // x_p wraps the pressure portion of x. Modifying x_p modifies x directly.
    mfem::Vector x_p(x.GetData() + ne, nv);

    // Represent the constant function f(x) = 1 in the FE space.
    mfem::GridFunction        ones(h1_space_.get());
    mfem::ConstantCoefficient one_coeff(1.0);
    ones.ProjectCoefficient(one_coeff);

    // Vector to hold the result of the mass matrix applied to 'ones'
    mfem::Vector M_ones(nv);

    if (opmode_ == OperatorMode::Galerkin)
    {
        // Mult works perfectly with both Partial and Full assembly!
        mass_h1_->Mult(ones, M_ones);
    }
    else  // DEC
    {
        MFEM_ASSERT(
            ml_ != MassLumping::None,
            "EliminateConstants: Need mass lumping in DEC mode");

        M_ones = ones;
        M_ones *= mass_h1_lumped_;
    }

    // Compute orthogonal projection
    const double denom = M_ones * ones;
    const double num   = M_ones * x_p;
    const double proj  = num / denom;

    // Subtract the constant mode from the pressure solution in-place
    x_p.Add(-proj, ones);
}

// -------------------------------------------------------------------------
// Operations
// -------------------------------------------------------------------------

void StokesNitscheOperator::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    if (opmode_ == OperatorMode::Galerkin)
        MultGalerkin(x, y);
    else
        MultDEC(x, y);
}

void StokesNitscheOperator::MultDEC(const mfem::Vector& x, mfem::Vector& y)
    const
{
    const int nv = h1_space_->GetNDofs();
    const int ne = hcurl_space_->GetNDofs();

    MFEM_ASSERT(
        x.Size() == nv + ne && y.Size() == nv + ne, "Vector size mismatch");

    const mfem::Vector x_u(x.GetData(), ne);
    const mfem::Vector x_p(x.GetData() + ne, nv);

    mfem::Vector y_u(y.GetData(), ne);
    mfem::Vector y_p(y.GetData() + ne, nv);

    mfem::Vector tmp_u(y_u.Size());
    mfem::Vector tmp_du(hdiv_or_l2_space_->GetNDofs());

    // Curl-Curl part
    d1_.Mult(x_u, tmp_du);
    tmp_du *= mass_hdiv_or_l2_lumped_;
    d1_.MultTranspose(tmp_du, y_u);

    nitsche_->AddMult(x_u, y_u);

    // Outflow symmetric term -<p, v.n> (weak; gets the 1/M_hcurl scaling below).
    if (outflow_marker_.Size() > 0)
        b_out_->AddMultTranspose(x_p, y_u, -1.0);

    y_u /= mass_hcurl_lumped_;

    // Gradient part
    d0_.AddMult(x_p, y_u);

    // Divergence part
    tmp_u = x_u;
    tmp_u *= mass_hcurl_lumped_;
    d0_.MultTranspose(tmp_u, y_p);

    // Outflow consistency -<u.n,q> + penalty -gamma<p,q> (1/M_h1-scaled below).
    if (outflow_marker_.Size() > 0)
    {
        b_out_->AddMult(x_u, y_p, -1.0);
        m_out_->AddMult(x_p, y_p, -1.0);
    }

    y_p /= mass_h1_lumped_;

    // Time (L2 Mass via ID scaling in DEC)
    if (tau_ != 0.0)
        y_u.Add(tau_, x_u);
}

void StokesNitscheOperator::MultGalerkin(const mfem::Vector& x, mfem::Vector& y)
    const
{
    const int nv = h1_space_->GetNDofs();
    const int ne = hcurl_space_->GetNDofs();

    MFEM_ASSERT(
        x.Size() == nv + ne && y.Size() == nv + ne, "Vector size mismatch");

    const mfem::Vector x_u(x.GetData(), ne);
    const mfem::Vector x_p(x.GetData() + ne, nv);

    mfem::Vector y_u(y.GetData(), ne);
    mfem::Vector y_p(y.GetData() + ne, nv);

    mfem::Vector tmp_u(y_u.Size());
    mfem::Vector tmp_du(hdiv_or_l2_space_->GetNDofs());
    mfem::Vector tmp_mdu(hdiv_or_l2_space_->GetNDofs());

    // Curl-Curl Term
    d1_.Mult(x_u, tmp_du);
    mass_hdiv_or_l2_->Mult(tmp_du, tmp_mdu);
    d1_.MultTranspose(tmp_mdu, y_u);

    // Gradient Term
    d0_.Mult(x_p, tmp_u);
    mass_hcurl_->AddMult(tmp_u, y_u);

    // Divergence Term
    mass_hcurl_->Mult(x_u, tmp_u);
    d0_.MultTranspose(tmp_u, y_p);

    // Nitsche Boundary Term
    nitsche_->AddMult(x_u, y_u);

    // Outflow consistent-Nitsche terms (consistent weak form, no mass scaling):
    //   y_u -= <p, v.n>,  y_p -= <u.n, q> + gamma<p,q>
    if (outflow_marker_.Size() > 0)
    {
        b_out_->AddMultTranspose(x_p, y_u, -1.0);
        b_out_->AddMult(x_u, y_p, -1.0);
        m_out_->AddMult(x_p, y_p, -1.0);
    }

    // Time (L2 Mass)
    if (tau_ != 0.0)
        mass_hcurl_->AddMult(x_u, y_u, tau_);
}

}  // namespace StokesNitsche
