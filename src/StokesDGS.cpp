#include "StokesDGS.h"

namespace StokesNitsche
{

double StokesNitscheDGS::getTau() const
{
    return op_->getTau();
}

void StokesNitscheDGS::setTau(const double tau)
{
    op_->setTau(tau);
    
    // Completely rebuild the transformation and smoothers with the new tau
    initTransformation();
    initTransformedSystem();
    initSmoothers();
}

void StokesNitscheDGS::initTransformation()
{
    T_ = std::make_unique<mfem::BlockOperator>(op_->getOffsets());

    grad_adj_ =
        std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(op_->getD0()));

    mfem::Vector inv_mass_hcurl_lumped = op_->getMassHCurlLumped();
    mfem::Vector inv_mass_h1_lumped    = op_->getMassH1Lumped();

    inv_mass_hcurl_lumped.Reciprocal();
    inv_mass_h1_lumped.Reciprocal();

    grad_adj_->ScaleRows(inv_mass_h1_lumped);
    grad_adj_->ScaleColumns(op_->getMassHCurlLumped());

    T_->SetBlock(0, 0, &id_u_);
    T_->SetBlock(0, 1, &op_->getD0());
    T_->SetBlock(1, 0, grad_adj_.get());

    // Matrix-free insertion of the -tau * ID block
    if (op_->getTau() != 0.0)
    {
        const int nv = op_->getH1Space().GetNDofs();
        neg_tau_id_p_ = std::make_unique<ScaledIdentityOperator>(nv, -op_->getTau());
        T_->SetBlock(1, 1, neg_tau_id_p_.get());
    }
}

void StokesNitscheDGS::initTransformedSystem()
{
    Lp_ = std::unique_ptr<mfem::SparseMatrix>(
        mfem::Mult(*grad_adj_, op_->getD0()));

    // Consistent-Nitsche outflow: the distributive transformation T has a
    // -tau*I pressure block, so the transformed pressure operator is
    //   (K T)_{11} = grad_adj*d0 + tau * C,   C = M_h1^{-1}(gamma <p,q>)_Gout.
    // The tau factor (= 1/(dt*nu), O(1e4)) is essential -- without it the
    // penalty is invisible to the smoother and the preconditioner fails for the
    // (well-posed) large-gamma regime on multi-level MG.
    if (op_->hasOutflow() && op_->getOutflowPenaltyMass())
    {
        auto pen = std::make_unique<mfem::SparseMatrix>(
            *op_->getOutflowPenaltyMass());
        mfem::Vector inv_mass_h1(op_->getMassH1Lumped());
        inv_mass_h1.Reciprocal();
        pen->ScaleRows(inv_mass_h1);  // M_h1^{-1} (gamma <p,q>)
        const double tau = op_->getTau();
        Lp_ = std::unique_ptr<mfem::SparseMatrix>(
            mfem::Add(1.0, *Lp_, (tau != 0.0 ? tau : 1.0), *pen));
    }

    mfem::Vector inv_mass_hcurl_lumped = op_->getMassHCurlLumped();
    inv_mass_hcurl_lumped.Reciprocal();

    bd_ = std::unique_ptr<mfem::SparseMatrix>(
        mfem::Mult(op_->getNitsche().SpMat(), op_->getD0()));
    bd_->ScaleRows(inv_mass_hcurl_lumped);

    std::unique_ptr<mfem::SparseMatrix> curlcurl_nitsche;
    {
        std::unique_ptr<mfem::SparseMatrix> product;
        {
            auto tmp = std::make_unique<mfem::SparseMatrix>(op_->getD1());
            tmp->ScaleRows(op_->getMassHDivOrL2Lumped());

            auto d1T = std::unique_ptr<mfem::SparseMatrix>(
                mfem::Transpose(op_->getD1()));
            product =
                std::unique_ptr<mfem::SparseMatrix>(mfem::Mult(*d1T, *tmp));
        }

        curlcurl_nitsche = std::unique_ptr<mfem::SparseMatrix>(
            mfem::Add(*product, op_->getNitsche().SpMat()));
        curlcurl_nitsche->ScaleRows(inv_mass_hcurl_lumped);
    }

    auto graddiv = std::unique_ptr<mfem::SparseMatrix>(
        mfem::Mult(op_->getD0(), *grad_adj_));

    Lu_ = std::unique_ptr<mfem::SparseMatrix>(
        mfem::Add(*graddiv, *curlcurl_nitsche));

    // Add +tau * ID to the velocity operator Lu_ (Explicitly assembled for algebraic smoothers)
    if (op_->getTau() != 0.0)
    {
        mfem::SparseMatrix tau_id_u(Lu_->Height());
        for (int i = 0; i < Lu_->Height(); ++i)
        {
            tau_id_u.Add(i, i, op_->getTau());
        }
        tau_id_u.Finalize();

        Lu_ = std::unique_ptr<mfem::SparseMatrix>(
            mfem::Add(*Lu_, tau_id_u));
    }
}

void StokesNitscheDGS::initSmoothers()
{
    if (st_ == SmootherType::Custom)
        return;

    if (st_ == SmootherType::GaussSeidelForw ||
        st_ == SmootherType::GaussSeidelSym)
    {
        int gs_type = (st_ == SmootherType::GaussSeidelSym) ? 0 : 1;
        smoother_u_ = std::make_unique<mfem::GSSmoother>(*Lu_, gs_type);
        smoother_p_ = std::make_unique<mfem::GSSmoother>(*Lp_, gs_type);
    }
    else if (st_ == SmootherType::Jacobi)
    {
        smoother_u_ = std::make_unique<mfem::DSmoother>(*Lu_);
        smoother_p_ = std::make_unique<mfem::DSmoother>(*Lp_);
    }
    // Inefficient, but for testing, we can try Chebyshev
    else if (st_ == SmootherType::Chebyshev)
    {
        const int poly_order = 3;

        // Chebyshev smoother requires an array of essential true DOFs.
        // We pass an empty array since Nitsche's method handles boundaries
        // weakly.
        mfem::Array<int> ess_tdofs;

        // Extract diagonal for Jacobi preconditioning used in Chebyshev
        mfem::Vector diag_u;
        Lu_->GetDiag(diag_u);

        // Passing 'power_iterations' (int) makes MFEM internally instantiate
        // mfem::PowerMethod and estimate the largest eigenvalue of D^{-1} * Lu
        smoother_u_ = std::make_unique<mfem::OperatorChebyshevSmoother>(
            *Lu_, diag_u, ess_tdofs, poly_order);

        mfem::Vector diag_p;
        Lp_->GetDiag(diag_p);

        smoother_p_ = std::make_unique<mfem::OperatorChebyshevSmoother>(
            *Lp_, diag_p, ess_tdofs, poly_order);
    }
    else
    {
        MFEM_ABORT("StokesNitscheDGS: unknown smoother type");
    }

    buildBlockPreconditioner();
}

void StokesNitscheDGS::buildBlockPreconditioner()
{
    block_prec_ = std::make_unique<mfem::BlockLowerTriangularPreconditioner>(
        op_->getOffsets());

    block_prec_->SetDiagonalBlock(0, smoother_u_.get());
    block_prec_->SetDiagonalBlock(1, smoother_p_.get());
    block_prec_->SetBlock(1, 0, grad_adj_.get());
}

void StokesNitscheDGS::SetSmootherU(std::unique_ptr<mfem::Solver> smoother)
{
    smoother_u_ = std::move(smoother);
    if (smoother_p_)
        buildBlockPreconditioner();
}

void StokesNitscheDGS::SetSmootherP(std::unique_ptr<mfem::Solver> smoother)
{
    smoother_p_ = std::move(smoother);
    if (smoother_u_)
        buildBlockPreconditioner();
}

void StokesNitscheDGS::computeResidual(
    const mfem::Vector& x,
    const mfem::Vector& y) const
{
    if (!iterative_mode)
    {
        MFEM_ABORT("StokesNitscheDGS::computeResidual in non-iterative mode!");
    }
    else
    {
        op_->MultDEC(y, residual_);
        residual_ -= x;
    }
}

void StokesNitscheDGS::computeCorrection() const
{
    corr_ = 0.0;
    block_prec_->Mult(residual_, corr_);

    // const unsigned ne = op_->getHCurlSpace().GetNDofs(),
    //                nv = op_->getH1Space().GetNDofs();
    //
    // mfem::Vector r_u(residual_, 0, ne);
    // mfem::Vector r_p(residual_, ne, nv);
    //
    // mfem::Vector corr_u(corr_, 0, ne);
    // mfem::Vector corr_p(corr_, ne, nv);
    //
    // mfem::Vector tmp(ne);
    //
    // bd_->Mult(corr_p, tmp);
    // smoother_u_->AddMult(tmp, corr_u, -1.0);
}

void StokesNitscheDGS::distributeCorrection(mfem::Vector& y) const
{
    if (!iterative_mode)
        y = 0.0;
    T_->AddMult(corr_, y, -1.0);
}

StokesNitscheDGS::StokesNitscheDGS(
    std::shared_ptr<StokesNitscheOperator> op,
    const SmootherType                     type)
    : mfem::Solver(op->NumRows(), true),
      op_(op),
      st_(type),
      id_u_(op->getHCurlSpace().GetNDofs()),
      residual_(op->NumRows()),
      corr_(op->NumRows())
{
    if (op->getMassLumping() == MassLumping::None)
        MFEM_ABORT("StokesNitscheDGS: MassLumping is None");

    initTransformation();
    initTransformedSystem();
    initSmoothers();
}

double StokesNitscheDGS::computeResidualNorm(
    const mfem::Vector& x,
    const mfem::Vector& y) const
{
    computeResidual(x, y);
    return residual_.Norml2();
}

void StokesNitscheDGS::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    computeResidual(x, y);
    computeCorrection();
    distributeCorrection(y);
}

void StokesNitscheDGS::SetOperator(const mfem::Operator&)
{
    MFEM_ABORT("StokesNitscheDGS::SetOperator undefined!");
}

}  // namespace StokesNitsche
