#include "StokesMG.h"

double computeCReg(mfem::Mesh& mesh)
{
    const int ne       = mesh.GetNE();
    double    max_creg = 0.0;

    // Helper lambda to precisely integrate the area of a face
    auto ComputeFaceArea = [](mfem::ElementTransformation* Tr) -> double
    {
        if (!Tr)
            return 0.0;

        int                          order = std::max(1, 2 * Tr->OrderW());
        const mfem::IntegrationRule& ir =
            mfem::IntRules.Get(Tr->GetGeometryType(), order);

        double area = 0.0;
        for (int p = 0; p < ir.GetNPoints(); p++)
        {
            const mfem::IntegrationPoint& ip = ir.IntPoint(p);
            Tr->SetIntPoint(&ip);
            area += Tr->Weight() * ip.weight;
        }
        return area;
    };

    // The "faces" of an element are its (dim-1)-dimensional boundary
    // entities: actual faces in 3D, edges in 2D.  GetElementFaces requires
    // the 3D element-to-face table and aborts on a 2D mesh, so dispatch on
    // the topological dimension.
    const int dim = mesh.Dimension();
    mfem::Array<int> faces, ori;

    for (int i = 0; i < ne; ++i)
    {
        double volume       = mesh.GetElementVolume(i);
        double h_K          = mesh.GetElementSize(i);
        double surface_area = 0.0;

        // Fetch the (dim-1)-faces of the current element
        if (dim == 3)
        {
            mesh.GetElementFaces(i, faces, ori);
        }
        else
        {
            mesh.GetElementEdges(i, faces, ori);
        }
        for (int f = 0; f < faces.Size(); ++f)
        {
            mfem::ElementTransformation* Tf =
                (dim == 3) ? mesh.GetFaceTransformation(faces[f])
                           : mesh.GetEdgeTransformation(faces[f]);
            surface_area += ComputeFaceArea(Tf);
        }

        double current_creg = (surface_area / volume) * h_K;
        max_creg            = std::max(max_creg, current_creg);
    }

    return max_creg;
}

double
computeCWBound(mfem::Mesh& mesh, const unsigned order, const double factor)
{
    const double res =
        factor * 4. * order * (order + 2) * computeCReg(mesh) / 3.;
    return res;
}

namespace StokesNitsche
{

StokesMG::StokesMG(
    std::shared_ptr<mfem::Mesh> coarse_mesh,
    const double                tau,
    const double                theta,
    const double                penalty,
    const double                factor,
    const MassLumping           ml,
    const SmootherType          st,
    const mfem::Array<int>*     outflow_marker,
    const double                outflow_penalty)
    : mfem::Solver(0, 0),
      tau_(tau),
      theta_(theta),
      factor_(factor),
      penalty_bound_coarse_(computeCWBound(*coarse_mesh)),
      penalty_(penalty < 0 ? computeCWBound(*coarse_mesh) : penalty),
      ml_(ml),
      st_(st),
      outflow_penalty_(outflow_penalty)
{
    iterative_mode = false;

    if (outflow_marker && outflow_marker->Size() > 0)
        outflow_marker_ = *outflow_marker;

    auto op = std::make_shared<StokesNitscheOperator>(
        coarse_mesh, tau_, order_, theta_, penalty_, factor_, ml_,
        outflowPtr(), outflow_penalty_);
    auto smoother = std::make_shared<StokesNitscheDGS>(op, st_);

    height = op->NumRows();
    width  = op->NumCols();

    levels_.push_back(
        std::make_unique<Level>(std::move(op), std::move(smoother), nullptr));

#ifdef MFEM_USE_SUITESPARSE
    coarse_mat_ = levels_[0]->op->getFullDECSystem();

    if (coarse_mat_->NumRows() > height)
    {
        coarse_b_ext_.SetSize(coarse_mat_->NumRows());
        coarse_x_ext_.SetSize(coarse_mat_->NumRows());
        coarse_b_ext_ = 0.0;
        coarse_x_ext_ = 0.0;
    }

    umf_solver_ = std::make_unique<mfem::UMFPackSolver>();
    umf_solver_->SetOperator(*coarse_mat_);
    MFEM_ASSERT(
        umf_solver_->Info[UMFPACK_STATUS] == UMFPACK_OK,
        "StokesMG::StokesMG: coarse factorization failed.");
#else
    mfem::out << "Warning: StokesMG: MFEM not built with SuiteSparse. "
              << "Defaulting to smoothing for coarse solve." << std::endl;
#endif
}

void StokesMG::setTau(const double tau)
{
    tau_ = tau;
    
    // Propagate the tau change to all levels
    // The smoother's setTau calls the operator's setTau and rebuilds itself
    for (auto& lvl : levels_)
    {
        lvl->smoother->setTau(tau_);
    }

#ifdef MFEM_USE_SUITESPARSE
    // Rebuild the explicit coarse solver system with the updated tau
    coarse_mat_ = levels_[0]->op->getFullDECSystem();
    if (umf_solver_)
    {
        umf_solver_->SetOperator(*coarse_mat_);
    }
#endif
}

void StokesMG::addRefinement(const unsigned order_ref, double penalty)
{
    Level& coarse_lvl = *levels_.back();

    std::shared_ptr<mfem::Mesh>            fine_mesh;
    std::shared_ptr<StokesNitscheOperator> fine_op;

    if (!order_ref)
    {
        MFEM_VERIFY(
            order_ == 1,
            "StokesMG::addRefinement: Doing geometric refinement after "
            "p-refinement!");
        // Copy mesh and refine
        fine_mesh = std::make_shared<mfem::Mesh>(coarse_lvl.op->getMesh());
        fine_mesh->UniformRefinement();
        // Add lowest order
        fine_op = std::make_shared<StokesNitscheOperator>(
            fine_mesh, tau_, order_, theta_, penalty < 0 ? penalty_ : penalty,
            factor_, ml_, outflowPtr(), outflow_penalty_);
    }
    else  // reftype == RefinementType::PRef
    {
        order_ += order_ref;
        // fine_mesh == coarse_mesh, only p is different
        fine_mesh = coarse_lvl.op->getMeshPtr();

        if (penalty < 0)
            penalty = (penalty_ / penalty_bound_coarse_) *
                      computeCWBound(*fine_mesh, order_);

        fine_op = std::make_shared<StokesNitscheOperator>(
            fine_mesh, tau_, order_, theta_, penalty, factor_, ml_,
            outflowPtr(), outflow_penalty_);
    }
    auto fine_smoother = std::make_shared<StokesNitscheDGS>(fine_op, st_);

    std::unique_ptr<const mfem::Operator> T;
    buildTransfers(*coarse_lvl.op, *fine_op, T, order_ref);

    height = fine_op->NumRows();
    width  = fine_op->NumCols();

    levels_.push_back(
        std::make_unique<Level>(
            std::move(fine_op), std::move(fine_smoother), std::move(T)));
}

void StokesMG::removeRefinement()
{
    if (levels_.size() > 1)
        levels_.pop_back();
    order_ = levels_.back()->op->getOrder();
}

// Helper Struct for transfer operators
struct TransferOperator : public mfem::Operator
{
    const std::unique_ptr<const mfem::Operator> P;
    const mfem::Vector&                         Mc;
    const mfem::Vector&                         Mf;
    mutable mfem::Vector                        tmp;

    // TAKES OWNERSHIP OF P
    TransferOperator(
        mfem::Operator* prol,
        const mfem::Vector& Mcoarse,
        const mfem::Vector& Mfine)
        : mfem::Operator(Mfine.Size(), Mcoarse.Size()),
          P(prol),
          Mc(Mcoarse),
          Mf(Mfine),
          tmp(Mfine.Size())
    {
    }

    void Mult(const mfem::Vector& x, mfem::Vector& y) const override
    {
        P->Mult(x, y);
    }

    void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
    {
        tmp = x;
        tmp *= Mf;
        P->MultTranspose(tmp, y);
        y /= Mc;
    }
};

void StokesMG::buildTransfers(
    const StokesNitscheOperator&           coarse,
    const StokesNitscheOperator&           fine,
    std::unique_ptr<const mfem::Operator>& T,
    const unsigned                         order_ref) const
{
    auto T_block = std::make_unique<mfem::BlockOperator>(
        fine.getOffsets(), coarse.getOffsets());

    auto create_transfer = [&](const mfem::FiniteElementSpace& c_fes,
                               const mfem::FiniteElementSpace& f_fes,
                               const mfem::Vector&             c_mass,
                               const mfem::Vector& f_mass) -> mfem::Operator*
    {
        mfem::Operator* base_P = nullptr;

        if (!order_ref)
        {
            mfem::OperatorPtr P_ptr;
            f_fes.GetTransferOperator(c_fes, P_ptr);
            P_ptr.SetOperatorOwner(false);  // We take ownership
            base_P = P_ptr.Ptr();
        }
        else  // p-refinement
        {
            base_P = new mfem::PRefinementTransferOperator(c_fes, f_fes);
        }

        return new TransferOperator(base_P, c_mass, f_mass);
    };

    // --- H(curl) Block ---
    T_block->SetBlock(
        0, 0,
        create_transfer(
            coarse.getHCurlSpace(), fine.getHCurlSpace(),
            coarse.getMassHCurlLumped(), fine.getMassHCurlLumped()));

    // --- H1 Block ---
    T_block->SetBlock(
        1, 1,
        create_transfer(
            coarse.getH1Space(), fine.getH1Space(), coarse.getMassH1Lumped(),
            fine.getMassH1Lumped()));

    T_block->owns_blocks = 1;
    T                    = std::move(T_block);
}

void StokesMG::cycle(
    const int           level_idx,
    const mfem::Vector& b,
    mfem::Vector&       x) const
{
    const Level& lvl = *levels_[level_idx];

    if (level_idx == 0)
    {
        if (coarse_solver_)
            coarse_solver_->Mult(b, x);
#ifdef MFEM_USE_SUITESPARSE
        else if (umf_solver_)
        {
            if (coarse_b_ext_.Size() > 0)
            {
                coarse_b_ext_ = 0.0;
                coarse_b_ext_.SetVector(b, 0);
                umf_solver_->Mult(coarse_b_ext_, coarse_x_ext_);
                mfem::Vector x_view(coarse_x_ext_.GetData(), x.Size());
                x = x_view;
            }
            else
            {
                umf_solver_->Mult(b, x);
            }
        }
#endif
        else  // fallback to a few smoothening interations as the coarse solve
            for (int i = 0; i < pre_smooth_ + post_smooth_; ++i)
                lvl.smoother->Mult(b, x);
        return;
    }

    Level& coarse_lvl = *levels_[level_idx - 1];

    unsigned fac = 1;
    if (cycle_type_ == MGCycleType::VariableVCycle)
        fac = getFinestOperator().NumRows() / lvl.op->NumRows();

    for (int i = 0; i < fac * pre_smooth_; ++i)
        lvl.smoother->Mult(b, x);

    lvl.op->MultDEC(x, lvl.res);
    lvl.res -= b;

    lvl.T->MultTranspose(lvl.res, coarse_lvl.b);

    coarse_lvl.x = 0.0;
    cycle(level_idx - 1, coarse_lvl.b, coarse_lvl.x);

    if (cycle_type_ == MGCycleType::WCycle)
        cycle(level_idx - 1, coarse_lvl.b, coarse_lvl.x);

    lvl.T->Mult(coarse_lvl.x, lvl.res);
    x -= lvl.res;

    for (int i = 0; i < fac * post_smooth_; ++i)
        lvl.smoother->Mult(b, x);
}

void StokesMG::Mult(const mfem::Vector& b, mfem::Vector& x) const
{
    if (levels_.empty())
        MFEM_ABORT("StokesMG::Mult: No levels defined.");

    if (!iterative_mode)
        x = 0.0;

    if (mode_ == OperatorMode::Galerkin)
    {
        b_scaled_.SetSize(b.Size());

        const auto&         finest_op = getFinestOperator();
        const mfem::Vector& M_u       = finest_op.getMassHCurlLumped();
        const mfem::Vector& M_p       = finest_op.getMassH1Lumped();

        const int nv = finest_op.getH1Space().GetNDofs(),
                  ne = finest_op.getHCurlSpace().GetNDofs();

        const mfem::Vector b_u(b.GetData(), ne);
        const mfem::Vector b_p(b.GetData() + ne, nv);

        mfem::Vector b_scaled_u(b_scaled_.GetData(), ne);
        mfem::Vector b_scaled_p(b_scaled_.GetData() + ne, nv);

        b_scaled_u = b_u;
        b_scaled_u /= M_u;

        b_scaled_p = b_p;
        b_scaled_p /= M_p;

        cycle(static_cast<int>(levels_.size()) - 1, b_scaled_, x);
    }
    else
    {
        cycle(static_cast<int>(levels_.size()) - 1, b, x);
    }
}

void StokesMG::SetOperator(const mfem::Operator&)
{
    MFEM_ABORT(
        "StokesMG::SetOperator: Use addRefinedLevel to manage the hierarchy.");
}

}  // namespace StokesNitsche
