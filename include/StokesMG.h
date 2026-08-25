#ifndef STOKES_MG_HPP
#define STOKES_MG_HPP

#include <memory>
#include <vector>

#include "StokesDGS.h"
#include "StokesOperator.h"
#include "mfem.hpp"

double computeCReg(mfem::Mesh& mesh);
double computeCWBound(
    mfem::Mesh&    mesh,
    const unsigned order  = 1,
    const double   factor = 1);

namespace StokesNitsche
{

enum class RefinementType
{
    Geometric,
    PRef
};
enum class MGCycleType
{
    VCycle,
    WCycle,
    VariableVCycle
};

class StokesMG : public mfem::Solver
{
public:
    /** @brief Geometric Multigrid solver for Stokes-Nitsche systems.
     * Defaults to operator mode(s): DEC.
     *
     * Can NOT solve the true FEM system by itself,
     * use it as a precontioner with e.g. GMRES.
     *
     * If use it to solve the Galerkin (FEM) system, remember to do
     * setOperatorMode(StokesNitsche::OperatorMode::Galerkin)
     * before passing it as a precontioner. */
    StokesMG(
        std::shared_ptr<mfem::Mesh> coarse_mesh,
        const double                tau     = 0.0,
        const double                theta   = 1,
        const double                penalty = -1,
        const double                factor  = 1,
        const MassLumping           ml      = MassLumping::RowSum,
        const SmootherType          st      = SmootherType::GaussSeidelForw,
        // Consistent-Nitsche outflow marker + pressure penalty, forwarded to
        // every level's StokesNitscheOperator (empty => zero-mean as before).
        const mfem::Array<int>*     outflow_marker  = nullptr,
        const double                outflow_penalty = 0.0);

    ~StokesMG() override = default;

    StokesMG(const StokesMG&)            = delete;
    StokesMG& operator=(const StokesMG&) = delete;

    /** @brief Refines and adds a new MG level. order_ref == 0: do
     * h-refinement*/
    void addRefinement(const unsigned order_ref = 0, double penalty = -1);

    void removeRefinement();

    void setCoarseSolver(std::shared_ptr<const mfem::Solver> solver)
    {
        coarse_solver_ = std::move(solver);
    }

    void setSmoothIterations(const int pre, const int post = -1)
    {
        pre_smooth_  = pre;
        post_smooth_ = (post < 0 ? pre : post);
    }

    void setCycleType(const MGCycleType type) { cycle_type_ = type; }
    void setIterativeMode(const bool mode) { iterative_mode = mode; }

    /** @brief Set the mode of the input system.
     * If Galerkin: Input 'b' is scaled by M_lumped^{-1} before the MG cycle.
     * If DEC: Input 'b' is used as-is. */
    void setOperatorMode(const OperatorMode mode) { mode_ = mode; }

    double getTau() const { return tau_; }
    void setTau(const double tau);

    void Mult(const mfem::Vector& b, mfem::Vector& x) const override;
    void SetOperator(const mfem::Operator& op) override;

    const StokesNitscheOperator& getOperator(const int level) const
    {
        MFEM_VERIFY(
            level >= 0 && level < levels_.size(),
            "StokesMG::getOperator: Level index out of bounds.");
        return *levels_[level]->op;
    }

    const StokesNitscheDGS& getSmoother(const int level) const
    {
        MFEM_VERIFY(
            level >= 0 && level < levels_.size(),
            "StokesMG::getSmoother: Level index out of bounds.");
        return *levels_[level]->smoother;
    }

    const StokesNitscheOperator& getFinestOperator() const
    {
        MFEM_VERIFY(
            !levels_.empty(), "StokesMG::getFinestOperator: No levels exist.");
        return *levels_.back()->op;
    }

    const StokesNitscheDGS& getFinestSmoother() const
    {
        MFEM_VERIFY(
            !levels_.empty(), "StokesMG::getFinestSmoother: No levels exist.");
        return *levels_.back()->smoother;
    }

    MGCycleType  getCycleType() const { return cycle_type_; }
    MassLumping  getMassLumping() const { return ml_; }
    SmootherType getSmootherType() const { return st_; }
    unsigned getNumLevels() const { return static_cast<int>(levels_.size()); }
    OperatorMode getOperatorMode() const { return mode_; }

private:
    struct Level
    {
        // NOTE: We cannot put const here, because mfem
        //       requires some non-constness somewhere for some reason
        const std::shared_ptr<StokesNitscheOperator> op;
        const std::shared_ptr<StokesNitscheDGS>      smoother;
        const std::unique_ptr<const mfem::Operator>  T;

        mutable mfem::Vector x, b, res;

        Level(
            std::shared_ptr<StokesNitscheOperator> o,
            std::shared_ptr<StokesNitscheDGS>      s,
            std::unique_ptr<const mfem::Operator>  t)
            : op(std::move(o)),
              smoother(std::move(s)),
              T(std::move(t)),
              x(op->NumRows()),
              b(op->NumRows()),
              res(op->NumRows())
        {
        }
    };

    std::vector<std::unique_ptr<Level>> levels_;
    std::shared_ptr<const mfem::Solver> coarse_solver_;

#ifdef MFEM_USE_SUITESPARSE
    std::unique_ptr<mfem::SparseMatrix>  coarse_mat_;
    std::unique_ptr<mfem::UMFPackSolver> umf_solver_;
    mutable mfem::Vector                 coarse_b_ext_;
    mutable mfem::Vector                 coarse_x_ext_;
#endif

    double             tau_;
    const double       theta_, penalty_, factor_, penalty_bound_coarse_;
    const MassLumping  ml_;
    const SmootherType st_;
    unsigned           order_ = 1;

    // Outflow marker forwarded to each level's operator (empty => unused).
    mfem::Array<int>   outflow_marker_;
    const double       outflow_penalty_ = 0.0;
    const mfem::Array<int>* outflowPtr() const
    {
        return outflow_marker_.Size() > 0 ? &outflow_marker_ : nullptr;
    }

    int                  pre_smooth_  = 1;
    int                  post_smooth_ = 1;
    MGCycleType          cycle_type_  = MGCycleType::VCycle;
    mutable OperatorMode mode_        = OperatorMode::Galerkin;

    mutable mfem::Vector b_scaled_;

    void cycle(const int level_idx, const mfem::Vector& b, mfem::Vector& x)
        const;

    void buildTransfers(
        const StokesNitscheOperator&           coarse,
        const StokesNitscheOperator&           fine,
        std::unique_ptr<const mfem::Operator>& T,
        const unsigned                         order_ref) const;
};

}  // namespace StokesNitsche

#endif
