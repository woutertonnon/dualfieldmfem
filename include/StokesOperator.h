#ifndef STOKES_NITSCHE_OPERATOR_H
#define STOKES_NITSCHE_OPERATOR_H

#include <memory>

#include "mfem.hpp"

namespace StokesNitsche
{

enum class MassLumping
{
    None,
    Diagonal,
    Barycentric,
    RowSum
};
enum class OperatorMode
{
    Galerkin,
    DEC
};

// This class implements a Stokes operator using Nitsche's method for weak
// boundary imposition, supporting both standard Galerkin and Discrete Exterior
// Calculus (DEC) modes. It manages the underlying FE spaces, discrete exterior
// derivatives, and mass operators (consistent or lumped).

class StokesNitscheOperator : public mfem::Operator
{
public:
    StokesNitscheOperator(
        std::shared_ptr<mfem::Mesh> mesh_ptr,
        const double                tau,
        const unsigned              order,
        const double                theta,
        const double                penalty,
        const double                factor,
        const MassLumping           ml = MassLumping::Diagonal);

    StokesNitscheOperator(const StokesNitscheOperator&)            = delete;
    StokesNitscheOperator& operator=(const StokesNitscheOperator&) = delete;

    MassLumping getMassLumping() const { return ml_; }
    OperatorMode getOperatorMode() const { return opmode_; }

    void setOperatorMode(const OperatorMode mode) const { opmode_ = mode; }

    double getTau() const { return tau_; }
    void setTau(const double tau) { tau_ = tau; }

    // Nitsche parameters actually used to assemble this operator's boundary
    // form. The penalty is rescaled across p-refinement (see StokesMG), so a
    // matching right-hand side (StokesRHS) MUST use these values rather than a
    // hardcoded constant, otherwise the Nitsche imposition is inconsistent and
    // the discrete solution does not reproduce the exact boundary trace.
    double getPenalty() const { return penalty_; }
    double getTheta() const { return theta_; }
    
    const mfem::SparseMatrix& getD0() const { return d0_; }
    const mfem::SparseMatrix& getD1() const { return d1_; }
    mfem::SparseMatrix&       getD0() { return d0_; }
    mfem::SparseMatrix&       getD1() { return d1_; }

    const mfem::FiniteElementSpace& getH1Space() const { return *h1_space_; }
    const mfem::FiniteElementSpace& getHCurlSpace() const
    {
        return *hcurl_space_;
    }
    const mfem::FiniteElementSpace& getHDivOrL2Space() const
    {
        return *hdiv_or_l2_space_;
    }

    mfem::FiniteElementSpace& getH1Space() { return *h1_space_; }
    mfem::FiniteElementSpace& getHCurlSpace() { return *hcurl_space_; }
    mfem::FiniteElementSpace& getHDivOrL2Space() { return *hdiv_or_l2_space_; }

    const mfem::BilinearForm& getMassH1() const { return *mass_h1_; }
    const mfem::BilinearForm& getMassHCurl() const { return *mass_hcurl_; }
    const mfem::BilinearForm& getMassHDivOrL2() const
    {
        return *mass_hdiv_or_l2_;
    }
    const mfem::BilinearForm& getNitsche() const { return *nitsche_; }

    const mfem::Vector& getMassH1Lumped() const { return mass_h1_lumped_; }
    const mfem::Vector& getMassHCurlLumped() const
    {
        return mass_hcurl_lumped_;
    }
    const mfem::Vector& getMassHDivOrL2Lumped() const
    {
        return mass_hdiv_or_l2_lumped_;
    }

    const mfem::Mesh& getMesh() const { return *mesh_; }
    mfem::Mesh&       getMesh() { return *mesh_; }

    std::shared_ptr<mfem::Mesh>             getMeshPtr() { return mesh_; }
    const std::shared_ptr<const mfem::Mesh> getMeshPtr() const { return mesh_; }

    const mfem::Array<int>& getOffsets() const { return offsets_; }
    unsigned                getOrder() const { return order_; }

    std::unique_ptr<mfem::SparseMatrix> getFullGalerkinSystem() const;
    std::unique_ptr<mfem::SparseMatrix> getFullDECSystem() const;
    std::unique_ptr<mfem::SparseMatrix> getFullSystem() const;

    /** @brief Orthogonalizes vector x against constant modes (if applicable).
     */
    void eliminateConstants(mfem::Vector& x) const;

    /** @brief Application of the operator: y = A(x). Mode depends on
     * setOperatorMode(). */
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    void MultDEC(const mfem::Vector& x, mfem::Vector& y) const;
    void MultGalerkin(const mfem::Vector& x, mfem::Vector& y) const;

private:
    void initFESpaces();
    void initIncidence();
    void initMass();
    void initLumpedMass();
    void
    initNitsche(const double theta, const double penalty, const double factor);

    std::shared_ptr<mfem::Mesh> mesh_;
    const unsigned              order_;
    double                      tau_ = 0.0;
    double                      theta_ = 1.0;
    double                      penalty_ = 0.0;
    const MassLumping           ml_;
    mfem::Array<int>            offsets_;
    mutable OperatorMode        opmode_ = OperatorMode::Galerkin;

    std::unique_ptr<mfem::FiniteElementCollection> h1_fec_;
    std::unique_ptr<mfem::FiniteElementCollection> hcurl_fec_;
    std::unique_ptr<mfem::FiniteElementCollection> hdiv_or_l2_fec_;

    std::unique_ptr<mfem::FiniteElementSpace> h1_space_;
    std::unique_ptr<mfem::FiniteElementSpace> hcurl_space_;
    std::unique_ptr<mfem::FiniteElementSpace> hdiv_or_l2_space_;

    mfem::SparseMatrix d0_, d1_;

    std::unique_ptr<mfem::BilinearForm> mass_h1_;
    std::unique_ptr<mfem::BilinearForm> mass_hcurl_;
    std::unique_ptr<mfem::BilinearForm> mass_hdiv_or_l2_;
    std::unique_ptr<mfem::BilinearForm> nitsche_;

    mfem::Vector mass_h1_lumped_;
    mfem::Vector mass_hcurl_lumped_;
    mfem::Vector mass_hdiv_or_l2_lumped_;
};

}  // namespace StokesNitsche

#endif
