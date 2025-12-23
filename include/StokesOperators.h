#ifndef STOKESOPERATORS_H
#define STOKESOPERATORS_H

#include <memory>
#include <utility>
#include <vector>

#include "mfem.hpp"
#include "BoundaryOperators.h"

// Utility that computes block offsets for a list of finite element spaces.
struct OffsetsHolder
{
    mfem::Array<int> offsets_;

    explicit OffsetsHolder(const std::vector<mfem::FiniteElementSpace *> &fes_array)
    {
        offsets_.SetSize(static_cast<int>(fes_array.size()) + 1);
        offsets_[0] = 0;

        for (int i = 0; i < (int)fes_array.size(); ++i)
        {
            MFEM_VERIFY(fes_array[i] != nullptr, "Null FiniteElementSpace* in fes_array");
            offsets_[i + 1] = fes_array[i]->GetVSize(); // or GetTrueVSize() if you use true-dof vectors
        }

        offsets_.PartialSum();
    }
};

class StokesSolution
    : private OffsetsHolder,
      public mfem::Vector
{
private:
    mfem::GridFunction u_;
    mfem::GridFunction p_;

public:
    StokesSolution(mfem::FiniteElementSpace &ND,
                        mfem::FiniteElementSpace &CG)
              : OffsetsHolder({&ND, &CG}),
                mfem::Vector(offsets_.Last()),
                u_(&ND,*this,0), p_(&CG,*this,offsets_[1])
                {
                    u_ = 0.;
                    p_ = 0.;
                }

    mfem::GridFunction& get_u(){return u_;};
    mfem::GridFunction& get_p(){return p_;};

};

class StokesRHS
    : private OffsetsHolder,
      public mfem::BlockVector
{
private:
    mfem::LinearForm f_lf_, g_lf_;
    mfem::VectorFunctionCoefficient f_coef_;
    mfem::VectorFunctionCoefficient tr_u_coef_;
    double theta_, Cw_;

public:
    StokesRHS(mfem::FiniteElementSpace &ND,
              mfem::FiniteElementSpace &CG,
              std::function<void(const mfem::Vector&, double, mfem::Vector&)> f,
              std::function<void(const mfem::Vector&, double, mfem::Vector&)> tr_u,
              double theta = 1.,
              double Cw = 100.)
              : OffsetsHolder({&ND, &CG}),
                mfem::BlockVector(offsets_),
                f_lf_(&ND),
                g_lf_(&CG),
                f_coef_(CG.GetMesh()->Dimension(), std::move(f)),
                tr_u_coef_(CG.GetMesh()->Dimension(), std::move(tr_u)),
                theta_(theta), Cw_(Cw)
                {
                    f_lf_.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coef_));
                    f_lf_.AddBdrFaceIntegrator(new WouterLFIntegrator(theta_, Cw_, tr_u_coef_));
                    f_lf_.Assemble();

                    g_lf_.AddBoundaryIntegrator(new mfem::BoundaryNormalLFIntegrator(tr_u_coef_));
                    g_lf_.Assemble();

                    GetBlock(0).Set(1.,f_lf_);
                    GetBlock(1).Set(1.,g_lf_);
                }

};

class StokesSystem
    : private OffsetsHolder,
      public mfem::BlockMatrix
{
private:
    mfem::BilinearForm blf_A;
    mfem::MixedBilinearForm blf_B;
    mfem::TransposeOperator BT;
    mfem::SparseMatrix* BT_mat;
    double mass_, viscosity_, theta_, Cw_;

public:
    StokesSystem(mfem::FiniteElementSpace &ND,
                 mfem::FiniteElementSpace &CG,
                 double mass, double viscosity, double theta, double Cw)
        : OffsetsHolder({&ND, &CG}) // 1) offsets constructed first
          ,
          mfem::BlockMatrix(offsets_) // 2) base MakeRef(offsets)
          ,
          blf_A(&ND), blf_B(&CG, &ND), BT(blf_B), mass_(mass), viscosity_(viscosity), theta_(theta), Cw_(Cw)
    {
        // assemble operators
        mfem::ConstantCoefficient mass_coef(mass_), viscosity_coef(viscosity_);

        blf_A.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(mass_coef));
        blf_A.AddDomainIntegrator(new mfem::CurlCurlIntegrator(viscosity_coef));
        blf_A.AddBdrFaceIntegrator(new WouterIntegrator(theta_, Cw_));
        blf_A.Assemble();
        blf_A.Finalize(); // only if you need the explicit matrix (SparseMatrix)

        blf_B.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator());
        blf_B.Assemble();
        blf_B.Finalize(); // only if you need the explicit matrix

        BT_mat = mfem::Transpose(blf_B.SpMat());
        // hook blocks
        SetBlock(0, 0, &blf_A.SpMat());
        SetBlock(0, 1, &blf_B.SpMat());
        SetBlock(1, 0, BT_mat);
    }
};

class SobolevPreconditioner
    : private OffsetsHolder,
      public mfem::BlockDiagonalPreconditioner
{
private:
    std::vector<std::unique_ptr<mfem::BilinearForm>> bil_forms_;
    std::vector<std::unique_ptr<mfem::CGSolver>> solvers_;
    std::vector<std::unique_ptr<mfem::OperatorJacobiSmoother>> smoothers_;

    static void AddMassDiffIntegrators(mfem::BilinearForm &blf)
    {
        using namespace mfem;

        const FiniteElementCollection *fec = blf.FESpace()->FEColl();

        if (dynamic_cast<const H1_FECollection *>(fec))
        {
            blf.AddDomainIntegrator(new MassIntegrator());
            //blf.AddDomainIntegrator(new DiffusionIntegrator());
        }
        else if (dynamic_cast<const ND_FECollection *>(fec))
        {
            blf.AddDomainIntegrator(new VectorFEMassIntegrator());
            //blf.AddDomainIntegrator(new CurlCurlIntegrator());
        }
        else if (dynamic_cast<const RT_FECollection *>(fec))
        {
            blf.AddDomainIntegrator(new VectorFEMassIntegrator());
            blf.AddDomainIntegrator(new DivDivIntegrator());
        }
        else if (dynamic_cast<const DG_FECollection *>(fec))
        {
            blf.AddDomainIntegrator(new MassIntegrator());
        }
        else
        {
            MFEM_ABORT("SobolevPreconditioner: unsupported FECollection type.");
        }
    }

public:
    explicit SobolevPreconditioner(const std::vector<mfem::FiniteElementSpace *> &fes_array)
        : OffsetsHolder(fes_array) // 1) offsets built first
          ,
          mfem::BlockDiagonalPreconditioner(offsets_) // 2) MFEM stores MakeRef(offsets)
          ,
          bil_forms_(fes_array.size()), solvers_(fes_array.size()), smoothers_(fes_array.size())
    {
        for (int i = 0; i < (int)fes_array.size(); ++i)
        {
            auto *fes = fes_array[i];

            bil_forms_[i] = std::make_unique<mfem::BilinearForm>(fes);
            AddMassDiffIntegrators(*bil_forms_[i]);
            bil_forms_[i]->Assemble();
            bil_forms_[i]->Finalize(); // important for Jacobi diagonal

            smoothers_[i] = std::make_unique<mfem::OperatorJacobiSmoother>(*bil_forms_[i],
                                                                           mfem::Array<int>{});

            solvers_[i] = std::make_unique<mfem::CGSolver>();
            solvers_[i]->SetMaxIter(100);
            solvers_[i]->SetRelTol(0.0);
            solvers_[i]->SetAbsTol(1e-4);
            solvers_[i]->SetPrintLevel(0);
            solvers_[i]->SetPreconditioner(*smoothers_[i]);
            solvers_[i]->SetOperator(*bil_forms_[i]);

            SetDiagonalBlock(i, solvers_[i].get());
        }
    }
};

#endif // STOKESOPERATORS_H
