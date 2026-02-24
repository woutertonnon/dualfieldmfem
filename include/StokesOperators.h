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

class SobolevPreconditioner
    : private OffsetsHolder,
      public mfem::BlockDiagonalPreconditioner
{
private:
    std::vector<std::unique_ptr<mfem::BilinearForm>> bil_forms_;
    std::vector<std::unique_ptr<mfem::CGSolver>> solvers_;
    std::vector<std::unique_ptr<mfem::GSSmoother>> smoothers_;
    std::vector<mfem::ConstantCoefficient> mass_weights_, diff_weights_;

    static void AddMassDiffIntegrators(mfem::BilinearForm &blf, mfem::Coefficient &mass_weight_coef, mfem::Coefficient &diff_weight_coef)
    {
        using namespace mfem;
        const FiniteElementCollection *fec = blf.FESpace()->FEColl();
        if (dynamic_cast<const H1_FECollection *>(fec))
        {
            blf.AddDomainIntegrator(new MassIntegrator(mass_weight_coef));
            blf.AddDomainIntegrator(new DiffusionIntegrator(diff_weight_coef));
        }
        else if (dynamic_cast<const ND_FECollection *>(fec))
        {
            blf.AddDomainIntegrator(new VectorFEMassIntegrator(mass_weight_coef));
            blf.AddDomainIntegrator(new CurlCurlIntegrator(diff_weight_coef));
        }
        else if (dynamic_cast<const RT_FECollection *>(fec))
        {
            blf.AddDomainIntegrator(new VectorFEMassIntegrator(mass_weight_coef));
            blf.AddDomainIntegrator(new DivDivIntegrator(diff_weight_coef));
        }
        else if (dynamic_cast<const DG_FECollection *>(fec))
        {
            blf.AddDomainIntegrator(new MassIntegrator(mass_weight_coef));
        }
        else
        {
            MFEM_ABORT("SobolevPreconditioner: unsupported FECollection type.");
        }
    }

public:
    explicit SobolevPreconditioner(const std::vector<mfem::FiniteElementSpace *> &fes_array, const std::vector<double> mass_weights, const std::vector<double> diff_weights)
        : OffsetsHolder(fes_array), mass_weights_(mass_weights.begin(), mass_weights.end()), diff_weights_(diff_weights.begin(), diff_weights.end()) // 1) offsets built first
          ,
          mfem::BlockDiagonalPreconditioner(offsets_),
          bil_forms_(fes_array.size()), solvers_(fes_array.size()), smoothers_(fes_array.size())
    {
        for (int i = 0; i < (int)fes_array.size(); ++i)
        {
            auto *fes = fes_array[i];

            bil_forms_[i] = std::make_unique<mfem::BilinearForm>(fes);
            AddMassDiffIntegrators(*bil_forms_[i], mass_weights_[i], diff_weights_[i]);
            bil_forms_[i]->Assemble();
            bil_forms_[i]->Finalize();

            smoothers_[i] = std::make_unique<mfem::GSSmoother>(bil_forms_[i]->SpMat());

            solvers_[i] = std::make_unique<mfem::CGSolver>();
            solvers_[i]->SetMaxIter(100);
            solvers_[i]->SetRelTol(0.0);
            solvers_[i]->SetAbsTol(1e-4);
            solvers_[i]->SetPrintLevel(0);
            solvers_[i]->SetPreconditioner(*smoothers_[i]);
            solvers_[i]->SetOperator(bil_forms_[i]->SpMat());

            SetDiagonalBlock(i, solvers_[i].get());
        }
    }
};



namespace hcurl{
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
            u_(&ND, *this, 0), p_(&CG, *this, offsets_[1])
        {
            u_ = 0.;
            p_ = 0.;
        }

        mfem::GridFunction &get_u() { return u_; };
        mfem::GridFunction &get_p() { return p_; };
    };


    class StokesRHS
        : private OffsetsHolder,
        public mfem::BlockVector
    {
    private:
        mfem::FiniteElementSpace *ND_, *CG_;
        mfem::VectorFunctionCoefficient f_coef_;
        mfem::VectorFunctionCoefficient tr_u_coef_;
        double theta_, Cw_, viscosity_;

    public:
        StokesRHS(mfem::FiniteElementSpace &ND,
                mfem::FiniteElementSpace &CG,
                std::function<void(const mfem::Vector &, double, mfem::Vector &)> f,
                std::function<void(const mfem::Vector &, double, mfem::Vector &)> tr_u,
                double theta = 1.,
                double Cw = 100.,
                double viscosity = 1.)
            : OffsetsHolder({&ND, &CG}),
            mfem::BlockVector(offsets_),
            ND_(&ND),
            CG_(&CG),
            f_coef_(CG.GetMesh()->Dimension(), std::move(f)),
            tr_u_coef_(CG.GetMesh()->Dimension(), std::move(tr_u)),
            theta_(theta), Cw_(Cw), viscosity_(viscosity)
        {
            mfem::LinearForm f_lf(ND_);
            f_lf.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coef_));
            f_lf.AddBdrFaceIntegrator(new ND_NitscheLFIntegrator(theta_, Cw_, tr_u_coef_, viscosity_));
            f_lf.Assemble();

            mfem::LinearForm g_lf(CG_);
            g_lf.AddBoundaryIntegrator(new mfem::BoundaryNormalLFIntegrator(tr_u_coef_));
            g_lf.Assemble();

            GetBlock(0).Set(1., f_lf);
            GetBlock(1).Set(1., g_lf);
        }

        void Update(mfem::GridFunction& u_prev, double t, double mass)
        {
            mfem::VectorGridFunctionCoefficient u_prev_coef(&u_prev);
            mfem::ScalarVectorProductCoefficient mass_u_prev_coef(mass, u_prev_coef);

            f_coef_.SetTime(t);
            tr_u_coef_.SetTime(t);


            mfem::LinearForm f_lf(ND_);
            f_lf.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(mass_u_prev_coef));
            f_lf.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coef_));
            f_lf.AddBdrFaceIntegrator(new ND_NitscheLFIntegrator(theta_, Cw_, tr_u_coef_, viscosity_));
            f_lf.Assemble();

            mfem::LinearForm g_lf(CG_);
            g_lf.AddBoundaryIntegrator(new mfem::BoundaryNormalLFIntegrator(tr_u_coef_));
            g_lf.Assemble();

            GetBlock(0).Set(1., f_lf);
            GetBlock(1).Set(1., g_lf);
        }
    };

    class StokesSystem
        : private OffsetsHolder,
        public mfem::BlockMatrix
    {
    private:

        mfem::FiniteElementSpace *ND_;
        mfem::MixedBilinearForm blf_B;
        mfem::TransposeOperator BT;
        mfem::SparseMatrix *BT_mat, *A_mat;
        double mass_, viscosity_, theta_, Cw_;

    public:
        StokesSystem(mfem::FiniteElementSpace &ND,
                    mfem::FiniteElementSpace &CG,
                    double mass, double viscosity, double theta, double Cw)
            : OffsetsHolder({&ND, &CG}) // 1) offsets constructed first
            ,
            mfem::BlockMatrix(offsets_) // 2) base MakeRef(offsets)
            ,
            ND_(&ND), blf_B(&CG, &ND), BT(blf_B), mass_(mass), viscosity_(viscosity), theta_(theta), Cw_(Cw)
        {
            // assemble operators
            mfem::ConstantCoefficient mass_coef(mass_), viscosity_coef(viscosity_);

            mfem::BilinearForm blf_A(ND_);
            blf_A.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(mass_coef));
            blf_A.AddDomainIntegrator(new mfem::CurlCurlIntegrator(viscosity_coef));
            blf_A.AddBdrFaceIntegrator(new ND_NitscheIntegrator(theta_, Cw_, viscosity_));
            blf_A.Assemble();
            blf_A.Finalize(); // only if you need the explicit matrix (SparseMatrix)

            blf_B.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator());
            blf_B.Assemble();
            blf_B.Finalize(); // only if you need the explicit matrix

            BT_mat = mfem::Transpose(blf_B.SpMat());
            // hook blocks
            A_mat = blf_A.LoseMat();
            SetBlock(0, 0, A_mat);
            SetBlock(0, 1, &blf_B.SpMat());
            SetBlock(1, 0, BT_mat);
        }

        void Update(mfem::VectorCoefficient &w_coef)
        {

            mfem::ConstantCoefficient mass_coef(mass_), viscosity_coef(viscosity_);

            delete A_mat;
            
            mfem::BilinearForm blf_A(ND_);
            blf_A.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(mass_coef));
            blf_A.AddDomainIntegrator(new mfem::CurlCurlIntegrator(viscosity_coef));
            blf_A.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_coef));
            blf_A.AddBdrFaceIntegrator(new ND_NitscheIntegrator(theta_, Cw_, viscosity_));
            blf_A.Assemble();
            blf_A.Finalize(); 

            A_mat = blf_A.LoseMat();

            SetBlock(0, 0, A_mat);

        }

        ~StokesSystem() {
            delete A_mat;
        }
    };

    class SchurSolver
    : private OffsetsHolder,
      public mfem::Solver
    {
    private:
        mfem::BlockMatrix *op_;
        double mass_, viscosity_, tol_;
        int &iterations_;
        mfem::FiniteElementSpace &ND_, &CG_;
        mfem::UMFPackSolver invA;
        mfem::GSSmoother smoother_;
        mfem::CGSolver cgsolver_;
        mfem::BilinearForm mass_bil_;

    public:
        SchurSolver(mfem::FiniteElementSpace &ND,
                    mfem::FiniteElementSpace &CG,
                    double mass, double viscosity, int& iterations, double tol = 1e-8)
            : mfem::Solver(ND.GetVDim() + CG.GetVDim()), OffsetsHolder({&ND, &CG}), mass_(mass), viscosity_(viscosity), iterations_(iterations), tol_(tol), ND_(ND), CG_(CG), invA(), smoother_(), cgsolver_(), mass_bil_(&ND)
        {
            mass_bil_.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
            mass_bil_.Assemble();
        }

        void SetOperator(const mfem::Operator &op)
        {
            throw std::invalid_argument("SchurPreconditioner::SetOperator(): expected mfem::BlockOperator.");
        }

        void SetOperator(mfem::BlockMatrix &op)
        {
            MFEM_VERIFY(op.RowOffsets().Size() == op.ColOffsets().Size(), "Operator is not square.");
            for (int i = 0; i < op.RowOffsets().Size(); i++)
                MFEM_VERIFY(op.RowOffsets()[i] == op.ColOffsets()[i], "Operator is not square!");
            MFEM_VERIFY(offsets_.Size() == op.RowOffsets().Size(), "Dimensions do not match.");
            for (int i = 0; i < op.RowOffsets().Size(); i++)
                MFEM_VERIFY(op.RowOffsets()[i] == offsets_[i], "Operator size does not match!");

            op_ = &op;
            


            smoother_.SetOperator(mass_bil_.SpMat());
            cgsolver_.SetOperator(mass_bil_);
            cgsolver_.SetAbsTol(1e-15);
            cgsolver_.SetRelTol(tol_);
            
            //invA.SetPreconditioner(cgsolver_);
            invA.SetOperator(op_->GetBlock(0, 0));
            //invA.SetAbsTol(1e-15);
            //invA.SetRelTol(tol_);
            //invA.SetPrintLevel(0);
            //invA.SetMaxIter(10000);
        }

        void Mult(const mfem::Vector &x, mfem::Vector &y) const override
        {
            mfem::Vector x0, x1, y0, y1;

            x0.MakeRef(const_cast<mfem::Vector &>(x), offsets_[0], offsets_[1] - offsets_[0]);
            x1.MakeRef(const_cast<mfem::Vector &>(x), offsets_[1], offsets_[2] - offsets_[1]);

            y0.MakeRef(y, offsets_[0], offsets_[1] - offsets_[0]);
            y1.MakeRef(y, offsets_[1], offsets_[2] - offsets_[1]);


            mfem::GMRESSolver invS;

            SobolevPreconditioner invS_pre({&CG_},{.0001},{1.});

            mfem::Vector invA_x0(x0.Size());
            mfem::Vector BT_invA_x0_min_x1(x1.Size());
            mfem::Vector p(x1.Size());
            mfem::Vector u(x0.Size());
            mfem::Vector B_p(x0.Size());
            // y0 = 1.;
            invA.Mult(x0, invA_x0);
            op_->GetBlock(1, 0).Mult(invA_x0, BT_invA_x0_min_x1);
            BT_invA_x0_min_x1 -= x1;

            mfem::ProductOperator invA_B(&invA, &op_->GetBlock(0, 1), false, false);
            mfem::ProductOperator BT_invA_B(&op_->GetBlock(1, 0), &invA_B, false, false);

            invS.SetOperator(BT_invA_B);
            invS.SetKDim(3000);
            invS.SetPrintLevel(0);
            invS.SetAbsTol(1e-15);
            invS.SetRelTol(tol_);
            invS.SetMaxIter(10000);
            invS.Mult(BT_invA_x0_min_x1, y1);
            iterations_ = invS.GetNumIterations();

            mfem::Vector x0_min_B_y1(x0.Size());
            x0_min_B_y1.Set(1., x0);
            op_->GetBlock(0, 1).AddMult(y1, x0_min_B_y1, -1.);

            invA.Mult(x0_min_B_y1, y0);
        }
    };

    class SchurPreconditioner
        : private OffsetsHolder,
        public mfem::Solver
    {
    private:
        mfem::BlockMatrix *op_;
        double mass_, viscosity_;
        mfem::FiniteElementSpace &ND_, &CG_;

    public:
        SchurPreconditioner(mfem::FiniteElementSpace &ND,
                            mfem::FiniteElementSpace &CG,
                            double mass, double viscosity)
            : mfem::Solver(ND.GetVDim() + CG.GetVDim()), OffsetsHolder({&ND, &CG}), mass_(mass), viscosity_(viscosity), ND_(ND), CG_(CG)
        {
        }

        void SetOperator(const mfem::Operator &op)
        {
            throw std::invalid_argument("SchurPreconditioner::SetOperator(): expected mfem::BlockOperator.");
        }

        void SetOperator(mfem::BlockMatrix &op)
        {
            MFEM_VERIFY(op.RowOffsets().Size() == op.ColOffsets().Size(), "Operator is not square.");
            for (int i = 0; i < op.RowOffsets().Size(); i++)
                MFEM_VERIFY(op.RowOffsets()[i] == op.ColOffsets()[i], "Operator is not square!");
            MFEM_VERIFY(offsets_.Size() == op.RowOffsets().Size(), "Dimensions do not match.");
            for (int i = 0; i < op.RowOffsets().Size(); i++)
                MFEM_VERIFY(op.RowOffsets()[i] == offsets_[i], "Operator size does not match!");

            op_ = &op;
        }

        void Mult(const mfem::Vector &x, mfem::Vector &y) const override
        {
            mfem::Vector x0, x1, y0, y1;

            x0.MakeRef(const_cast<mfem::Vector &>(x), offsets_[0], offsets_[1] - offsets_[0]);
            x1.MakeRef(const_cast<mfem::Vector &>(x), offsets_[1], offsets_[2] - offsets_[1]);

            y0.MakeRef(y, offsets_[0], offsets_[1] - offsets_[0]);
            y1.MakeRef(y, offsets_[1], offsets_[2] - offsets_[1]);

            SobolevPreconditioner invA({&ND_}, {mass_}, {viscosity_});
            SobolevPreconditioner invS({&CG_}, {mass_}, {1. / viscosity_});

            mfem::Vector invA_f(x0.Size());
            mfem::Vector BT_invA_f_min_g(x1.Size());
            mfem::Vector p(x1.Size());
            mfem::Vector u(x0.Size());
            mfem::Vector B_p(x0.Size());
            // y0 = 1.;
            invA.Mult(x0, invA_f);
            op_->GetBlock(1, 0).Mult(invA_f, BT_invA_f_min_g);
            BT_invA_f_min_g -= x1;
            // mfem::ProductOperator invA_B(&invA, &op_->GetBlock(0,1), false, false);
            // mfem::ProductOperator BT_invA_B(&op_->GetBlock(1,0),&invA_B, false, false);

            invS.Mult(BT_invA_f_min_g, p);
            mfem::Vector f_min_B_p(x0.Size());
            f_min_B_p.Set(1., x0);
            op_->GetBlock(0, 1).AddMult(p, f_min_B_p, -1.);

            invA.Mult(f_min_B_p, u);
            y0.Set(1., u);
            y1.Set(1., p);
        }
    };
}

namespace hdiv{
    class StokesSolution
        : private OffsetsHolder,
        public mfem::Vector
    {
    private:
        mfem::GridFunction u_;
        mfem::GridFunction w_;
        mfem::GridFunction p_;

    public:
        StokesSolution(mfem::FiniteElementSpace &RT,
                    mfem::FiniteElementSpace &ND,
                    mfem::FiniteElementSpace &DG)
            : OffsetsHolder({&RT, &ND, &DG}),
            mfem::Vector(offsets_.Last()),
            u_(&RT, *this, 0), w_(&ND, *this, offsets_[1]), p_(&DG, *this, offsets_[2])
        {
            u_ = 0.;
            w_ = 0.;
            p_ = 0.;
        }

        mfem::GridFunction &get_u() { return u_; };
        mfem::GridFunction &get_w() { return w_; }
        mfem::GridFunction &get_p() { return p_; };
    };

    class StokesSystem
        : private OffsetsHolder,
        public mfem::BlockMatrix
    {
    private:

        mfem::FiniteElementSpace *RT_;
        mfem::MixedBilinearForm blf_B, blf_C, blf_BT, blf_CT;
        mfem::BilinearForm blf_D_;
        mfem::TransposeOperator BT, CT;
        mfem::SparseMatrix *BT_mat, *CT_mat, *A_mat;
        double mass_, viscosity_;
        mfem::ConstantCoefficient minus_one_coef_, viscosity_coef_;
        const mfem::Array<int> &ess_tdof_list_;

    public:
        StokesSystem(mfem::FiniteElementSpace &RT,
                    mfem::FiniteElementSpace &ND,
                    mfem::FiniteElementSpace &DG,
                    const mfem::Array<int> &ess_tdof_list,
                    double mass, double viscosity)
            : OffsetsHolder({&RT, &ND, &DG}) // 1) offsets constructed first
            ,
            mfem::BlockMatrix(offsets_) // 2) base MakeRef(offsets)
            ,
            RT_(&RT), blf_B(&RT, &ND), blf_BT(&ND,&RT), blf_CT(&RT,&DG), blf_C(&RT,&DG), blf_D_(&RT), BT(blf_B), CT(blf_C), ess_tdof_list_(ess_tdof_list), mass_(mass), viscosity_(viscosity), minus_one_coef_(-1.), viscosity_coef_(viscosity_)
        {
            // assemble operators
            mfem::ConstantCoefficient mass_coef(mass_), viscosity_coef(viscosity_);

            mfem::BilinearForm blf_A(RT_);
            blf_A.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(mass_coef));
            blf_A.Assemble();
            blf_A.Finalize(); // only if you need the explicit matrix (SparseMatrix)

            blf_B.AddDomainIntegrator(new mfem::MixedVectorWeakCurlIntegrator(viscosity_coef_));
            blf_B.Assemble();
            blf_B.Finalize(); // only if you need the explicit matrix

            blf_BT.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator(viscosity_coef_));
            blf_BT.Assemble();
            blf_BT.Finalize(); // only if you need the explicit matrix

            blf_C.AddDomainIntegrator(new mfem::MixedScalarDivergenceIntegrator(minus_one_coef_));
            blf_C.Assemble();
            blf_C.Finalize(); // only if you need the explicit matrix

            blf_CT.AddDomainIntegrator(new mfem::MixedScalarDivergenceIntegrator(minus_one_coef_));
            blf_CT.Assemble();
            blf_CT.Finalize(); // only if you need the explicit matrix

            blf_D_.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(minus_one_coef_));
            blf_D_.Assemble();
            blf_D_.Finalize();

            CT_mat = mfem::Transpose(blf_C.SpMat());
            // hook blocks
            A_mat = blf_A.LoseMat();
            SetBlock(0, 0, A_mat);
            SetBlock(0, 1, &blf_BT.SpMat());
            SetBlock(0, 2, CT_mat);
            SetBlock(1, 0, &blf_B.SpMat());
            SetBlock(1, 1, &blf_D_.SpMat());
            SetBlock(2, 0, &blf_C.SpMat());

            for(int i: ess_tdof_list_){
                GetBlock(0,0).EliminateRow(i, mfem::Operator::DiagonalPolicy::DIAG_ONE);
                GetBlock(0,1).EliminateRow(i, mfem::Operator::DiagonalPolicy::DIAG_ZERO);
                GetBlock(0,2).EliminateRow(i, mfem::Operator::DiagonalPolicy::DIAG_ZERO);
            }

        }



        void Update(mfem::VectorCoefficient &w_coef)
        {

            mfem::ConstantCoefficient mass_coef(mass_);

            delete A_mat;
            
            mfem::BilinearForm blf_A(RT_);
            blf_A.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(mass_coef));
            blf_A.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_coef));
            blf_A.Assemble();
            blf_A.Finalize(); 

            A_mat = blf_A.LoseMat();

            SetBlock(0, 0, A_mat);
            for(int i: ess_tdof_list_)
                GetBlock(0,0).EliminateRow(i, mfem::Operator::DiagonalPolicy::DIAG_ONE);

        }

        ~StokesSystem() {
            delete A_mat;
        }
    };

    class StokesRHS
        : private OffsetsHolder,
        public mfem::BlockVector
    {
    private:
        mfem::FiniteElementSpace *RT_, *ND_, *DG_;
        mfem::VectorFunctionCoefficient f_coef_;
        mfem::VectorFunctionCoefficient tr_u_coef_;
        const mfem::Array<int> &ess_tdof_list_;
        double viscosity_;

    public:
        StokesRHS(mfem::FiniteElementSpace &RT,
                mfem::FiniteElementSpace &ND,
                mfem::FiniteElementSpace &DG,
                const mfem::Array<int> &ess_tdof_list,
                std::function<void(const mfem::Vector &, double, mfem::Vector &)> f,
                std::function<void(const mfem::Vector &, double, mfem::Vector &)> tr_u,
                double viscosity = 1.)
            : OffsetsHolder({&RT, &ND, &DG}),
            mfem::BlockVector(offsets_),
            RT_(&RT),
            ND_(&ND),
            DG_(&DG),
            ess_tdof_list_(ess_tdof_list),
            f_coef_(DG.GetMesh()->Dimension(), std::move(f)),
            tr_u_coef_(DG.GetMesh()->Dimension(), std::move(tr_u)),
            viscosity_(viscosity)
        {
            mfem::GridFunction u_prev_dummy(RT_);
            u_prev_dummy = 0.;
            Update(u_prev_dummy, 0., 0.);
        }

        void Update(mfem::GridFunction& u_prev, double t, double mass)
        {
            mfem::VectorGridFunctionCoefficient u_prev_coef(&u_prev);
            mfem::ScalarVectorProductCoefficient mass_u_prev_coef(mass, u_prev_coef);

            f_coef_.SetTime(t);
            tr_u_coef_.SetTime(t);

            mfem::GridFunction tr_u(RT_);
            tr_u.ProjectCoefficient(tr_u_coef_);

            mfem::LinearForm f_lf(RT_);
            f_lf.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(mass_u_prev_coef));
            f_lf.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coef_));
            f_lf.Assemble();
            for(int i=0; i < ess_tdof_list_.Size(); ++i)
                f_lf[ess_tdof_list_[i]] = tr_u[ess_tdof_list_[i]];
            //f_lf.SetSubVector(ess_tdof_list_, tr_u);

            mfem::LinearForm g_lf(ND_);
            g_lf.AddBoundaryIntegrator(new mfem::VectorFEBoundaryTangentLFIntegrator(tr_u_coef_));
            g_lf.Assemble();

            GetBlock(0).Set(1., f_lf);
            GetBlock(1).Set(-1., g_lf);
            GetBlock(2) = 0.;
        }
    };



    class Solver
    : public mfem::Solver
    {
    private:
        double tol_;
        int &iterations_;
        mfem::GMRESSolver gmressolver_;

    public:
        Solver(int &iterations, double tol = 1e-10)
            : iterations_(iterations), tol_(tol)
        {}
        void SetOperator(const mfem::Operator &op)
        {    
            gmressolver_.SetOperator(op);
            gmressolver_.SetAbsTol(1e-15);
            gmressolver_.SetRelTol(tol_);
            gmressolver_.SetMaxIter(10000);
            gmressolver_.SetPrintLevel(1);
        }

        void Mult(const mfem::Vector &x, mfem::Vector &y) const override
        {
            gmressolver_.Mult(x,y);
        }
    };

    
}




#endif // STOKESOPERATORS_H
