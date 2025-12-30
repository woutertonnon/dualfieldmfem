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
    mfem::LinearForm f_lf_, g_lf_;
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
          f_lf_(&ND),
          g_lf_(&CG),
          f_coef_(CG.GetMesh()->Dimension(), std::move(f)),
          tr_u_coef_(CG.GetMesh()->Dimension(), std::move(tr_u)),
          theta_(theta), Cw_(Cw), viscosity_(viscosity)
    {
        f_lf_.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coef_));
        f_lf_.AddBdrFaceIntegrator(new WouterLFIntegrator(theta_, Cw_, tr_u_coef_, viscosity));
        f_lf_.Assemble();

        g_lf_.AddBoundaryIntegrator(new mfem::BoundaryNormalLFIntegrator(tr_u_coef_));
        g_lf_.Assemble();

        GetBlock(0).Set(1., f_lf_);
        GetBlock(1).Set(1., g_lf_);
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
    mfem::SparseMatrix *BT_mat;
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
        blf_A.AddBdrFaceIntegrator(new WouterIntegrator(theta_, Cw_, viscosity_));
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
/*
class StokesPressureOperator : public Operator
{
   const Operator *invA, *invAT, *B;
   mutable Vector z, q;

public:
   StokesPressureOperator(const Operator *invA, const Operator *invAT, const Operator *B);

   void Mult(const Vector &x, Vector &y) const override
   { B->Mult(x, z); invA->Mult(z, q); B->MultTranspose(q,y);  }

   void MultTranspose(const Vector &x, Vector &y) const override
   { A->MultTranspose(x, z); B->MultTranspose(z, y); }

   virtual ~ProductOperator();
};
*/


class SchurSolver
    : private OffsetsHolder,
      public mfem::Solver
{
private:
    mfem::BlockMatrix *op_;
    double mass_, viscosity_, tol_;
    mfem::FiniteElementSpace &ND_, &CG_;
    mfem::KLUSolver invA;

public:
    SchurSolver(mfem::FiniteElementSpace &ND,
                mfem::FiniteElementSpace &CG,
                double mass, double viscosity, double tol = 1e-8)
        : mfem::Solver(ND.GetVDim() + CG.GetVDim()), OffsetsHolder({&ND, &CG}), mass_(mass), viscosity_(viscosity), tol_(tol), ND_(ND), CG_(CG), invA()
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
        invA.SetOperator(op_->GetBlock(0, 0));
        //invA.SetAbsTol(tol_);
        //invA.SetRelTol(0.);
        //invA.SetMaxIter(10000);
    }

    void Mult(const mfem::Vector &x, mfem::Vector &y) const override
    {
        mfem::Vector x0, x1, y0, y1;

        x0.MakeRef(const_cast<mfem::Vector &>(x), offsets_[0], offsets_[1] - offsets_[0]);
        x1.MakeRef(const_cast<mfem::Vector &>(x), offsets_[1], offsets_[2] - offsets_[1]);

        y0.MakeRef(y, offsets_[0], offsets_[1] - offsets_[0]);
        y1.MakeRef(y, offsets_[1], offsets_[2] - offsets_[1]);

        std::cout << x1.Sum() << std::endl;
        x1 -= x1.Sum()/x1.Size();
        //x1 -= x1.Sum()/x1.Size();
        //std::cout << x1.Sum() << std::endl;

        mfem::GMRESSolver invS;

        SobolevPreconditioner invS_pre({&CG_},{.02},{1.});

        mfem::Vector invA_x0(x0.Size());
        mfem::Vector BT_invA_x0_min_x1(x1.Size());
        mfem::Vector p(x1.Size());
        mfem::Vector u(x0.Size());
        mfem::Vector B_p(x0.Size());
        // y0 = 1.;
        std::cout << "test1\n";
        invA.Mult(x0, invA_x0);
        std::cout << "test2\n";
        op_->GetBlock(1, 0).Mult(invA_x0, BT_invA_x0_min_x1);
        std::cout << "test3\n";
        BT_invA_x0_min_x1 -= x1;

        std::cout << BT_invA_x0_min_x1.Sum() << std::endl;
        std::cout << "test4\n";
        mfem::ProductOperator invA_B(&invA, &op_->GetBlock(0, 1), false, false);
        std::cout << "test5\n";
        mfem::ProductOperator BT_invA_B(&op_->GetBlock(1, 0), &invA_B, false, false);
        std::cout << "test6\n";

        invS.SetOperator(BT_invA_B);
        invS.SetKDim(3000);
        invS.SetPrintLevel(1);
        invS.SetAbsTol(tol_);
        invS.SetRelTol(0.);
        invS.SetMaxIter(10000);
        //invS.SetPreconditioner(invS_pre);
        std::cout << "test7\n";
        invS.Mult(BT_invA_x0_min_x1, y1);
        std::cout << "test8\n";


        mfem::Vector x0test(x0.Size());
        mfem::Vector x1test(x1.Size());
        mfem::Vector x0test2(x0.Size());
        x0test = 0.;
        x1test = 0.;
        x0test2=0.;
        //op_->GetBlock(0,1).Mult(y1,x0test);
        //invA.Mult(x0test, x0test2);
        //op_->GetBlock(1,0).Mult(x0test2,x1test);
        BT_invA_B.Mult(y1, x1test);
       // for(int i=0; i<x1test.Size();++i)
       //     std::cout << x1test[i]-x1[i] << std::endl;




        mfem::Vector x0_min_B_y1(x0.Size());
        x0_min_B_y1.Set(1., x0);
        op_->GetBlock(0, 1).AddMult(y1, x0_min_B_y1, -1.);

        // testing
        //mfem::Vector x0test(x0.Size());
        //op_->GetBlock(0,0).Mult(y0,x0test);
        //op_->GetBlock(0,1).AddMult(y1,x0test,1.);
        //for(int i=0; i<x0test.Size();++i)
        //    std::cout << x0test[i]-x0[i] << std::endl;


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

#endif // STOKESOPERATORS_H
