#ifndef STOKESOPERATORS_H
#define STOKESOPERATORS_H

#include <memory>
#include <utility>
#include <vector>

#include "mfem.hpp"
#include "BoundaryOperators.h"


/// Prints residual norm every @a freq iterations.
class PeriodicResidualMonitor : public mfem::IterativeSolverMonitor
{
    int freq_;
public:
    PeriodicResidualMonitor(int freq = 100) : freq_(freq) {}
    void MonitorResidual(int it, mfem::real_t norm, const mfem::Vector &r,
                         bool final) override
    {
        if (final || it % freq_ == 0)
        {
            mfem::out << "  iteration " << it << ", residual norm = "
                      << norm << std::endl;
        }
    }
};


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
    std::vector<std::unique_ptr<mfem::UMFPackSolver>> solvers_;
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
          bil_forms_(fes_array.size()), solvers_(fes_array.size())
    {
        for (int i = 0; i < (int)fes_array.size(); ++i)
        {
            auto *fes = fes_array[i];

            bil_forms_[i] = std::make_unique<mfem::BilinearForm>(fes);
            AddMassDiffIntegrators(*bil_forms_[i], mass_weights_[i], diff_weights_[i]);
            bil_forms_[i]->Assemble();
            bil_forms_[i]->Finalize();

            solvers_[i] = std::make_unique<mfem::UMFPackSolver>();
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
        double theta_, Cw_, viscosity_, delta_;
        // Optional boundary-attribute marker.  When non-empty, Nitsche RHS
        // and pressure-flux integrators are restricted to the marked faces
        // (e.g. the lid).  Unmarked faces get homogeneous Dirichlet (u=0)
        // through the bilinear form alone.
        mfem::Array<int> bdr_marker_;

    public:
        /// Construct the RHS.  If \p bdr_marker is non-null, the Nitsche
        /// and pressure boundary integrators are applied only on the marked
        /// boundary attributes (1-based, MFEM convention).
        StokesRHS(mfem::FiniteElementSpace &ND,
                mfem::FiniteElementSpace &CG,
                std::function<void(const mfem::Vector &, double, mfem::Vector &)> f,
                std::function<void(const mfem::Vector &, double, mfem::Vector &)> tr_u,
                double theta = 1.,
                double Cw = 100.,
                double viscosity = 1.,
                double delta = 0.0,
                const mfem::Array<int> *bdr_marker = nullptr)
            : OffsetsHolder({&ND, &CG}),
            mfem::BlockVector(offsets_),
            ND_(&ND),
            CG_(&CG),
            f_coef_(CG.GetMesh()->Dimension(), std::move(f)),
            tr_u_coef_(CG.GetMesh()->Dimension(), std::move(tr_u)),
            theta_(theta), Cw_(Cw), viscosity_(viscosity), delta_(delta)
        {
            if (bdr_marker) bdr_marker_ = *bdr_marker;

            mfem::LinearForm f_lf(ND_);
            f_lf.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coef_));
            if (bdr_marker_.Size() > 0)
                f_lf.AddBdrFaceIntegrator(new ND_NitscheLFIntegrator(theta_, Cw_, tr_u_coef_, viscosity_), bdr_marker_);
            else
                f_lf.AddBdrFaceIntegrator(new ND_NitscheLFIntegrator(theta_, Cw_, tr_u_coef_, viscosity_));
            f_lf.Assemble();

            mfem::LinearForm g_lf(CG_);
            if (bdr_marker_.Size() > 0)
                g_lf.AddBoundaryIntegrator(new mfem::BoundaryNormalLFIntegrator(tr_u_coef_), bdr_marker_);
            else
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
            if (bdr_marker_.Size() > 0)
                f_lf.AddBdrFaceIntegrator(new ND_NitscheLFIntegrator(theta_, Cw_, tr_u_coef_, viscosity_), bdr_marker_);
            else
                f_lf.AddBdrFaceIntegrator(new ND_NitscheLFIntegrator(theta_, Cw_, tr_u_coef_, viscosity_));
            f_lf.Assemble();

            mfem::LinearForm g_lf(CG_);
            if (bdr_marker_.Size() > 0)
                g_lf.AddBoundaryIntegrator(new mfem::BoundaryNormalLFIntegrator(tr_u_coef_), bdr_marker_);
            else
                g_lf.AddBoundaryIntegrator(new mfem::BoundaryNormalLFIntegrator(tr_u_coef_));
            g_lf.Assemble();

            GetBlock(0).Set(1., f_lf);
            GetBlock(1).Set(1., g_lf);

            // Consistent PSPG stabilization for the pressure block:
            //   +delta * int_Omega (f + (1/dt)*u^n) . grad(q) dx
            if (delta_ > 0.0)
            {
                mfem::ScalarVectorProductCoefficient delta_f(delta_, f_coef_);
                mfem::ScalarVectorProductCoefficient delta_mass_u(delta_ * mass, u_prev_coef);

                mfem::LinearForm stab_lf(CG_);
                stab_lf.AddDomainIntegrator(new mfem::DomainLFGradIntegrator(delta_f));
                stab_lf.AddDomainIntegrator(new mfem::DomainLFGradIntegrator(delta_mass_u));
                stab_lf.Assemble();

                GetBlock(1).Add(1., stab_lf);
            }
        }
    };

    class StokesSystem
        : private OffsetsHolder,
        public mfem::BlockMatrix
    {
    private:

        mfem::FiniteElementSpace *ND_, *CG_;
        mfem::MixedBilinearForm blf_B;
        mfem::TransposeOperator BT;
        mfem::SparseMatrix *BT_mat, *A_mat, *C_mat;
        double mass_, viscosity_, theta_, Cw_, sigma_, gamma_, upwind_scale_, delta_;

    public:
        StokesSystem(mfem::FiniteElementSpace &ND,
                    mfem::FiniteElementSpace &CG,
                    double mass, double viscosity, double theta, double Cw,
                    double sigma = 10.0, double gamma = 1.0,
                    double upwind_scale = 1.0, double delta = 0.0)
            : OffsetsHolder({&ND, &CG}) // 1) offsets constructed first
            ,
            mfem::BlockMatrix(offsets_) // 2) base MakeRef(offsets)
            ,
            ND_(&ND), CG_(&CG), blf_B(&CG, &ND), BT(blf_B), C_mat(nullptr), mass_(mass), viscosity_(viscosity), theta_(theta), Cw_(Cw), sigma_(sigma), gamma_(gamma), upwind_scale_(upwind_scale), delta_(delta)
        {
            // assemble operators
            mfem::ConstantCoefficient mass_coef(mass_), viscosity_coef(viscosity_);

            mfem::BilinearForm blf_A(ND_);
            blf_A.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(mass_coef));
            blf_A.AddDomainIntegrator(new mfem::CurlCurlIntegrator(viscosity_coef));
            blf_A.AddInteriorFaceIntegrator(new ND_DGPenaltyIntegrator(sigma_, viscosity_));
            blf_A.AddInteriorFaceIntegrator(new ND_CurlJumpIntegrator(gamma_, viscosity_));
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

            // Least-squares pressure stabilization (PSPG):
            //   -delta * int_Omega grad(p) . grad(q) dx
            // Regularises the (1,1) block of the saddle-point system.
            if (delta_ > 0.0)
            {
                mfem::ConstantCoefficient neg_delta_coef(-delta_);
                mfem::BilinearForm blf_C(CG_);
                blf_C.AddDomainIntegrator(new mfem::DiffusionIntegrator(neg_delta_coef));
                blf_C.Assemble();
                blf_C.Finalize();
                C_mat = blf_C.LoseMat();
                SetBlock(1, 1, C_mat);
            }
        }

        // Update the A block for a new timestep.
        //
        // w_coef      : vorticity used in the rotational convection term
        //               (curl(u^n) x u^{n+1}, v).
        // upwind_coef : velocity used as transport wind beta in the Heumann
        //               face stabilization  |beta.n_F| [[u]]·[[v]].
        //               Must be the streamline direction of the flow, not the
        //               vorticity (Proposition 2.3, Heumann & Hiptmair 2013):
        //                 single field:    beta = u      (self-transport)
        //                 dual field sys1: beta = u2     (curl(u2) transports
        //                                                 u1 along u2)
        //                 dual field sys2: beta = u1
        //               Defaults to w_coef when nullptr (for tests / backward
        //               compatibility where the distinction does not matter).
        void Update(mfem::VectorCoefficient &w_coef,
                    mfem::VectorCoefficient *upwind_coef = nullptr)
        {
            mfem::VectorCoefficient &face_wind =
                upwind_coef ? *upwind_coef : w_coef;

            mfem::ConstantCoefficient mass_coef(mass_), viscosity_coef(viscosity_);

            delete A_mat;

            mfem::BilinearForm blf_A(ND_);
            blf_A.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(mass_coef));
            blf_A.AddDomainIntegrator(new mfem::CurlCurlIntegrator(viscosity_coef));
            // Convection: (w x u, v)
            {
                auto *conv = new mfem::MixedCrossProductIntegrator(w_coef);
                const int k = ND_->GetMaxElementOrder();
                conv->SetIntRule(&mfem::IntRules.Get(
                    ND_->GetMesh()->GetElementGeometry(0), 5 * k));
                blf_A.AddDomainIntegrator(conv);
            }
            blf_A.AddInteriorFaceIntegrator(new ND_DGPenaltyIntegrator(sigma_, viscosity_));
            blf_A.AddInteriorFaceIntegrator(new ND_CurlJumpIntegrator(gamma_, viscosity_));
            // Heumann upwind: |beta.n_F| [[u]]·[[v]] with beta = transport velocity.
            // Wind-dependent, so included only in Update(), not the constructor.
            blf_A.AddInteriorFaceIntegrator(new ND_UpwindIntegrator(face_wind, upwind_scale_));
            blf_A.AddBdrFaceIntegrator(new ND_NitscheIntegrator(theta_, Cw_, viscosity_));
            blf_A.Assemble();
            blf_A.Finalize();

            A_mat = blf_A.LoseMat();

            SetBlock(0, 0, A_mat);

        }

        ~StokesSystem() {
            delete A_mat;
            delete C_mat;
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
        mfem::BilinearForm mass_bil_;

    public:
        SchurSolver(mfem::FiniteElementSpace &ND,
                    mfem::FiniteElementSpace &CG,
                    double mass, double viscosity, int& iterations, double tol = 1e-8)
            : mfem::Solver(ND.GetVDim() + CG.GetVDim()), OffsetsHolder({&ND, &CG}), mass_(mass), viscosity_(viscosity), iterations_(iterations), tol_(tol), ND_(ND), CG_(CG), invA(), mass_bil_(&ND)
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

    // Direct solver for the 2×2 H(curl) Stokes system
    //
    //   [A   B ] [u]   [f]
    //   [BT  0 ] [p] = [g]
    //
    // Assembles the BlockMatrix as a monolithic SparseMatrix and solves with
    // UMFPack.  This gives machine-precision accuracy without requiring a
    // preconditioner or tuning GMRES restart parameters.
    class DirectSolver
        : private OffsetsHolder,
          public mfem::Solver
    {
    private:
        mfem::BlockMatrix *op_;
        int &iterations_;

    public:
        DirectSolver(mfem::FiniteElementSpace &ND,
                     mfem::FiniteElementSpace &CG,
                     int &iterations)
            : OffsetsHolder({&ND, &CG}),
              mfem::Solver(offsets_.Last()),
              iterations_(iterations),
              op_(nullptr)
        {}

        void SetOperator(const mfem::Operator &op) override
        {
            throw std::invalid_argument("hcurl::DirectSolver::SetOperator(): expected mfem::BlockMatrix.");
        }

        void SetOperator(mfem::BlockMatrix &op)
        {
            op_ = &op;
        }

        void Mult(const mfem::Vector &x, mfem::Vector &y) const override
        {
            y.SetSize(op_->Height());
            mfem::Vector rhs(x);
            std::unique_ptr<mfem::SparseMatrix> mono(op_->CreateMonolithic());

            // Pin the first CG (pressure) DOF to zero to remove the constant-
            // pressure null space.  The monolithic matrix has no structural
            // diagonal entry in the CG rows (the (1,1) block is zero), so we
            // cannot call EliminateRowCol directly — we first inject the
            // diagonal entry by adding a single-entry matrix, then zero the row.
            const int pin = offsets_[1];
            const int N   = mono->Height();

            mfem::SparseMatrix diag_inj(N, N);
            diag_inj.Set(pin, pin, 1.0);
            diag_inj.Finalize();
            std::unique_ptr<mfem::SparseMatrix> aug(mfem::Add(*mono, diag_inj));

            // Zero row 'pin', set diagonal to 1, set rhs to 0.
            // Column 'pin' is intentionally left intact: since the solution will
            // have p[pin]=0, those entries multiply zero and do not affect y.
            {
                int    *I = aug->GetI();
                int    *J = aug->GetJ();
                double *A = aug->GetData();
                for (int k = I[pin]; k < I[pin+1]; ++k)
                    A[k] = (J[k] == pin) ? 1.0 : 0.0;
            }
            rhs[pin] = 0.0;

            mfem::UMFPackSolver direct;
            direct.SetOperator(*aug);
            direct.Mult(rhs, y);
            iterations_ = 1;
        }
    };

    // GMRES solver for the 2×2 H(curl) Stokes system
    //
    //   [A   B ] [u]   [f]
    //   [BT  0 ] [p] = [g]
    //
    // Block-diagonal preconditioner with exact UMFPack block solves:
    //   Block 0: exact solve on A (re-set each timestep)
    //   Block 1: exact solve on (1/viscosity)*M_CG (Schur complement approximation)
    class GMRESSolver
        : private OffsetsHolder,
          public mfem::Solver
    {
    private:
        mfem::BlockMatrix *op_;
        int &iterations_;
        double rel_tol_;

        mutable mfem::UMFPackSolver solver_A_;
        mfem::BilinearForm prec_p_;
        mfem::UMFPackSolver solver_p_;
        mutable mfem::BlockDiagonalPreconditioner prec_;

    public:
        GMRESSolver(mfem::FiniteElementSpace &ND,
                    mfem::FiniteElementSpace &CG,
                    int &iterations,
                    double viscosity,
                    double rel_tol = 1e-6)
            : OffsetsHolder({&ND, &CG}),
              mfem::Solver(offsets_.Last()),
              iterations_(iterations),
              op_(nullptr),
              rel_tol_(rel_tol),
              prec_p_(&CG),
              prec_(offsets_)
        {
            mfem::ConstantCoefficient inv_visc(1.0 / viscosity);
            prec_p_.AddDomainIntegrator(new mfem::MassIntegrator(inv_visc));
            prec_p_.Assemble();
            prec_p_.Finalize();
            solver_p_.SetOperator(prec_p_.SpMat());
        }

        void SetOperator(const mfem::Operator &op) override
        {
            throw std::invalid_argument(
                "hcurl::GMRESSolver::SetOperator(): expected mfem::BlockMatrix.");
        }

        void SetOperator(mfem::BlockMatrix &op)
        {
            op_ = &op;
            solver_A_.SetOperator(op.GetBlock(0, 0));
            prec_.SetDiagonalBlock(0, &solver_A_);
            prec_.SetDiagonalBlock(1, &solver_p_);
        }

        void Mult(const mfem::Vector &x, mfem::Vector &y) const override
        {
            PeriodicResidualMonitor monitor(100);
            mfem::GMRESSolver gmres;
            gmres.iterative_mode = true;
            gmres.SetOperator(*op_);
            gmres.SetPreconditioner(prec_);
            gmres.SetMonitor(monitor);
            gmres.SetRelTol(rel_tol_);
            gmres.SetAbsTol(1e-10);
            gmres.SetMaxIter(100000);
            gmres.SetKDim(5000);
            gmres.SetPrintLevel(0);

            y.SetSize(op_->Height());
            gmres.Mult(x, y);
            iterations_ = gmres.GetNumIterations();
        }
    };

#if defined(MFEM_USE_MPI) && defined(MFEM_HYPRE_VERSION)
    // GMRES solver for the 2x2 H(curl) Stokes system with HypreAMS on block A
    // and UMFPack on the pressure mass block.
    class GMRESAMSSolver
        : private OffsetsHolder,
          public mfem::Solver
    {
    private:
        mfem::BlockMatrix *op_;
        int &iterations_;
        double rel_tol_;
        mfem::FiniteElementSpace &ND_;

        mfem::BilinearForm prec_p_;
        mfem::UMFPackSolver solver_p_;
        mutable mfem::BlockDiagonalPreconditioner prec_;

        mutable std::unique_ptr<mfem::HypreParMatrix> parA_;
        mutable std::unique_ptr<mfem::HypreAMS> ams_;
        mutable std::unique_ptr<mfem::ParMesh> aux_pmesh_;
        mutable std::unique_ptr<mfem::ParFiniteElementSpace> aux_nd_par_;

        mfem::ParFiniteElementSpace *GetNDParFESpace() const
        {
            if (!mfem::Mpi::IsInitialized()) { mfem::Mpi::Init(); }

            auto *nd_par = dynamic_cast<mfem::ParFiniteElementSpace *>(&ND_);
            if (nd_par) { return nd_par; }

            const int mpi_size = mfem::Mpi::WorldSize();
            MFEM_VERIFY(mpi_size == 1,
                        "hcurl::GMRESAMSSolver with serial FiniteElementSpace "
                        "requires running on one MPI rank");

            if (!aux_nd_par_)
            {
                auto *mesh = ND_.GetMesh();
                MFEM_VERIFY(mesh != nullptr, "hcurl::GMRESAMSSolver: ND mesh is null");
                std::vector<int> partitioning(mesh->GetNE(), 0);
                aux_pmesh_ = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, *mesh, partitioning.data());
                aux_nd_par_ = std::make_unique<mfem::ParFiniteElementSpace>(aux_pmesh_.get(), ND_.FEColl());
            }
            return aux_nd_par_.get();
        }

    public:
        GMRESAMSSolver(mfem::FiniteElementSpace &ND,
                       mfem::FiniteElementSpace &CG,
                       int &iterations,
                       double viscosity,
                       double rel_tol = 1e-6)
            : OffsetsHolder({&ND, &CG}),
              mfem::Solver(offsets_.Last()),
              op_(nullptr),
              iterations_(iterations),
              rel_tol_(rel_tol),
              ND_(ND),
              prec_p_(&CG),
              prec_(offsets_)
        {
            mfem::ConstantCoefficient inv_visc(1.0 / viscosity);
            prec_p_.AddDomainIntegrator(new mfem::MassIntegrator(inv_visc));
            prec_p_.Assemble();
            prec_p_.Finalize();
            solver_p_.SetOperator(prec_p_.SpMat());
        }

        void SetOperator(const mfem::Operator &op) override
        {
            throw std::invalid_argument(
                "hcurl::GMRESAMSSolver::SetOperator(): expected mfem::BlockMatrix.");
        }

        void SetOperator(mfem::BlockMatrix &op)
        {
            op_ = &op;

            auto *A_sp = dynamic_cast<mfem::SparseMatrix *>(&op.GetBlock(0, 0));
            MFEM_VERIFY(A_sp != nullptr, "hcurl::GMRESAMSSolver expects SparseMatrix block (0,0)");

            auto *nd_par = GetNDParFESpace();

            HYPRE_BigInt row_starts[2] = {0, static_cast<HYPRE_BigInt>(A_sp->Height())};
            parA_ = std::make_unique<mfem::HypreParMatrix>(MPI_COMM_WORLD, A_sp->Height(), row_starts, A_sp);
            parA_->CopyRowStarts();

            ams_ = std::make_unique<mfem::HypreAMS>(*parA_, nd_par);
            ams_->SetPrintLevel(0);

            prec_.SetDiagonalBlock(0, ams_.get());
            prec_.SetDiagonalBlock(1, &solver_p_);
        }

        void Mult(const mfem::Vector &x, mfem::Vector &y) const override
        {
            PeriodicResidualMonitor monitor(100);
            mfem::FGMRESSolver gmres;
            gmres.iterative_mode = true;
            gmres.SetOperator(*op_);
            gmres.SetPreconditioner(prec_);
            gmres.SetMonitor(monitor);
            gmres.SetRelTol(rel_tol_);
            gmres.SetAbsTol(1e-10);
            gmres.SetMaxIter(100000);
            gmres.SetKDim(5000);
            gmres.SetPrintLevel(0);

            y.SetSize(op_->Height());
            gmres.Mult(x, y);
            iterations_ = gmres.GetNumIterations();
        }
    };
#else
    class GMRESAMSSolver
        : private OffsetsHolder,
          public mfem::Solver
    {
    public:
        GMRESAMSSolver(mfem::FiniteElementSpace &ND,
                       mfem::FiniteElementSpace &CG,
                       int &iterations,
                       double viscosity)
            : OffsetsHolder({&ND, &CG}), mfem::Solver(offsets_.Last())
        {
            MFEM_ABORT("hcurl::GMRESAMSSolver requires MFEM_USE_MPI and MFEM_USE_HYPRE");
        }
        void SetOperator(const mfem::Operator &op) override { }
        void SetOperator(mfem::BlockMatrix &op) { }
        void Mult(const mfem::Vector &x, mfem::Vector &y) const override { }
    };
#endif
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
        double mass_, viscosity_, sigma_, gamma_, Cw_;
        mfem::ConstantCoefficient minus_one_coef_, viscosity_coef_;
        const mfem::Array<int> &ess_tdof_list_;

    public:
        StokesSystem(mfem::FiniteElementSpace &RT,
                    mfem::FiniteElementSpace &ND,
                    mfem::FiniteElementSpace &DG,
                    const mfem::Array<int> &ess_tdof_list,
                    double mass, double viscosity, double sigma = 2.0, double Cw = 0.0, double gamma = 0.0)
            : OffsetsHolder({&RT, &ND, &DG}) // 1) offsets constructed first
            ,
            mfem::BlockMatrix(offsets_) // 2) base MakeRef(offsets)
            ,
            RT_(&RT), blf_B(&RT, &ND), blf_BT(&ND,&RT), blf_CT(&RT,&DG), blf_C(&RT,&DG), blf_D_(&ND), BT(blf_B), CT(blf_C), ess_tdof_list_(ess_tdof_list), mass_(mass), viscosity_(viscosity), sigma_(sigma), gamma_(gamma), Cw_(Cw), minus_one_coef_(-1.), viscosity_coef_(viscosity_)
        {
            // assemble operators
            mfem::ConstantCoefficient mass_coef(mass_), viscosity_coef(viscosity_);

            mfem::BilinearForm blf_A(RT_);
            blf_A.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(mass_coef));
            blf_A.AddInteriorFaceIntegrator(new RT_DGPenaltyIntegrator(sigma_, viscosity_));
            blf_A.AddInteriorFaceIntegrator(new RT_DivJumpIntegrator(gamma_, viscosity_));
            if (Cw_ > 0.0)
                blf_A.AddBdrFaceIntegrator(new RT_BdrTangentPenaltyIntegrator(Cw_));
            blf_A.Assemble();
            blf_A.Finalize(); // only if you need the explicit matrix (SparseMatrix)

            blf_B.AddDomainIntegrator(new mfem::MixedVectorWeakCurlIntegrator());
            blf_B.AddBdrFaceIntegrator(new RT_ND_BdrCrossProductIntegrator());
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
            // Convection: (w x u, v)
            {
                auto *conv = new mfem::MixedCrossProductIntegrator(w_coef);
                const int k = RT_->GetMaxElementOrder();
                conv->SetIntRule(&mfem::IntRules.Get(
                    RT_->GetMesh()->GetElementGeometry(0), 5 * k));
                blf_A.AddDomainIntegrator(conv);
            }
            blf_A.AddInteriorFaceIntegrator(new RT_DGPenaltyIntegrator(sigma_, viscosity_));
            blf_A.AddInteriorFaceIntegrator(new RT_DivJumpIntegrator(gamma_, viscosity_));
            if (Cw_ > 0.0)
                blf_A.AddBdrFaceIntegrator(new RT_BdrTangentPenaltyIntegrator(Cw_));
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
        mfem::Array<int> ess_tdof_lid_;  // DOFs on lid (get tr_u values)
        mfem::Array<int> bdr_marker_;   // boundary attribute marker for lid faces
        double viscosity_, Cw_;
        bool has_lid_;

    public:
        /// Construct the RHS.  If \p bdr_marker is non-null, tr_u is applied
        /// only on the marked boundary faces; all other essential DOFs get u=0.
        StokesRHS(mfem::FiniteElementSpace &RT,
                mfem::FiniteElementSpace &ND,
                mfem::FiniteElementSpace &DG,
                const mfem::Array<int> &ess_tdof_list,
                std::function<void(const mfem::Vector &, double, mfem::Vector &)> f,
                std::function<void(const mfem::Vector &, double, mfem::Vector &)> tr_u,
                double viscosity = 1.,
                const mfem::Array<int> *bdr_marker = nullptr,
                double Cw = 0.0)
            : OffsetsHolder({&RT, &ND, &DG}),
            mfem::BlockVector(offsets_),
            RT_(&RT),
            ND_(&ND),
            DG_(&DG),
            ess_tdof_list_(ess_tdof_list),
            f_coef_(DG.GetMesh()->Dimension(), std::move(f)),
            tr_u_coef_(DG.GetMesh()->Dimension(), std::move(tr_u)),
            viscosity_(viscosity),
            Cw_(Cw),
            has_lid_(bdr_marker != nullptr)
        {
            if (has_lid_)
            {
                bdr_marker_ = *bdr_marker;
                RT.GetEssentialTrueDofs(bdr_marker_, ess_tdof_lid_);
            }

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
            if (Cw_ > 0.0)
            {
                if (has_lid_)
                    f_lf.AddBdrFaceIntegrator(new RT_BdrTangentPenaltyLFIntegrator(Cw_, tr_u_coef_), bdr_marker_);
                else
                    f_lf.AddBdrFaceIntegrator(new RT_BdrTangentPenaltyLFIntegrator(Cw_, tr_u_coef_));
            }
            f_lf.Assemble();

            if (has_lid_)
            {
                // Zero all essential DOFs, then set lid DOFs to tr_u
                for (int i = 0; i < ess_tdof_list_.Size(); ++i)
                    f_lf[ess_tdof_list_[i]] = 0.0;
                for (int i = 0; i < ess_tdof_lid_.Size(); ++i)
                    f_lf[ess_tdof_lid_[i]] = tr_u[ess_tdof_lid_[i]];
            }
            else
            {
                // Apply tr_u on all essential DOFs
                for (int i = 0; i < ess_tdof_list_.Size(); ++i)
                    f_lf[ess_tdof_list_[i]] = tr_u[ess_tdof_list_[i]];
            }

            GetBlock(0).Set(1., f_lf);
            // Block (1) enforces the weak vorticity relation via
            // B(u) - M w = 0. The boundary cross term is already part of
            // B through RT_ND_BdrCrossProductIntegrator in StokesSystem.
            // Adding an extra boundary RHS here double-counts that trace and
            // corrupts the coupled vorticity used by the H(curl) solve.
            GetBlock(1) = 0.;
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

    class BlockDiagPreconditioner
        : private OffsetsHolder,
          public mfem::Solver
    {
    private:
        mfem::FiniteElementSpace &RT_, &ND_, &DG_;
        double mass_;

        // A block: exact solve, re-set each timestep
        mutable mfem::UMFPackSolver solver_A_;

        // D block: exact solve on -M_ND (static, set once in constructor)
        mfem::BilinearForm mass_D_;
        mfem::UMFPackSolver solver_D_;

        // Pressure block: exact solve on (1/mass)*M_DG
        mfem::BilinearForm mass_p_;
        mfem::UMFPackSolver solver_p_;

    public:
        BlockDiagPreconditioner(mfem::FiniteElementSpace &RT,
                                mfem::FiniteElementSpace &ND,
                                mfem::FiniteElementSpace &DG,
                                double mass)
            : OffsetsHolder({&RT, &ND, &DG}),
              mfem::Solver(offsets_.Last()),
              RT_(RT), ND_(ND), DG_(DG), mass_(mass),
              mass_D_(&ND),
              mass_p_(&DG)
        {
            // D block: assemble -M_ND
            mfem::ConstantCoefficient minus_one(-1.0);
            mass_D_.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(minus_one));
            mass_D_.Assemble();
            mass_D_.Finalize();
            solver_D_.SetOperator(mass_D_.SpMat());

            // Pressure block: (1/mass) * M_DG
            mfem::ConstantCoefficient inv_mass_coef(1.0 / mass_);
            mass_p_.AddDomainIntegrator(new mfem::MassIntegrator(inv_mass_coef));
            mass_p_.Assemble();
            mass_p_.Finalize();
            solver_p_.SetOperator(mass_p_.SpMat());
        }

        void SetOperator(const mfem::Operator &op) override
        {
            throw std::invalid_argument("BlockDiagPreconditioner::SetOperator(): expected mfem::BlockMatrix.");
        }

        void SetOperator(mfem::BlockMatrix &op)
        {
            solver_A_.SetOperator(op.GetBlock(0, 0));
        }

        void Mult(const mfem::Vector &x, mfem::Vector &y) const override
        {
            mfem::Vector x0, x1, x2, y0, y1, y2;

            x0.MakeRef(const_cast<mfem::Vector &>(x), offsets_[0], offsets_[1] - offsets_[0]);
            x1.MakeRef(const_cast<mfem::Vector &>(x), offsets_[1], offsets_[2] - offsets_[1]);
            x2.MakeRef(const_cast<mfem::Vector &>(x), offsets_[2], offsets_[3] - offsets_[2]);

            y0.MakeRef(y, offsets_[0], offsets_[1] - offsets_[0]);
            y1.MakeRef(y, offsets_[1], offsets_[2] - offsets_[1]);
            y2.MakeRef(y, offsets_[2], offsets_[3] - offsets_[2]);

            solver_A_.Mult(x0, y0);   // A block: exact solve
            solver_D_.Mult(x1, y1);   // D block: exact solve
            solver_p_.Mult(x2, y2);   // Pressure block: exact solve
        }
    };

    // Direct solver for the 3×3 H(div) Stokes system
    //
    //   [A    BT   CT] [u]   [f]
    //   [B    D    0 ] [w] = [g]
    //   [C    0    0 ] [p]   [h]
    //
    // Assembles the BlockMatrix as a monolithic SparseMatrix and solves with
    // UMFPack.  This gives machine-precision accuracy without requiring a
    // preconditioner or tuning GMRES restart parameters.
    class DirectSolver
        : private OffsetsHolder,
          public mfem::Solver
    {
    private:
        mfem::BlockMatrix *op_;
        int &iterations_;
        mfem::FiniteElementSpace &RT_, &ND_, &DG_;

    public:
        DirectSolver(mfem::FiniteElementSpace &RT,
                     mfem::FiniteElementSpace &ND,
                     mfem::FiniteElementSpace &DG,
                     int &iterations)
            : OffsetsHolder({&RT, &ND, &DG}),
              mfem::Solver(offsets_.Last()),
              RT_(RT), ND_(ND), DG_(DG),
              iterations_(iterations),
              op_(nullptr)
        {}

        void SetOperator(const mfem::Operator &op) override
        {
            throw std::invalid_argument("hdiv::DirectSolver::SetOperator(): expected mfem::BlockMatrix.");
        }

        void SetOperator(mfem::BlockMatrix &op)
        {
            op_ = &op;
        }

        void Mult(const mfem::Vector &x, mfem::Vector &y) const override
        {
            y.SetSize(op_->Height());
            mfem::Vector rhs(x);
            std::unique_ptr<mfem::SparseMatrix> mono(op_->CreateMonolithic());

            // Pin the first DG (pressure) DOF to zero to remove the constant-
            // pressure null space.  The (2,2) block is zero, so we inject a
            // diagonal entry and then zero the row, same as hcurl::DirectSolver.
            const int pin = offsets_[2];
            const int N   = mono->Height();

            mfem::SparseMatrix diag_inj(N, N);
            diag_inj.Set(pin, pin, 1.0);
            diag_inj.Finalize();
            std::unique_ptr<mfem::SparseMatrix> aug(mfem::Add(*mono, diag_inj));

            {
                int    *I = aug->GetI();
                int    *J = aug->GetJ();
                double *A = aug->GetData();
                for (int k = I[pin]; k < I[pin+1]; ++k)
                    A[k] = (J[k] == pin) ? 1.0 : 0.0;
            }
            rhs[pin] = 0.0;

            mfem::UMFPackSolver direct;
            direct.SetOperator(*aug);
            direct.Mult(rhs, y);
            iterations_ = 1;
        }
    };

    // GMRES solver for the 3×3 H(div) Stokes system
    //
    //   [A    BT   CT] [u]   [f]
    //   [B    D    0 ] [w] = [g]
    //   [C    0    0 ] [p]   [h]
    //
    // Block-diagonal preconditioner with exact UMFPack block solves:
    //   Block 0: exact solve on A (re-set each timestep)
    //   Block 1: exact solve on D = -M_ND (static)
    //   Block 2: exact solve on (1/mass)*M_DG (Schur complement approximation)
    class GMRESSolver
        : private OffsetsHolder,
          public mfem::Solver
    {
    private:
        mfem::BlockMatrix *op_;
        int &iterations_;
        double rel_tol_;

        // Block-diagonal exact preconditioner
        mutable BlockDiagPreconditioner prec_;

    public:
        GMRESSolver(mfem::FiniteElementSpace &RT,
                    mfem::FiniteElementSpace &ND,
                    mfem::FiniteElementSpace &DG,
                    int &iterations,
                    double mass,
                    double rel_tol = 1e-6)
            : OffsetsHolder({&RT, &ND, &DG}),
              mfem::Solver(offsets_.Last()),
              iterations_(iterations),
              op_(nullptr),
              rel_tol_(rel_tol),
              prec_(RT, ND, DG, mass)
        {}

        void SetOperator(const mfem::Operator &op) override
        {
            throw std::invalid_argument(
                "hdiv::GMRESSolver::SetOperator(): expected mfem::BlockMatrix.");
        }

        void SetOperator(mfem::BlockMatrix &op)
        {
            op_ = &op;
            prec_.SetOperator(op);
        }

        void Mult(const mfem::Vector &x, mfem::Vector &y) const override
        {
            PeriodicResidualMonitor monitor(100);
            mfem::GMRESSolver gmres;
            gmres.iterative_mode = true;
            gmres.SetOperator(*op_);
            gmres.SetPreconditioner(prec_);
            gmres.SetMonitor(monitor);
            gmres.SetRelTol(rel_tol_);
            gmres.SetAbsTol(1e-10);
            gmres.SetMaxIter(100000);
            gmres.SetKDim(5000);
            gmres.SetPrintLevel(0);

            y.SetSize(op_->Height());
            gmres.Mult(x, y);
            iterations_ = gmres.GetNumIterations();
        }
    };

#if defined(MFEM_USE_MPI) && defined(MFEM_HYPRE_VERSION)
    class ADSBlockDiagPreconditioner
        : private OffsetsHolder,
          public mfem::Solver
    {
    private:
        mfem::FiniteElementSpace &RT_, &ND_, &DG_;
        double mass_;

        mutable std::unique_ptr<mfem::HypreParMatrix> parA_;
        mutable std::unique_ptr<mfem::HypreADS> ads_;
        mutable std::unique_ptr<mfem::ParMesh> aux_pmesh_;
        mutable std::unique_ptr<mfem::ParFiniteElementSpace> aux_rt_par_;

        mfem::ParFiniteElementSpace *GetRTParFESpace() const
        {
            if (!mfem::Mpi::IsInitialized()) { mfem::Mpi::Init(); }

            auto *rt_par = dynamic_cast<mfem::ParFiniteElementSpace *>(&RT_);
            if (rt_par) { return rt_par; }

            const int mpi_size = mfem::Mpi::WorldSize();
            MFEM_VERIFY(mpi_size == 1,
                        "hdiv::ADSBlockDiagPreconditioner with serial FiniteElementSpace "
                        "requires running on one MPI rank");

            if (!aux_rt_par_)
            {
                auto *mesh = RT_.GetMesh();
                MFEM_VERIFY(mesh != nullptr, "hdiv::ADSBlockDiagPreconditioner: RT mesh is null");
                std::vector<int> partitioning(mesh->GetNE(), 0);
                aux_pmesh_ = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, *mesh, partitioning.data());
                aux_rt_par_ = std::make_unique<mfem::ParFiniteElementSpace>(aux_pmesh_.get(), RT_.FEColl());
            }
            return aux_rt_par_.get();
        }

        mfem::BilinearForm mass_D_;
        mfem::UMFPackSolver solver_D_;

        mfem::BilinearForm mass_p_;
        mfem::UMFPackSolver solver_p_;

    public:
        ADSBlockDiagPreconditioner(mfem::FiniteElementSpace &RT,
                                   mfem::FiniteElementSpace &ND,
                                   mfem::FiniteElementSpace &DG,
                                   double mass)
            : OffsetsHolder({&RT, &ND, &DG}),
              mfem::Solver(offsets_.Last()),
              RT_(RT), ND_(ND), DG_(DG), mass_(mass),
              mass_D_(&ND),
              mass_p_(&DG)
        {
            mfem::ConstantCoefficient minus_one(-1.0);
            mass_D_.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(minus_one));
            mass_D_.Assemble();
            mass_D_.Finalize();
            solver_D_.SetOperator(mass_D_.SpMat());

            mfem::ConstantCoefficient inv_mass_coef(1.0 / mass_);
            mass_p_.AddDomainIntegrator(new mfem::MassIntegrator(inv_mass_coef));
            mass_p_.Assemble();
            mass_p_.Finalize();
            solver_p_.SetOperator(mass_p_.SpMat());
        }

        void SetOperator(const mfem::Operator &op) override
        {
            throw std::invalid_argument(
                "ADSBlockDiagPreconditioner::SetOperator(): expected mfem::BlockMatrix.");
        }

        void SetOperator(mfem::BlockMatrix &op)
        {
            auto *A_sp = dynamic_cast<mfem::SparseMatrix *>(&op.GetBlock(0, 0));
            MFEM_VERIFY(A_sp != nullptr, "ADSBlockDiagPreconditioner expects SparseMatrix block (0,0)");

            auto *rt_par = GetRTParFESpace();

            HYPRE_BigInt row_starts[2] = {0, static_cast<HYPRE_BigInt>(A_sp->Height())};
            parA_ = std::make_unique<mfem::HypreParMatrix>(MPI_COMM_WORLD, A_sp->Height(), row_starts, A_sp);
            parA_->CopyRowStarts();

            ads_ = std::make_unique<mfem::HypreADS>(*parA_, rt_par);
            ads_->SetPrintLevel(0);
        }

        void Mult(const mfem::Vector &x, mfem::Vector &y) const override
        {
            mfem::Vector x0, x1, x2, y0, y1, y2;

            x0.MakeRef(const_cast<mfem::Vector &>(x), offsets_[0], offsets_[1] - offsets_[0]);
            x1.MakeRef(const_cast<mfem::Vector &>(x), offsets_[1], offsets_[2] - offsets_[1]);
            x2.MakeRef(const_cast<mfem::Vector &>(x), offsets_[2], offsets_[3] - offsets_[2]);

            y0.MakeRef(y, offsets_[0], offsets_[1] - offsets_[0]);
            y1.MakeRef(y, offsets_[1], offsets_[2] - offsets_[1]);
            y2.MakeRef(y, offsets_[2], offsets_[3] - offsets_[2]);

            ads_->Mult(x0, y0);
            solver_D_.Mult(x1, y1);
            solver_p_.Mult(x2, y2);
        }
    };

    class GMRESADSSolver
        : private OffsetsHolder,
          public mfem::Solver
    {
    private:
        mfem::BlockMatrix *op_;
        int &iterations_;
        double rel_tol_;
        mutable ADSBlockDiagPreconditioner prec_;

    public:
        GMRESADSSolver(mfem::FiniteElementSpace &RT,
                       mfem::FiniteElementSpace &ND,
                       mfem::FiniteElementSpace &DG,
                       int &iterations,
                       double mass,
                       double rel_tol = 1e-6)
            : OffsetsHolder({&RT, &ND, &DG}),
              mfem::Solver(offsets_.Last()),
              op_(nullptr),
              iterations_(iterations),
              rel_tol_(rel_tol),
              prec_(RT, ND, DG, mass)
        {
        }

        void SetOperator(const mfem::Operator &op) override
        {
            throw std::invalid_argument(
                "hdiv::GMRESADSSolver::SetOperator(): expected mfem::BlockMatrix.");
        }

        void SetOperator(mfem::BlockMatrix &op)
        {
            op_ = &op;
            prec_.SetOperator(op);
        }

        void Mult(const mfem::Vector &x, mfem::Vector &y) const override
        {
            PeriodicResidualMonitor monitor(100);
            mfem::FGMRESSolver gmres;
            gmres.iterative_mode = true;
            gmres.SetOperator(*op_);
            gmres.SetPreconditioner(prec_);
            gmres.SetMonitor(monitor);
            gmres.SetRelTol(rel_tol_);
            gmres.SetAbsTol(1e-10);
            gmres.SetMaxIter(100000);
            gmres.SetKDim(5000);
            gmres.SetPrintLevel(0);

            y.SetSize(op_->Height());
            gmres.Mult(x, y);
            iterations_ = gmres.GetNumIterations();
        }
    };
#else
    class GMRESADSSolver
        : private OffsetsHolder,
          public mfem::Solver
    {
    public:
        GMRESADSSolver(mfem::FiniteElementSpace &RT,
                       mfem::FiniteElementSpace &ND,
                       mfem::FiniteElementSpace &DG,
                       int &iterations,
                       double mass,
                       double rel_tol = 1e-6)
            : OffsetsHolder({&RT, &ND, &DG}), mfem::Solver(offsets_.Last())
        {
            MFEM_ABORT("hdiv::GMRESADSSolver requires MFEM_USE_MPI and MFEM_USE_HYPRE");
        }
        void SetOperator(const mfem::Operator &op) override { }
        void SetOperator(mfem::BlockMatrix &op) { }
        void Mult(const mfem::Vector &x, mfem::Vector &y) const override { }
    };
#endif


}



#endif // STOKESOPERATORS_H
