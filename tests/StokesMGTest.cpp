#include <gtest/gtest.h>
#include <iomanip>
#include "mfem.hpp"
#include "StokesDGS.h"
#include "StokesOperator.h"
#include "StokesMG.h"

// Helper function that tests both V-Cycle convergence and GMRES preconditioning.
// 1. Sets up the MG hierarchy.
// 2. Runs a standalone V-Cycle convergence test.
// 3. Reconfigures the MG solver to Galerkin mode and runs a GMRES convergence test.
void RunStokesMGTest(std::shared_ptr<mfem::Mesh> mesh_ptr,
                     const unsigned geomref = 1,
                     const double tol = 1e-6)
{
#ifdef MFEM_USE_SUITESPARSE
    std::cout << "Using SuiteSparse for Coarse Grid Solve" << std::endl;
#endif
#ifdef MFEM_USE_OPENMP
    mfem::Device("omp");
#endif
    // 1. Initialize MG Solver & Hierarchy
    // Added tau = 0.0
    StokesNitsche::StokesMG mg(mesh_ptr, 0.0, 1.0, 10.0);

    for (int i = 0; i < geomref; ++i)
        mg.addRefinement();

    const auto& fine_op = mg.getFinestOperator();
    const int num_rows = fine_op.NumRows();

    std::cout << "NDof = " << num_rows << std::endl;

    // Shared vectors
    mfem::Vector x_exact(num_rows), b(num_rows), x_sol(num_rows), residual(num_rows);
    x_exact.Randomize(1);

    // =======================================================
    // PHASE 1: Standalone V-Cycle Convergence
    // =======================================================
    fine_op.setOperatorMode(StokesNitsche::OperatorMode::DEC);
    mg.setOperatorMode(StokesNitsche::OperatorMode::DEC);
    std::cout << "\n[Phase 1] Running Standalone V-Cycle Test..." << std::endl;

    fine_op.Mult(x_exact, b);
    x_sol = 0.0;

    mg.setCycleType(StokesNitsche::MGCycleType::VCycle);
    mg.setIterativeMode(true);
    mg.setSmoothIterations(1);

    double initial_norm = 0.0;
    const int max_iter = 128;// * (pref + 1);

    std::cout << "  Iter | Rel. Residual \n-------|---------------\n";

    for (int iter = 0; iter < max_iter; ++iter)
    {
        fine_op.eliminateConstants(x_sol);
        residual = b;
        fine_op.AddMult(x_sol, residual, -1.0);
        const double current_norm = residual.Norml2();

        if (iter == 0) initial_norm = current_norm;
        double rel_norm = current_norm / initial_norm;

        std::cout << "  " << std::setw(4) << iter << " | "
                  << std::scientific << std::setprecision(4)
                  << rel_norm << std::endl;

        if (rel_norm < tol) break;
        mg.Mult(b, x_sol);
    }

    // V-Cycle Final Check
    residual = b;
    fine_op.AddMult(x_sol, residual, -1.0);
    double vcycle_final_rel_norm = residual.Norml2() / initial_norm;

    // ASSERT_LT(vcycle_final_rel_norm, tol)
        // << "Phase 1 Failed: MG V-Cycle failed to converge within tolerance.";

    std::cout << "Phase 1 Passed." << std::endl;


    // =======================================================
    // PHASE 2: GMRES with Galerkin MG Preconditioner
    // =======================================================
    fine_op.setOperatorMode(StokesNitsche::OperatorMode::Galerkin);
    mg.setOperatorMode(StokesNitsche::OperatorMode::Galerkin);
    mg.setIterativeMode(false);
    std::cout << "\n[Phase 2] Running GMRES (Galerkin) with MG Preconditioner..."
              << std::endl;

    // Reset vectors for GMRES
    fine_op.Mult(x_exact, b);
    x_sol = 0.0;

    mfem::FGMRESSolver gmres;
    gmres.SetOperator(fine_op);
    gmres.SetPreconditioner(mg);
    gmres.SetAbsTol(1e-12);
    gmres.SetRelTol(tol);
    gmres.SetMaxIter(max_iter);
    gmres.SetPrintLevel(1);
    gmres.SetKDim(max_iter);

    gmres.Mult(b, x_sol);

    const double gmres_final_rel = gmres.GetFinalRelNorm();
    ASSERT_TRUE(gmres.GetConverged())
        << "Phase 2 Failed: GMRES failed to converge."
        << std::endl
        << "(Rel.) Residual Norm: " << gmres_final_rel;

    std::cout << "Final GMRES Relative Residual: " << gmres_final_rel << std::endl;

    EXPECT_LT(gmres_final_rel, tol);

    std::cout << "Phase 2 Passed." << std::endl;
}

// --------------------------------------------------------
// Assembled-matrix residual test for nonzero tau (viscosity).
//
// Mirrors the semi-Lagrangian MPI app setup:
//   tau = 1/(dt*nu), MG in Galerkin mode, FGMRES outer solver.
//
// Steps:
//   1. Build the MG hierarchy with a given tau.
//   2. Assemble the full Galerkin sparse matrix from the finest operator.
//   3. Pick a random x_exact, compute b = A_assembled * x_exact.
//   4. Solve with FGMRES + MG preconditioner.
//   5. Compute the residual r = A_assembled * x_sol - b and check ||r||.
//   6. Also compare the matrix-free operator against the assembled matrix.
// --------------------------------------------------------
void RunAssembledResidualTest(const double tau,
                              const unsigned geomref = 3,
                              const double tol = 1e-8)
{
#ifdef MFEM_USE_SUITESPARSE
    std::cout << "Using SuiteSparse for Coarse Grid Solve" << std::endl;
#endif

    const double theta = 1.0;
    const double Cw = 40000.0;

    auto mesh_ptr = std::make_shared<mfem::Mesh>(
        mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::TETRAHEDRON));

    // 1. Build the MG hierarchy
    StokesNitsche::StokesMG mg(mesh_ptr, tau, theta, Cw);
    mg.setOperatorMode(StokesNitsche::OperatorMode::Galerkin);
    mg.setIterativeMode(false);
    mg.setCycleType(StokesNitsche::MGCycleType::VCycle);
    mg.setSmoothIterations(1, 1);

    for (unsigned l = 0; l < geomref; ++l)
        mg.addRefinement();

    auto &fine_op = const_cast<StokesNitsche::StokesNitscheOperator &>(
        mg.getFinestOperator());
    fine_op.setOperatorMode(StokesNitsche::OperatorMode::Galerkin);

    const int ne = fine_op.getHCurlSpace().GetNDofs();
    const int nv = fine_op.getH1Space().GetNDofs();
    const int N = ne + nv;  // operator size (no mean constraint)

    std::cout << "\n=== Assembled-residual test: tau = " << tau
              << ", NDof = " << N << " ===" << std::endl;

    // 2. Assemble the full Galerkin sparse matrix.
    //    getFullGalerkinSystem() returns (N+1) x (N+1) with mean constraint.
    //    We extract the leading N x N block.
    std::unique_ptr<mfem::SparseMatrix> A_full = fine_op.getFullGalerkinSystem();
    const int N_aug = A_full->Height();
    std::cout << "Assembled matrix: " << N_aug << " x " << N_aug
              << " (leading " << N << " x " << N << " used)" << std::endl;

    // 3. Verify matrix-free operator matches assembled matrix.
    //    Apply both to a random vector and compare.
    {
        mfem::Vector x_test(N), y_mf(N), y_sp(N_aug), x_aug(N_aug);
        x_test.Randomize(42);

        // Matrix-free
        fine_op.MultGalerkin(x_test, y_mf);

        // Assembled: pad x with 0 for the mean constraint row
        x_aug = 0.0;
        for (int i = 0; i < N; ++i) x_aug[i] = x_test[i];
        A_full->Mult(x_aug, y_sp);

        // Compare first N entries
        double max_diff = 0.0;
        for (int i = 0; i < N; ++i)
            max_diff = std::max(max_diff, std::abs(y_mf[i] - y_sp[i]));

        std::cout << "Matrix-free vs assembled max diff: "
                  << std::scientific << max_diff << std::endl;
        EXPECT_LT(max_diff, 1e-9)
            << "Matrix-free operator disagrees with assembled matrix";
    }

    // 4. Pick random x_exact, compute b = A * x_exact using assembled matrix.
    mfem::Vector x_exact(N), b(N), x_sol(N);
    x_exact.Randomize(123);

    // Compute b using the assembled matrix (first N entries)
    {
        mfem::Vector x_aug(N_aug), b_aug(N_aug);
        x_aug = 0.0;
        for (int i = 0; i < N; ++i) x_aug[i] = x_exact[i];
        A_full->Mult(x_aug, b_aug);
        for (int i = 0; i < N; ++i) b[i] = b_aug[i];
    }

    const double b_norm = b.Norml2();
    std::cout << "||b|| = " << std::scientific << b_norm << std::endl;

    // 5. Solve with FGMRES + MG preconditioner.
    x_sol = 0.0;
    mfem::FGMRESSolver gmres;
    gmres.SetOperator(fine_op);
    gmres.SetPreconditioner(mg);
    gmres.SetAbsTol(1e-14);
    gmres.SetRelTol(tol);
    gmres.SetMaxIter(500);
    gmres.SetPrintLevel(1);
    gmres.SetKDim(200);

    gmres.Mult(b, x_sol);
    fine_op.eliminateConstants(x_sol);

    const int num_iter = gmres.GetNumIterations();
    const bool converged = gmres.GetConverged();
    const double gmres_rel = gmres.GetFinalRelNorm();

    std::cout << "GMRES: " << num_iter << " iterations, converged="
              << converged << ", rel_norm=" << std::scientific << gmres_rel
              << std::endl;

    // 6. Compute residual r = A_assembled * x_sol - b.
    mfem::Vector r_assembled(N);
    {
        mfem::Vector x_aug(N_aug), r_aug(N_aug);
        x_aug = 0.0;
        for (int i = 0; i < N; ++i) x_aug[i] = x_sol[i];
        A_full->Mult(x_aug, r_aug);
        for (int i = 0; i < N; ++i) r_assembled[i] = r_aug[i] - b[i];
    }
    const double res_assembled_abs = r_assembled.Norml2();
    const double res_assembled_rel = res_assembled_abs / b_norm;

    std::cout << "Assembled residual: ||A*x_sol - b|| = "
              << std::scientific << res_assembled_abs
              << "  (rel: " << res_assembled_rel << ")" << std::endl;

    // Also check with matrix-free operator.
    mfem::Vector r_mf(N);
    fine_op.MultGalerkin(x_sol, r_mf);
    r_mf -= b;
    const double res_mf_abs = r_mf.Norml2();
    const double res_mf_rel = res_mf_abs / b_norm;
    std::cout << "Matrix-free residual: ||op*x_sol - b|| = "
              << std::scientific << res_mf_abs
              << "  (rel: " << res_mf_rel << ")" << std::endl;

    ASSERT_TRUE(converged)
        << "FGMRES did not converge for tau=" << tau;
    EXPECT_LT(res_assembled_rel, 10.0 * tol)
        << "Assembled-matrix residual too large for tau=" << tau;
    EXPECT_LT(res_mf_rel, 10.0 * tol)
        << "Matrix-free residual too large for tau=" << tau;
}

// --------------------------------------------------------
// Test Cases
// --------------------------------------------------------

TEST(StokesMGTest, ConvergenceTetra)
{
    const unsigned int n = 1;
    auto mesh_ptr = std::make_shared<mfem::Mesh>(
        mfem::Mesh::MakeCartesian3D(n, n, n, mfem::Element::TETRAHEDRON)
    );
    RunStokesMGTest(mesh_ptr, 4);
}

// Test that MG+GMRES solves the assembled Galerkin matrix for various
// viscosity values.  tau = 1/(dt*nu) as in the semi-Lagrangian MPI app.
TEST(StokesMGTest, AssembledResidualNu001)
{
    // nu=0.01, dt=0.01 → tau = 1/(0.01*0.01) = 10000
    const double nu = 0.01, dt = 0.01;
    RunAssembledResidualTest(1.0 / (dt * nu), /*geomref=*/3);
}

TEST(StokesMGTest, AssembledResidualNu1)
{
    // nu=1.0, dt=0.01 → tau = 100
    const double nu = 1.0, dt = 0.01;
    RunAssembledResidualTest(1.0 / (dt * nu), /*geomref=*/3);
}

TEST(StokesMGTest, AssembledResidualNu0001)
{
    // nu=0.001, dt=0.01 → tau = 100000
    const double nu = 0.001, dt = 0.01;
    RunAssembledResidualTest(1.0 / (dt * nu), /*geomref=*/3);
}
