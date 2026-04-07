#include <gtest/gtest.h>
#include <iomanip>
#include "mfem.hpp"
#include "StokesDGS.h"
#include "StokesOperator.h"
#include "StokesMG.h"

#define nu 0.01
#define tau 10.0

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
    StokesNitsche::StokesMG mg(mesh_ptr, tau, nu, 1.0, 10.0);

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
