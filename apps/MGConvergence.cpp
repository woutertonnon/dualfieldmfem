#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <complex>
#include <memory>
#include <cmath>
#include <random>
#include <iomanip>

#include <boost/program_options.hpp>
#include <Eigen/Dense>

#include "mfem.hpp"
#include "StokesMG.h"
#include "StokesOperator.h"
#include "SpectraErrorOp.h"

namespace po = boost::program_options;

// Solves A x = b with random b using GMRES preconditioned by P
int runGMRES(const mfem::Operator& A,
             mfem::Solver& P,
             const double tol,
             const int restart = 100,
             const int max_iter = 1000)
{
    mfem::FGMRESSolver gmres;
    gmres.SetOperator(A);
    gmres.SetPreconditioner(P);
    gmres.SetAbsTol(1e-12);
    gmres.SetRelTol(tol);
    gmres.SetMaxIter(max_iter);
    gmres.SetPrintLevel(0);
    gmres.SetKDim(restart);

    int num_rows = A.NumRows();
    mfem::Vector x(num_rows), b(num_rows);
    mfem::Vector x_exact(num_rows);

    x_exact.Randomize();
    A.Mult(x_exact, b);

    x = 0.0;
    gmres.Mult(b, x);

    if (!gmres.GetConverged())
    {
	throw std::runtime_error("GMRES did not converge!");
        return max_iter;
    }

    return gmres.GetNumIterations();
}

int main(int argc, char* argv[])
{
#ifdef MFEM_USE_OPENMP
    mfem::Device device("omp");
    // device.Print(std::cout);
#endif

    std::string mesh_file, output_file, cycle_str;
    int max_refinements, nev, n_gmres, pre_smooth, post_smooth;
    double gmres_tol, eval_tol, penalty, tau;
    bool verbose, save_eigenvectors_vtu;

    // Parse command line options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("mesh,m", po::value<std::string>(&mesh_file)->required(), "mesh filename")
        ("refinements,r", po::value<int>(&max_refinements)->required(), "number of refinements")
        ("output,o", po::value<std::string>(&output_file)->default_value("out.csv"), "output csv filename")
        ("tau,t", po::value<double>(&tau)->default_value(0.0), "time-stepping parameter (tau)")
        ("penalty,p", po::value<double>(&penalty)->default_value(10.0), "Nitsche penalty parameter")
        ("nev,n", po::value<int>(&nev)->default_value(1), "number of eigenvalues (0 to skip)")
        ("gmres,g", po::value<int>(&n_gmres)->default_value(1), "number of GMRES runs")
        ("pre_smooth", po::value<int>(&pre_smooth)->default_value(1), "number of pre-smoothing steps")
        ("post_smooth", po::value<int>(&post_smooth)->default_value(1), "number of post-smoothing steps")
        ("verbose,v", po::bool_switch(&verbose)->default_value(false), "enable verbose output")
        ("save_eigenvectors_vtu", po::bool_switch(&save_eigenvectors_vtu)->default_value(false),
         "save error-operator eigenvectors to VTU (one dataset per eigenvector)")
        ("gmres_tol", po::value<double>(&gmres_tol)->default_value(1e-6), "GMRES tolerance")
        ("eval_tol", po::value<double>(&eval_tol)->default_value(1e-4), "Eigenvalue solver tolerance")
        ("cycle,c", po::value<std::string>(&cycle_str)->default_value("V"), "cycle type (V or W)");

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc << "\n";
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // Parameters
    const double theta = 1.0;
    const double factor = 1.0;

    auto mesh_ptr = std::make_shared<mfem::Mesh>(mesh_file.c_str(), 1, 1);

    // Initialize Multigrid with tau and penalty from command line
    StokesNitsche::StokesMG mg(
      mesh_ptr, tau, theta, penalty, factor,
      StokesNitsche::MassLumping::RowSum,
      StokesNitsche::SmootherType::GaussSeidelForw
    );

    if (cycle_str == "W" || cycle_str == "w")
        mg.setCycleType(StokesNitsche::MGCycleType::WCycle);
    else
        mg.setCycleType(StokesNitsche::MGCycleType::VCycle);

    mg.setSmoothIterations(pre_smooth, post_smooth);

    // Prepare Output
    std::ofstream out(output_file);
    if (!out)
    {
        std::cerr << "Cannot open " << output_file << "\n";
        return 1;
    }

    out << "Refinements,DOFs,AvgGMRES";
    for (int i = 0; i < nev; ++i)
        out << ",AbsEval" << i;
    out << "\n";

    if (verbose)
    {
        std::cout << std::string(75, '=') << "\n";
        std::cout << "Mesh: " << mesh_file << "\n";
        std::cout << "Tau: " << tau << "\n";
        std::cout << "Penalty: " << penalty << "\n";
        std::cout << "Pre-smoothing steps: " << pre_smooth << "\n";
        std::cout << "Post-smoothing steps: " << post_smooth << "\n";
        std::cout << "Refinements: " << max_refinements << "\n";
        std::cout << std::string(75, '=') << "\n";
    }

    // Refinement Loop
    for (int r = 1; r <= max_refinements; ++r)
    {
        mg.addRefinement();

        const auto& finest_op = mg.getFinestOperator();
        const int dofs = finest_op.NumRows();

        if (verbose)
        {
            std::cout << "Refinement Level " << r << " (" << dofs << " DOFs)\n";
            std::cout << std::string(75, '-') << "\n";
        }

        // 1. Compute Eigenvalues (DEC Mode)
        finest_op.setOperatorMode(StokesNitsche::OperatorMode::DEC);
        mg.setOperatorMode(StokesNitsche::OperatorMode::DEC);
        mg.setIterativeMode(false);

        Eigen::VectorXcd evals;
        if (nev > 0)
        {
            const std::string eigvec_prefix =
                "out/paraview/mg_error_operator_ref_" + std::to_string(r);
            evals = computeErrorOperatorEigenvalues(
                finest_op,
                mg,
                nev,
                eval_tol,
                verbose,
                nullptr,
                save_eigenvectors_vtu,
                &finest_op,
                eigvec_prefix);
        }

        // 2. Run GMRES (Galerkin Mode)
        finest_op.setOperatorMode(StokesNitsche::OperatorMode::Galerkin);
        mg.setOperatorMode(StokesNitsche::OperatorMode::Galerkin);
        mg.setIterativeMode(false);

        double avg_gmres = 0.0;
        if (n_gmres > 0)
        {
            long total = 0;
            for (int i = 0; i < n_gmres; ++i)
                total += runGMRES(finest_op, mg, gmres_tol);
            avg_gmres = static_cast<double>(total) / n_gmres;
        }

        // 3. Write to CSV
        out << r << "," << dofs << "," << avg_gmres;
        for (int i = 0; i < nev; ++i)
        {
            if (i < evals.size())
                out << "," << std::abs(evals[i]);
            else
                out << ",NaN";
        }
        out << "\n";
        out.flush();

        if (verbose)
        {
            if (n_gmres > 0)
                std::cout << "GMRES: Avg Iterations: " << avg_gmres << "\n";

            std::cout << std::string(75, '=') << "\n";
        }
    }

    if (verbose)
        std::cout << "Results saved to " << output_file << "\n";

    return 0;
}
