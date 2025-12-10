#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <boost/program_options.hpp>

#include "mfem.hpp"
#include "BoundaryOperators.h"
#include "io.h" // SimulationConfig, EnergyCSVLogger

using namespace mfem;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char *argv[])
{
    // ---- Parse command-line options with Boost BEFORE MPI_Init (recommended) ----
    std::string config_path;

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("config,c",
                po::value<std::string>(&config_path)
                    ->default_value("../data/config/example2.json"),
                "path to JSON configuration file");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            // Just print from rank 0 later, but we don't know rank yet.
            // For now, print unconditionally (or move this after MPI_Init).
            std::cout << desc << "\n";
            return 0;
        }
    } catch (const std::exception &e) {
        std::cerr << "Error parsing command line: " << e.what() << "\n";
        return 1;
    }
    int rank = 0;

    // Optionally only rank 0 prints what config it’s using
    if (rank == 0) {
        std::cout << "Using config file: " << config_path << std::endl;
    }

    // ---- Use the parsed config path ----
    SimulationConfig config(config_path);
    config.InitializeLibrary(rank);

   // ------------------------------------------------------------------
   // 0. Configuration
   // ------------------------------------------------------------------
   double dt = config.get_dt();
   double T = config.get_T();
   double viscosity = config.get_viscosity();
   int refinements = config.get_refinements();
   int order = config.get_order();
   int visualisation = config.get_visualisation();
   int printlevel = config.get_printlevel();
   double tol = config.get_tol();
   bool has_exact_u = config.has_exact_u();
   std::string mesh_string = config.get_mesh();
   std::string output_file = config.get_outputfile();
   std::string solver_type = config.get_solver();

   std::function<void(const mfem::Vector &, double, mfem::Vector &)> boundary_data_u =
       std::bind(&SimulationConfig::boundary_data_u, &config,
                 std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
   std::function<void(const mfem::Vector &, double, mfem::Vector &)> exact_data_u =
       std::bind(&SimulationConfig::exact_data_u, &config,
                 std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
   std::function<void(const mfem::Vector &, double, mfem::Vector &)> force_data =
       std::bind(&SimulationConfig::force_data, &config,
                 std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
   std::function<void(const mfem::Vector &, mfem::Vector &)> initial_data_u =
       std::bind(&SimulationConfig::initial_data_u, &config,
                 std::placeholders::_1, std::placeholders::_2);
   std::function<void(const mfem::Vector &, mfem::Vector &)> initial_data_w =
       std::bind(&SimulationConfig::initial_data_w, &config,
                 std::placeholders::_1, std::placeholders::_2);


   // ------------------------------------------------------------------
   // 1. Mesh and FE spaces (PARALLEL)
   // ------------------------------------------------------------------
   Mesh mesh(mesh_string.c_str(), 1, 1);
   for (int l = 0; l < refinements; l++)
   {
      mesh.UniformRefinement();
   }
   int dim = mesh.Dimension();

   // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
   FiniteElementCollection *fec_ND = new ND_FECollection(order, dim);
   FiniteElementCollection *fec_CG = new H1_FECollection(order, dim);

   FiniteElementSpace ND(&mesh, fec_ND);
   FiniteElementSpace CG(&mesh, fec_CG);

   // ------------------------------------------------------------------
   // 2. Unknowns and gridfunctions (PARALLEL)
   // ------------------------------------------------------------------
   GridFunction u(&ND);
   GridFunction p(&CG);

   // Initial data from user-provided functions
   {
      VectorFunctionCoefficient u0(dim, initial_data_u);
      u.ProjectCoefficient(u0);
      p = 0.;
   }

   // ------------------------------------------------------------------
   // 3. System sizes and block layout
   //    NOTE: Sizes are local DOFs per rank.
   // ------------------------------------------------------------------
   int size_1 = u.Size() + p.Size();

   Vector x(size_1);
   x = 0.0;

   Array<int> u_dofs(u.Size()), p_dofs(p.Size());
   std::iota(u_dofs.begin(), u_dofs.end(), 0);
   std::iota(p_dofs.begin(), p_dofs.end(), u.Size() );

   // Block offsets for A1 (u,z,p) and A2 (v,w,q)
   Array<int> offsets_1(3);
   offsets_1[0] = 0;
   offsets_1[1] = u.Size();
   offsets_1[2] = p.Size();
   offsets_1.PartialSum();

   BlockOperator A1(offsets_1);
   BlockDiagonalPreconditioner pre1(offsets_1);

   // ------------------------------------------------------------------
   // 4. Time-independent bilinear/mixed forms (matrix-free / PA, PARALLEL)
   // ------------------------------------------------------------------
   ConstantCoefficient one_coeff(1.0);

   // Mass matrices M (on ND) and N (on RT)
   ConstantCoefficient mass_coeff(1./dt), diff_coef(viscosity);
   BilinearForm blf_A(&ND);
   blf_A.AddDomainIntegrator(new VectorFEMassIntegrator(mass_coeff));
   blf_A.AddDomainIntegrator(new CurlCurlIntegrator(diff_coef));
   blf_A.Assemble();
   Operator &A_op = blf_A;

   // G : CG -> ND (grad)
   MixedBilinearForm blf_B(&CG, &ND);
   blf_B.AddDomainIntegrator(new MixedVectorGradientIntegrator());
   blf_B.Assemble();
   Operator &B_op = blf_B;
   TransposeOperator BT_op(B_op);

   BilinearForm blf_H1_pre(&CG);
   blf_H1_pre.AddDomainIntegrator(new MassIntegrator());
   //blf_H1_pre.AddDomainIntegrator(new DiffusionIntegrator());
   blf_H1_pre.Assemble();
   blf_H1_pre.Finalize();

   BilinearForm blf_Hcurl_pre(&ND);
   blf_Hcurl_pre.AddDomainIntegrator(new VectorFEMassIntegrator(one_coeff));
   //blf_Hcurl_pre.AddDomainIntegrator(new CurlCurlIntegrator());
   blf_Hcurl_pre.Assemble();
   blf_Hcurl_pre.Finalize();

   auto H1_pre_pre = std::make_unique<mfem::OperatorJacobiSmoother>(blf_H1_pre, mfem::Array<int>{});
   auto H1_pre = std::make_unique<CGSolver>();
   H1_pre->SetMaxIter(100);
   H1_pre->SetAbsTol(1e-10);
   H1_pre->SetPreconditioner(*H1_pre_pre);
   H1_pre->SetOperator(blf_H1_pre);
   //mfem::DSmoother H1_pre(blf_H1_pre.SpMat());

   auto Hcurl_pre_pre = std::make_unique<mfem::OperatorJacobiSmoother>(blf_Hcurl_pre, mfem::Array<int>{});
   auto Hcurl_pre = std::make_unique<CGSolver>();
   Hcurl_pre->SetMaxIter(100);   
   Hcurl_pre->SetAbsTol(1e-10);
   Hcurl_pre->SetPreconditioner(*Hcurl_pre_pre);
   Hcurl_pre->SetOperator(blf_Hcurl_pre);
   //mfem::DSmoother Hcurl_pre(blf_Hcurl_pre.SpMat());
   
   pre1.SetDiagonalBlock(0, Hcurl_pre.get());
   pre1.SetDiagonalBlock(1, H1_pre.get());



   // Rhs containers
   Vector b1(size_1);
   // Solver selection
   unique_ptr<IterativeSolver> solver;
   if (solver_type == "MINRES")
   {
      auto minres = make_unique<MINRESSolver>();
      minres->SetAbsTol(tol);
      minres->SetRelTol(0.);
      minres->SetMaxIter(10000);
      minres->SetPreconditioner(pre1);
      minres->SetPrintLevel(printlevel);
      solver = std::move(minres);
   }
   else if (solver_type == "GMRES")
   {
      auto minres = make_unique<GMRESSolver>();
      minres->SetAbsTol(tol);
      minres->SetKDim(300);
      minres->SetRelTol(0.);
      minres->SetMaxIter(10000);
      minres->SetPreconditioner(pre1);
      minres->SetPrintLevel(printlevel);
      solver = std::move(minres);
   }
   else // CG
   {
      auto cg = make_unique<CGSolver>();
      cg->SetAbsTol(tol);
      cg->SetRelTol(0.);
      cg->SetMaxIter(1000);
      cg->SetPrintLevel(printlevel);
      cg->SetPreconditioner(pre1);
      solver = std::move(cg);
   }

   int num_it_A1, num_it_A2;
   num_it_A1 = num_it_A2 = 0;


   // A1 blocks:
   A1.SetBlock(0, 0, &A_op);
   A1.SetBlock(0, 1, &BT_op);
   A1.SetBlock(1, 0, &B_op);

   // rhs b1 = M_dt*u - R1*u - CT_Re*z + f1 + boundary terms...
   LinearForm f1_lf(&ND);
   VectorFunctionCoefficient force_coef(mesh.Dimension(), force_data);
   f1_lf.AddDomainIntegrator(new VectorFEDomainLFIntegrator(force_coef));
   f1_lf.Assemble();
   b1.AddSubVector(f1_lf, 0);

   NitscheStokesCSVLogger csv(config,u,num_it_A1);

   solver->SetPreconditioner(pre1);
   solver->SetOperator(A1);
   solver->Mult(b1, x);
   num_it_A1 = solver->GetNumIterations();

   x.GetSubVector(u_dofs, u);
   x.GetSubVector(p_dofs, p);

   csv.WriteRow();

   mfem::VectorFunctionCoefficient exact_data_u_coef(mesh.Dimension(),exact_data_u);
   delete fec_ND;
   delete fec_CG;

   //MPI_Finalize();
   return 0;
}
