#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <boost/program_options.hpp>

#include "mfem.hpp"
#include <mpi.h>
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

    // ---- Now start MPI ----
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank;
    MPI_Comm_rank(comm, &rank);

    // Optionally only rank 0 prints what config it’s using
    if (rank == 0) {
        std::cout << "Using config file: " << config_path << std::endl;
    }

    // ---- Use the parsed config path ----
    SimulationConfig config(config_path);
    config.InitializeLibrary(rank, comm);

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

   VectorFunctionCoefficient force_coef(3, force_data);

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
   FiniteElementCollection *fec_DG = new L2_FECollection(order - 1, dim);
   FiniteElementCollection *fec_ND = new ND_FECollection(order, dim);
   FiniteElementCollection *fec_RT = new RT_FECollection(order - 1, dim);
   FiniteElementCollection *fec_CG = new H1_FECollection(order, dim);

   FiniteElementSpace DG(&mesh, fec_DG);
   FiniteElementSpace ND(&mesh, fec_ND);
   FiniteElementSpace RT(&mesh, fec_RT);
   FiniteElementSpace CG(&mesh, fec_CG);

   // ------------------------------------------------------------------
   // 2. Unknowns and gridfunctions (PARALLEL)
   // ------------------------------------------------------------------
   GridFunction u(&ND);
   GridFunction z(&RT);
   GridFunction p(&CG);
   GridFunction v(&RT);
   GridFunction w(&ND);
   GridFunction q(&DG);

   // Initial data from user-provided functions
   {
      VectorFunctionCoefficient u0(dim, initial_data_u);
      VectorFunctionCoefficient w0(dim, initial_data_w);
      u.ProjectCoefficient(u0);
      v.ProjectCoefficient(u0);
      w.ProjectCoefficient(w0);
      z.ProjectCoefficient(w0);
      p = 0.;
      q = 0.;
   }

   // ------------------------------------------------------------------
   // 3. System sizes and block layout
   //    NOTE: Sizes are local DOFs per rank.
   // ------------------------------------------------------------------
   int size_1 = u.Size() + z.Size() + p.Size();
   int size_2 = v.Size() + w.Size() + q.Size();

   Vector x(size_1), y(size_2);
   x = 0.0;
   y = 0.0;

   Array<int> u_dofs(u.Size()), z_dofs(z.Size()), p_dofs(p.Size());
   Array<int> v_dofs(v.Size()), w_dofs(w.Size()), q_dofs(q.Size());
   std::iota(u_dofs.begin(), u_dofs.end(), 0);
   std::iota(z_dofs.begin(), z_dofs.end(), u.Size());
   std::iota(p_dofs.begin(), p_dofs.end(), u.Size() + z.Size());
   std::iota(v_dofs.begin(), v_dofs.end(), 0);
   std::iota(w_dofs.begin(), w_dofs.end(), v.Size());
   std::iota(q_dofs.begin(), q_dofs.end(), v.Size() + w.Size());

   // Block offsets for A1 (u,z,p) and A2 (v,w,q)
   Array<int> offsets_1(4), offsets_2(4);
   offsets_1[0] = 0;
   offsets_1[1] = u.Size();
   offsets_1[2] = z.Size();
   offsets_1[3] = p.Size();
   offsets_1.PartialSum();

   offsets_2[0] = 0;
   offsets_2[1] = v.Size();
   offsets_2[2] = w.Size();
   offsets_2[3] = q.Size();
   offsets_2.PartialSum();

   BlockOperator A1(offsets_1);
   BlockOperator A2(offsets_2);
   BlockDiagonalPreconditioner pre1(offsets_1);
   BlockDiagonalPreconditioner pre2(offsets_2);

   // ------------------------------------------------------------------
   // 4. Time-independent bilinear/mixed forms (matrix-free / PA, PARALLEL)
   // ------------------------------------------------------------------
   ConstantCoefficient one_coeff(1.0);

   // Mass matrices M (on ND) and N (on RT)
   BilinearForm blf_M(&ND);
   blf_M.AddDomainIntegrator(new VectorFEMassIntegrator(one_coeff));
   //blf_M.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_M.Assemble();
   Operator &M_op = blf_M;
   ScaledOperator M_dt_op(&M_op, 1.0 / dt);
   ScaledOperator M_n_op(&M_op, -1.0);

   BilinearForm blf_N(&RT);
   blf_N.AddDomainIntegrator(new VectorFEMassIntegrator(one_coeff));
   //blf_N.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_N.Assemble();
   Operator &N_op = blf_N;
   ScaledOperator N_dt_op(&N_op, 1.0 / dt);

   // C : ND -> RT (curl)
   MixedBilinearForm blf_C(&ND, &RT);
   blf_C.AddDomainIntegrator(new MixedVectorCurlIntegrator());
   //blf_C.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_C.Assemble();
   Operator &C_op = blf_C;
   ScaledOperator C_negative_op(&C_op, -1.);
   ScaledOperator C_Re_op(&C_op, viscosity / 2.0);
   TransposeOperator CT_op(C_op);
   ScaledOperator neg_CT_op(&CT_op, -1.);
   ScaledOperator CT_Re_op(&CT_op, viscosity / 2.0);

   // D : RT -> DG (div)
   MixedBilinearForm blf_D(&RT, &DG);
   blf_D.AddDomainIntegrator(new MixedScalarDivergenceIntegrator());
   // blf_D.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_D.Assemble();
   Operator &D_op = blf_D;
   TransposeOperator DT_op(D_op);
   ScaledOperator DT_n_op(&DT_op, -1.0);

   // G : CG -> ND (grad)
   MixedBilinearForm blf_G(&CG, &ND);
   blf_G.AddDomainIntegrator(new MixedVectorGradientIntegrator());
   // blf_G.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_G.Assemble();
   Operator &G_op = blf_G;
   TransposeOperator GT_op(G_op);

   BilinearForm blf_H1_pre(&CG);
   blf_H1_pre.AddDomainIntegrator(new MassIntegrator());
   blf_H1_pre.AddDomainIntegrator(new DiffusionIntegrator());
   // blf_H1_pre.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_H1_pre.Assemble();
   blf_H1_pre.Finalize();

   BilinearForm blf_Hdiv_pre(&RT);
   blf_Hdiv_pre.AddDomainIntegrator(new VectorFEMassIntegrator(one_coeff));
   //blf_Hdiv_pre.AddDomainIntegrator(new DivDivIntegrator());
   // blf_Hdiv_pre.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_Hdiv_pre.Assemble();
   blf_Hdiv_pre.Finalize();

   BilinearForm blf_Hcurl_pre(&ND);
   blf_Hcurl_pre.AddDomainIntegrator(new VectorFEMassIntegrator(one_coeff));
   //blf_Hcurl_pre.AddDomainIntegrator(new CurlCurlIntegrator());
   // blf_Hcurl_pre.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_Hcurl_pre.Assemble();
   blf_Hcurl_pre.Finalize();

   BilinearForm blf_L2_pre(&DG);
   blf_L2_pre.AddDomainIntegrator(new MassIntegrator());
   blf_L2_pre.Assemble();
   blf_L2_pre.Finalize();

   // auto H1_pre = std::make_unique<CGSolver>(comm);
   //H1_pre->SetMaxIter(100);
   //H1_pre->SetAbsTol(1e-10);
   //H1_pre->SetOperator(blf_H1_pre);
   mfem::DSmoother H1_pre(blf_H1_pre.SpMat());

   //auto Hcurl_pre = std::make_unique<CGSolver>(comm);
   //Hcurl_pre->SetMaxIter(100);   // Hcurl_pre->SetAbsTol(1e-10);
   //Hcurl_pre->SetOperator(blf_Hcurl_pre);
   mfem::DSmoother Hcurl_pre(blf_Hcurl_pre.SpMat());

   //auto Hdiv_pre = std::make_unique<CGSolver>(comm);
   //Hdiv_pre->SetMaxIter(100);
   //Hdiv_pre->SetAbsTol(1e-10);
   //Hdiv_pre->SetOperator(blf_Hdiv_pre);
   mfem::DSmoother Hdiv_pre(blf_Hdiv_pre.SpMat());

   //   auto L2_pre = std::make_unique<CGSolver>(comm);
   //L2_pre->SetMaxIter(100);
   //L2_pre->SetAbsTol(1e-10);
   //L2_pre->SetOperator(blf_L2_pre);
   mfem::DSmoother L2_pre(blf_L2_pre.SpMat());

   pre1.SetDiagonalBlock(0, &Hcurl_pre);
   pre1.SetDiagonalBlock(1, &Hdiv_pre);
   pre1.SetDiagonalBlock(2, &H1_pre);

   pre2.SetDiagonalBlock(0, &Hdiv_pre);
   pre2.SetDiagonalBlock(1, &Hcurl_pre);
   pre2.SetDiagonalBlock(2, &L2_pre);

   LinearForm f1_lf(&ND);
   f1_lf.AddDomainIntegrator(new VectorFEDomainLFIntegrator(force_coef));

   LinearForm f2_lf(&RT);
   f2_lf.AddDomainIntegrator(new VectorFEDomainLFIntegrator(force_coef));


   // ------------------------------------------------------------------
   // 5. Time-stepping
   // ------------------------------------------------------------------
   double t = 0.0;
   int cycle = 0;

   Vector zero_vec(dim);
   zero_vec = 0.0;
   VectorFunctionCoefficient f_coeff(dim, [](const Vector &x, Vector &val)
                                     { val = 0.0; });

   // Rhs containers
   Vector b1(size_1), b1sub(u.Size());
   Vector b2(size_2), b2sub(v.Size());

   // Solver selection
   unique_ptr<IterativeSolver> solver;
   if (solver_type == "MINRES")
   {
      auto minres = make_unique<MINRESSolver>(comm);
      minres->SetAbsTol(tol);
      minres->SetRelTol(0.);
      minres->SetMaxIter(10000);
      minres->SetPreconditioner(pre1);
      minres->SetPrintLevel(printlevel);
      solver = std::move(minres);
   }
   else if (solver_type == "GMRES")
   {
      auto minres = make_unique<GMRESSolver>(comm);
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
      auto cg = make_unique<CGSolver>(comm);
      cg->SetAbsTol(tol);
      cg->SetRelTol(0.);
      cg->SetMaxIter(1000);
      cg->SetPrintLevel(printlevel);
      cg->SetPreconditioner(pre1);
      solver = std::move(cg);
   }

   // Pre-allocate temporaries
   Vector tmp_u(u.Size()), tmp_v(v.Size());

   // Helper coefficients that depend on w or z
   VectorGridFunctionCoefficient w_gfcoeff(&w);
   VectorGridFunctionCoefficient z_gfcoeff(&z);
   ConstantCoefficient two_over_dt(2.0 / dt);
   ConstantCoefficient two(2.);

   int num_it_A1, num_it_A2;
   num_it_A1 = num_it_A2 = 0;
   // Euler step: build MR_eul operator (2/dt M + cross(w,·)) in PA
   if(true)
   {
      MixedBilinearForm blf_MR_eul(&ND, &ND);
      blf_MR_eul.AddDomainIntegrator(new VectorFEMassIntegrator(two_over_dt));
      blf_MR_eul.AddDomainIntegrator(new MixedCrossProductIntegrator(w_gfcoeff));
      // blf_MR_eul.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      blf_MR_eul.Assemble();
      Operator &MR_eul_op = blf_MR_eul;

      // CT_eul = 2 * CT_Re (operator scaling)
      ScaledOperator CT_eul_op(&CT_Re_op, 2.0);
      ScaledOperator C_eul_op(&C_op, -1.);

      // Build A1 for Euler step
      A1.SetBlock(0, 0, &MR_eul_op);
      A1.SetBlock(0, 1, &CT_eul_op);
      A1.SetBlock(0, 2, &G_op);
      A1.SetBlock(1, 0, &C_eul_op);
      A1.SetBlock(1, 1, &N_op);
      A1.SetBlock(2, 0, &GT_op);

      // Build rhs b1:  b1 = 2*M_dt*u + f1  (simplified)
      force_coef.SetTime(0.5*dt);
      f1_lf.Assemble();
      b1 = 0.0;
      b1sub = 0.0;
      M_dt_op.Mult(u, tmp_u);
      b1sub.Add(2.0, tmp_u);

      b1.AddSubVector(b1sub, 0);
      b1.AddSubVector(f1_lf, 0);

      std::cout << "start first solver\n";
      solver->SetPreconditioner(pre1);
      solver->SetOperator(A1);
      solver->Mult(b1, x);
      num_it_A1 = solver->GetNumIterations();
      // std::abort();

      // extract u,z,p
      x.GetSubVector(u_dofs, u);
      x.GetSubVector(z_dofs, z);
      x.GetSubVector(p_dofs, p);
   }


   // --- CSV output (only rank 0) ---
   EnergyCSVLogger *csv_logger_ptr = nullptr;
   if (rank == 0)
   {
      csv_logger_ptr = new EnergyCSVLogger(config, M_op, N_op, u, v, w, z, num_it_A1, num_it_A2);
      if (csv_logger_ptr->IsOpen())
      {
         csv_logger_ptr->WriteRow(0, 0., 0.);
      }
   }

   // ParaView data collection (parallel)
   mfem::ParaViewDataCollection vtk_dc(
       "./data/visualisation/paraview/" + output_file,
       &mesh);

   if (visualisation > 0)
   {
      vtk_dc.RegisterField("u1", &u);
      vtk_dc.RegisterField("u2", &v);
      vtk_dc.RegisterField("w1", &w);
      vtk_dc.RegisterField("w2", &z);
      vtk_dc.RegisterField("p0", &p);
      vtk_dc.RegisterField("p3", &q);
      vtk_dc.SetCycle(0);
      vtk_dc.SetTime(0.0);
      vtk_dc.Save();
   }





   // ------------------------------------------------------------------
   // 6. Time loop (matrix-free, PARALLEL)
   // ------------------------------------------------------------------
   while (t < T - 0.5 * dt)
   {
      t += dt;
      cycle++;

      force_coef.SetTime(t);
      f2_lf.Assemble();

         mfem::VectorFunctionCoefficient w_exact_coeff(3,
    [t](const mfem::Vector &x, mfem::Vector &out)
    { out.SetSize(3); out[0] = -2*M_PI*(t+1)*cos(2*M_PI*x[2]); out[1] = -2*M_PI*(1-t)*cos(2*M_PI*x[0]); out[2] =  2*M_PI*(2-t)*sin(2*M_PI*x[1]); });

      //z.ProjectCoefficient(w_exact_coeff);
      // === DUAL FIELD: build R2 and NR as PA operators ===
      MixedBilinearForm blf_R2(&RT, &RT);
      blf_R2.AddDomainIntegrator(new MixedCrossProductIntegrator(z_gfcoeff));
      blf_R2.Assemble();
      Operator &R2_op = blf_R2;
      ScaledOperator R2_half_op(&R2_op, 0.5);

      MixedBilinearForm blf_NR(&RT, &RT);
      blf_NR.AddDomainIntegrator(new VectorFEMassIntegrator(two_over_dt));
      blf_NR.AddDomainIntegrator(new MixedCrossProductIntegrator(z_gfcoeff));
      blf_NR.Assemble();
      Operator &NR_op = blf_NR;
      ScaledOperator NR_half_op(&NR_op, 0.5*dt);

      // A2 blocks:
      A2.SetBlock(0, 0, &NR_half_op);
      A2.SetBlock(0, 1, &C_Re_op);
      A2.SetBlock(0, 2, &DT_n_op);
      A2.SetBlock(1, 0, &neg_CT_op);
      A2.SetBlock(1, 1, &M_op);
      A2.SetBlock(2, 0, &D_op);

      // rhs b2 = N_dt*v - R2*v - C_Re*w + f2
      b2 = 0.0;
      b2sub = 0.0;

      N_dt_op.Mult(v, tmp_v);
      b2sub += tmp_v;

      R2_half_op.Mult(v, tmp_v);
      b2sub.Add(-1.0, tmp_v);

      C_Re_op.Mult(w, tmp_v);
      b2sub.Add(-1.0, tmp_v);
      b2sub.Add(1.,f2_lf);
      b2sub *= dt;

      b2.AddSubVector(b2sub, 0);
      //b2.AddSubVector(f2_lf, 0);

      solver->SetPreconditioner(pre2);
      solver->SetOperator(A2);
      solver->Mult(b2, y);
      num_it_A2 = solver->GetNumIterations();

      y.GetSubVector(v_dofs, v);
      y.GetSubVector(w_dofs, w);
      y.GetSubVector(q_dofs, q);
      force_coef.SetTime(t+0.5*dt);
      f1_lf.Assemble();
      //w.ProjectCoefficient(w_exact_coeff);
      // === PRIMAL FIELD: build R1 and MR as PA operators ===
      MixedBilinearForm blf_R1(&ND, &ND);
      blf_R1.AddDomainIntegrator(new MixedCrossProductIntegrator(w_gfcoeff));
      blf_R1.Assemble();
      Operator &R1_op = blf_R1;
      ScaledOperator R1_half_op(&R1_op, 0.5);

      MixedBilinearForm blf_MR(&ND, &ND);
      blf_MR.AddDomainIntegrator(new VectorFEMassIntegrator(two_over_dt));
      blf_MR.AddDomainIntegrator(new MixedCrossProductIntegrator(w_gfcoeff));
      blf_MR.Assemble();
      Operator &MR_op = blf_MR;
      ScaledOperator MR_half_op(&MR_op, 0.5);

      // A1 blocks:
      A1.SetBlock(0, 0, &MR_half_op);
      A1.SetBlock(0, 1, &CT_Re_op);
      A1.SetBlock(0, 2, &G_op);
      A1.SetBlock(1, 0, &C_negative_op);
      A1.SetBlock(1, 1, &N_op);
      A1.SetBlock(2, 0, &GT_op);

      // rhs b1 = M_dt*u - R1*u - CT_Re*z + f1 + boundary terms...
      b1 = 0.0;
      b1sub = 0.0;

      M_dt_op.Mult(u, tmp_u);
      b1sub += tmp_u;

      tmp_u = 0.;
      R1_half_op.Mult(u, tmp_u);
      b1sub.Add(-1.0, tmp_u);

      tmp_u = 0.;
      CT_Re_op.Mult(z, tmp_u);
      b1sub.Add(-1.0, tmp_u);

      b1.AddSubVector(b1sub, 0);
      b1.AddSubVector(f1_lf, 0);

      solver->SetPreconditioner(pre1);
      solver->SetOperator(A1);
      solver->Mult(b1, x);
      num_it_A1 = solver->GetNumIterations();

      x.GetSubVector(u_dofs, u);
      x.GetSubVector(z_dofs, z);
      x.GetSubVector(p_dofs, p);

      // Append norms to CSV (rank 0 only)
      if (rank == 0 && csv_logger_ptr && csv_logger_ptr->IsOpen())
      {
         csv_logger_ptr->WriteRow(cycle, t, t - 0.5 * dt);
      }

      // Save data to ParaView (all ranks)
      if (visualisation > 0)
      {
         vtk_dc.SetCycle(cycle);
         vtk_dc.SetTime(dt * cycle);
         vtk_dc.Save();
      }

      if (rank == 0)
      {
         cout << "Step " << cycle << ", t=" << t << " done.\n";
      }
   }

   // ------------------------------------------------------------------
   // 7. Cleanup
   // ------------------------------------------------------------------
   if (csv_logger_ptr)
   {
      delete csv_logger_ptr;
   }

   delete fec_DG;
   delete fec_ND;
   delete fec_RT;
   delete fec_CG;

   MPI_Finalize();
   return 0;
}
