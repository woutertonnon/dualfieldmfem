#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <iomanip>

#include "mfem.hpp"
#include <mpi.h>
#include "io.h"

// If you have the SimulationConfig class from earlier, include it here.
// #include "SimulationConfig.hpp"

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   SimulationConfig config("../data/config/example2.json");
   config.InitializeLibrary(rank, MPI_COMM_WORLD);

   // ------------------------------------------------------------------
   // 0. Configuration (replace with SimulationConfig if you have it)
   // ------------------------------------------------------------------
   // For illustration I'll hard-code a few things.
       // Parse configuration parameters
    double dt = config.get_dt(); // Time step
    double T = config.get_T();   // Total time
    double viscosity = config.get_viscosity();
    int refinements = config.get_refinements();     // Number of mesh refinements
    int order = config.get_order();                 // Finite element order
    int visualisation = config.get_visualisation(); // Visualisation level
    int printlevel = config.get_printlevel();
    double tol = config.get_tol();
    bool has_exact_u = config.has_exact_u();
    std::string mesh_string = config.get_mesh();       // Path to mesh file
    std::string output_file = config.get_outputfile(); // Output file for results
    std::string solver_type = config.get_solver();

    std::function<void(const mfem::Vector &, double, mfem::Vector &)> boundary_data_u =
        std::bind(&SimulationConfig::boundary_data_u, &config, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    std::function<void(const mfem::Vector &, double, mfem::Vector &)> exact_data_u =
        std::bind(&SimulationConfig::exact_data_u, &config, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    std::function<void(const mfem::Vector &, double, mfem::Vector &)> force_data =
        std::bind(&SimulationConfig::force_data, &config, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    std::function<void(const mfem::Vector &, mfem::Vector &)> initial_data_u =
        std::bind(&SimulationConfig::initial_data_u, &config, std::placeholders::_1, std::placeholders::_2);
    std::function<void(const mfem::Vector &, mfem::Vector &)> initial_data_w =
        std::bind(&SimulationConfig::initial_data_w, &config, std::placeholders::_1, std::placeholders::_2);

   // ------------------------------------------------------------------
   // 1. Mesh and FE spaces (serial for now)
   // ------------------------------------------------------------------
   Mesh mesh(mesh_string.c_str(), 1, 1);
   int dim = mesh.Dimension();

   for (int l = 0; l < refinements; l++)
   {
      mesh.UniformRefinement();
   }

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
   // 2. Unknowns and gridfunctions
   // ------------------------------------------------------------------
   GridFunction u(&ND);
   u = 0.0;
   GridFunction z(&RT);
   z = 0.0;
   GridFunction p(&CG);
   p = 0.0;
   GridFunction v(&RT);
   v = 0.0;
   GridFunction w(&ND);
   w = 0.0;
   GridFunction q(&DG);
   q = 0.0;

   // For now, just constant initial data
   {
      VectorFunctionCoefficient u0(dim, initial_data_u);
      VectorFunctionCoefficient w0(dim, initial_data_w);
      u.ProjectCoefficient(u0);
      v.ProjectCoefficient(u0);
      w.ProjectCoefficient(w0);
      z.ProjectCoefficient(w0);
   }

   // ------------------------------------------------------------------
   // 3. System sizes and block layout
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

   // ------------------------------------------------------------------
   // 4. Time-independent bilinear/mixed forms (matrix-free / PA)
   // ------------------------------------------------------------------
   // Mass matrices M (on ND) and N (on RT)
   ConstantCoefficient one_coeff(1.0);
   BilinearForm blf_M(&ND);
   blf_M.AddDomainIntegrator(new VectorFEMassIntegrator(one_coeff));
   blf_M.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_M.Assemble();
   Operator &M_op = blf_M;
   ScaledOperator M_dt_op(&M_op, 1.0 / dt);
   ScaledOperator M_n_op(&M_op, -1.0);

   BilinearForm blf_N(&RT);
   blf_N.AddDomainIntegrator(new VectorFEMassIntegrator(one_coeff));
   blf_N.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_N.Assemble();
   Operator &N_op = blf_N;
   ScaledOperator N_dt_op(&N_op, 1.0 / dt);
   ScaledOperator N_n_op(&N_op, -1.0);

   // C : ND -> RT (curl)
   MixedBilinearForm blf_C(&ND, &RT);
   blf_C.AddDomainIntegrator(new MixedVectorCurlIntegrator());
   blf_C.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_C.Assemble();
   Operator &C_op = blf_C;
   ScaledOperator C_Re_op(&C_op, viscosity / 2.0);
   TransposeOperator CT_op(C_op);
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
   blf_G.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_G.Assemble();
   Operator &G_op = blf_G;
   TransposeOperator GT_op(G_op);

   // ------------------------------------------------------------------
   // 5. Time-stepping
   // ------------------------------------------------------------------
   double t = 0.0;
   int cycle = 0;

   // For forcing: f = 0 here, but you can plug in your VectorFunctionCoefficient
   Vector zero_vec(dim);
   zero_vec = 0.0;
   VectorFunctionCoefficient f_coeff(dim, [](const Vector &x, Vector &val)
                                     { val = 0.0; });

   // Rhs containers
   Vector b1(size_1), b1sub(u.Size());
   Vector b2(size_2), b2sub(v.Size());

   // Solver selection (only iterative, since we are matrix-free)
   unique_ptr<Solver> solver;
   if (solver_type == "MINRES")
   {
      auto minres = make_unique<MINRESSolver>();
      minres->SetAbsTol(tol);
      minres->SetRelTol(0.);
      minres->SetMaxIter(10000);
      minres->SetPrintLevel(printlevel);
      solver = std::move(minres);
   }
   else // CG
   {
      auto cg = make_unique<CGSolver>();
      cg->SetAbsTol(tol);
      cg->SetRelTol(0.);
      cg->SetMaxIter(10000);
      cg->SetPrintLevel(printlevel);
      solver = std::move(cg);
   }

   // Pre-allocate temporaries
   Vector tmp_u(u.Size()), tmp_v(v.Size());

   // Helper coefficients that depend on w or z
   VectorGridFunctionCoefficient w_gfcoeff(&w);
   VectorGridFunctionCoefficient z_gfcoeff(&z);
   ConstantCoefficient two_over_dt(2.0 / dt);

   // Euler step: build MR_eul operator (2/dt M + cross(w,·)) in PA
   {
      MixedBilinearForm blf_MR_eul(&ND, &ND);
      blf_MR_eul.AddDomainIntegrator(new VectorFEMassIntegrator(two_over_dt));
      blf_MR_eul.AddDomainIntegrator(new MixedCrossProductIntegrator(w_gfcoeff));
      // blf_MR_eul.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      blf_MR_eul.Assemble();
      Operator &MR_eul_op = blf_MR_eul;

      // CT_eul = 2 * CT_Re (operator scaling)
      ScaledOperator CT_eul_op(&CT_Re_op, 2.0);

      // Build A1 for Euler step
      A1.SetBlock(0, 0, &MR_eul_op);
      A1.SetBlock(0, 1, &CT_eul_op);
      A1.SetBlock(0, 2, &G_op);
      A1.SetBlock(1, 0, &C_op);
      A1.SetBlock(1, 1, &N_n_op);
      A1.SetBlock(2, 0, &GT_op);

      // Build rhs b1:  b1 = 2*M_dt*u + f1  (simplified, no extra terms)
      b1 = 0.0;
      b1sub = 0.0;
      M_dt_op.Mult(u, tmp_u);
      b1sub.Add(2.0, tmp_u); // emulate AddMult(u, b1sub, 2)

      b1.AddSubVector(b1sub, 0);
      // you can add f1, boundary terms etc here as in your original code

      // Normal equations A1^T A1 x = A1^T b1
      TransposeOperator AT1(&A1);
      ProductOperator ATA1(&AT1, &A1, false, false);

      Vector ATb1(size_1);
      A1.MultTranspose(b1, ATb1);

      solver->SetOperator(ATA1);
      solver->Mult(ATb1, x);

      // extract u,z,p
      x.GetSubVector(u_dofs, u);
      x.GetSubVector(z_dofs, z);
      x.GetSubVector(p_dofs, p);
   }

   // --- CSV output for monitoring six variables (u,z,p,v,w,q) ---
   EnergyCSVLogger csv_logger(config, M_op, N_op, u, v, w, z);
   csv_logger.WriteRow(0, 0., 0.);

   // Set up ParaView data collection for visualization
   mfem::ParaViewDataCollection vtk_dc("/home/wtonnon/VisualStudioProjects/dualfieldmfem/data/visualisation/paraview/" + output_file, &mesh);
   if (visualisation > 0)
   {
      vtk_dc.RegisterField("u1", &u); // Register field for visualization
      vtk_dc.RegisterField("u2", &v); // Register field for visualization
      vtk_dc.RegisterField("w1", &w); // Register field for visualization
      vtk_dc.RegisterField("w2", &z); // Register field for visualization
      vtk_dc.RegisterField("p0", &p); // Register field for visualization
      vtk_dc.RegisterField("p3", &q); // Register field for visualization
      vtk_dc.SetCycle(0);             // Set initial cycle
      vtk_dc.SetTime(0.0);            // Set initial time
      vtk_dc.Save();                  // Save initial data
   }

   // ------------------------------------------------------------------
   // 6. Time loop (simplified, but matrix-free)
   // ------------------------------------------------------------------
   while (t < T - 0.5 * dt)
   {
      t += dt;
      cycle++;

      // === DUAL FIELD: build R2 and NR as PA operators ===
      MixedBilinearForm blf_R2(&RT, &RT);
      blf_R2.AddDomainIntegrator(new MixedCrossProductIntegrator(z_gfcoeff));
      // blf_R2.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      blf_R2.Assemble();
      Operator &R2_op = blf_R2;
      ScaledOperator R2_half_op(&R2_op, 0.5); // R2 * 1/2

      MixedBilinearForm blf_NR(&RT, &RT);
      blf_NR.AddDomainIntegrator(new VectorFEMassIntegrator(two_over_dt));
      blf_NR.AddDomainIntegrator(new MixedCrossProductIntegrator(z_gfcoeff));
      // blf_NR.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      blf_NR.Assemble();
      Operator &NR_op = blf_NR;
      ScaledOperator NR_half_op(&NR_op, 0.5);

      // A2 blocks:
      A2.SetBlock(0, 0, &NR_half_op);
      A2.SetBlock(0, 1, &C_Re_op);
      A2.SetBlock(0, 2, &DT_n_op);
      A2.SetBlock(1, 0, &CT_op);
      A2.SetBlock(1, 1, &M_n_op);
      A2.SetBlock(2, 0, &D_op);

      // rhs b2 = N_dt*v - R2*v - C_Re*w + f2
      b2 = 0.0;
      b2sub = 0.0;

      N_dt_op.Mult(v, tmp_v);
      b2sub += tmp_v;

      R2_op.Mult(v, tmp_v);
      b2sub.Add(-1.0, tmp_v);

      C_Re_op.Mult(w, tmp_v);
      b2sub.Add(-1.0, tmp_v);

      b2.AddSubVector(b2sub, 0);
      // add f2 etc as in your original code

      // normal equations
      TransposeOperator AT2(&A2);
      ProductOperator ATA2(&AT2, &A2, false, false);
      Vector ATb2(size_2);
      A2.MultTranspose(b2, ATb2);

      solver->SetOperator(ATA2);
      solver->Mult(ATb2, y);

      y.GetSubVector(v_dofs, v);
      y.GetSubVector(w_dofs, w);
      y.GetSubVector(q_dofs, q);

      // === PRIMAL FIELD: build R1 and MR as PA operators ===
      MixedBilinearForm blf_R1(&ND, &ND);
      blf_R1.AddDomainIntegrator(new MixedCrossProductIntegrator(w_gfcoeff));
      // blf_R1.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      blf_R1.Assemble();
      Operator &R1_op = blf_R1;
      ScaledOperator R1_half_op(&R1_op, 0.5);

      MixedBilinearForm blf_MR(&ND, &ND);
      blf_MR.AddDomainIntegrator(new VectorFEMassIntegrator(two_over_dt));
      blf_MR.AddDomainIntegrator(new MixedCrossProductIntegrator(w_gfcoeff));
      // blf_MR.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      blf_MR.Assemble();
      Operator &MR_op = blf_MR;
      ScaledOperator MR_half_op(&MR_op, 0.5);

      // A1 blocks:
      A1.SetBlock(0, 0, &MR_half_op);
      A1.SetBlock(0, 1, &CT_Re_op);
      A1.SetBlock(0, 2, &G_op);
      A1.SetBlock(1, 0, &C_op);
      A1.SetBlock(1, 1, &N_n_op);
      A1.SetBlock(2, 0, &GT_op);

      // rhs b1 = M_dt*u - R1*u - CT_Re*z + f1 + boundary terms...
      b1 = 0.0;
      b1sub = 0.0;

      M_dt_op.Mult(u, tmp_u);
      b1sub += tmp_u;

      R1_op.Mult(u, tmp_u);
      b1sub.Add(-1.0, tmp_u);

      CT_Re_op.Mult(z, tmp_u);
      b1sub.Add(-1.0, tmp_u);

      b1.AddSubVector(b1sub, 0);
      // add f1, lform_zxn, lform_un as in your original code

      // normal equations
      TransposeOperator AT1_loop(&A1);
      ProductOperator ATA1_loop(&AT1_loop, &A1, false, false);

      Vector ATb1_loop(size_1);
      A1.MultTranspose(b1, ATb1_loop);

      solver->SetOperator(ATA1_loop);
      solver->Mult(ATb1_loop, x);

      x.GetSubVector(u_dofs, u);
      x.GetSubVector(z_dofs, z);
      x.GetSubVector(p_dofs, p);

      // Append norms to CSV for this timestep (after full primal solve)
      csv_logger.WriteRow(cycle, t, t - .5 * dt);

      // Save data to ParaView if visualization is enabled
      if (visualisation > 0)
      {
         vtk_dc.SetCycle(cycle);     // Update cycle in ParaView
         vtk_dc.SetTime(dt * cycle); // Update time in ParaView
         vtk_dc.Save();              // Save data
      }

      if (rank == 0)
      {
         cout << "Step " << cycle << ", t=" << t << " done.\n";
      }
   }

   // ------------------------------------------------------------------
   // 7. Cleanup
   // ------------------------------------------------------------------
   delete fec_DG;
   delete fec_ND;
   delete fec_RT;
   delete fec_CG;

   MPI_Finalize();
   return 0;
}
