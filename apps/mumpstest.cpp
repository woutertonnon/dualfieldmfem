#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <mpi.h>

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   // Initialize MPI
   MPI_Init(&argc, &argv);
   int rank, nprocs;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   if (rank == 0)
   {
      cout << "Running parallel PA diffusion with " << nprocs << " MPI ranks.\n";
   }

   std::cout << "rank " << rank << " reports 1\n";

   // 1. Build a serial mesh and make it parallel
   Mesh mesh = Mesh::MakeCartesian2D(
       264 ,264 , Element::QUADRILATERAL, true, 1.0, 1.0);

   std::cout << "rank " << rank << " reports 2\n";
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear(); // serial mesh no longer needed

   std::cout << "rank " << rank << " reports 3\n";
   int order = 4;
   H1_FECollection fec(order, pmesh.Dimension());
   ParFiniteElementSpace fespace(&pmesh, &fec);

   std::cout << "rank " << rank << " reports 4\n";
   if (rank == 0)
   {
   //   cout << "Global number of DOFs: " << fespace.GlobalTrueVSize() << endl;
   }

   // 2. Diffusion bilinear form a(u,v) = ∫ grad u · grad v
   ConstantCoefficient k(1.0);
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new MassIntegrator());
   a.AddDomainIntegrator(new DiffusionIntegrator(k));

   // *** MATRIX-FREE / PARTIAL ASSEMBLY ***
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.Assemble();

   // 3. RHS: f = 1
   ParGridFunction x(&fespace);
   x = 0.0;

   ParLinearForm b(&fespace);
   ConstantCoefficient f(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(f));
   b.Assemble();

   // 4. Essential BCs: u = 0 on all boundaries
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 5. Build the linear system in matrix-free form
   OperatorPtr A;   // parallel operator
   Vector B, X;     // true-dof vectors (internally HypreParVector)
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   // Now *A is a parallel matrix-free Operator
   mfem::TransposeOperator AT(A.Ptr());
   mfem::ProductOperator ATA(&AT, A.Ptr(), false, false);
    mfem::Vector ATb(B);
    A->MultTranspose(B, ATb);

   // 6. Solve with parallel CG
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetOperator(ATA);
   cg.SetRelTol(1e-8);
   cg.SetAbsTol(0.0);
   cg.SetMaxIter(20000);
   cg.SetPrintLevel(3);

   // Optional: simple Jacobi smoother that works with partial assembly
   //OperatorJacobiSmoother M(a, ess_tdof_list);
   //cg.SetPreconditioner(M);

   cg.Mult(B, X);

   // 7. Recover solution
   a.RecoverFEMSolution(X, b, x);

   // 8. Save result (one file per rank)
   {
      ostringstream sol_name;
      sol_name << "sol." << setfill('0') << setw(6) << rank;
      ofstream sol_ofs(sol_name.str().c_str());
      x.Save(sol_ofs);
   }

   if (rank == 0)
   {
      cout << "Done.\n";
   }

   MPI_Finalize();
   return 0;
}
