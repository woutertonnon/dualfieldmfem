
//               We recommend viewing Example 22 before viewing this example.
 
#include "mfem.hpp"
#include <fstream>
#include <iostream>
 
 
 

int main(int argc, char *argv[])
{

   const int N=1e7;
   mfem::SparseMatrix A(N);
   A.Add(0,0,2.);
   A.Add(0,1,-1.);
   for(int i=1;i<N-1;++i){
      A.Add(i,i,2.);
      A.Add(i,i+1,-1.);
      A.Add(i,i-1,-1.);
   }
   A.Add(N-1,N-1,2.);
   A.Add(N-1,N-2,-1.);
   A.Finalize();



   //A.ToDenseMatrix()->Print(std::cout);

   mfem::UMFPackSolver solver;
   solver.SetOperator(A);
   auto bsetup = std::vector<double>(N,3);
   mfem::Vector b(bsetup.data(),N), x(N);
   //b.Print(std::cout);

   solver.Mult(b,x);
   //x.Print(std::cout);
   mfem::Vector res(b);
   A.AddMult(x,res,-1.);
   std::cout<< "Error: " << res.Norml2()/b.Norml2() << std::endl;

   return 0;
}
 