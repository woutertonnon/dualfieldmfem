#include <gtest/gtest.h>
#include "BoundaryOperators.h"
#include "mfem.hpp"

TEST(WouterIntegratorTest, DefaultsHaveNoCoefficient)
{
   // We computed <n x curl(u), v>_boundary for u = (0,0,xy) and v=(0,0,x+y). The exact solution is -1
   int refinements = 3;
   int order = 1;

   mfem::Mesh mesh("../extern/mfem/data/ref-cube.mesh", 1, 1);
   for (int l = 0; l < refinements; l++)
   {
      mesh.UniformRefinement();
   }
   int dim = mesh.Dimension();

   auto u_func = [](const mfem::Vector &x, double, mfem::Vector &y) -> void
   {
      y.Elem(0) = 0.;
      y.Elem(1) = 0.;
      y.Elem(2) = x.Elem(0) * x.Elem(1);
      return;
   };
   auto v_func = [](const mfem::Vector &x, double, mfem::Vector &y) -> void
   {
      y.Elem(0) = 0.;
      y.Elem(1) = 0.;
      y.Elem(2) = x.Elem(0) + x.Elem(1);
      return;
   };
   mfem::VectorFunctionCoefficient u_coef(3, u_func);
   mfem::VectorFunctionCoefficient v_coef(3, v_func);

   mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order, dim);
   mfem::FiniteElementSpace ND(&mesh, fec_ND);
   mfem::GridFunction u(&ND), v(&ND);
   u.ProjectCoefficient(u_coef);
   v.ProjectCoefficient(v_coef);

   // mfem::ConstantCoefficient mass_coeff(1.), diff_coef(viscosity);
   mfem::BilinearForm blf_A(&ND);
   blf_A.AddBdrFaceIntegrator(new WouterIntegrator(0.));
   blf_A.Assemble();

   mfem::Vector A_u(ND.GetNDofs());
   blf_A.Mult(u, A_u);
   std::cout << "Integrated quantity: " << A_u * v << std::endl;

   ASSERT_FLOAT_EQ(-1., v * A_u);
}

TEST(WouterIntegratorTest, DefaultsHaveNoCoefficient2)
{
   // We computed <n x curl(u), v>_boundary for u = (xyz,xxz,xyy) and v=(xx+y,yy+z,zz+x). The exact solution is -3/4
   int refinements = 0;
   int order = 2;

   mfem::Mesh mesh("../extern/mfem/data/ref-cube.mesh", 1, 1);
   for (int l = 0; l < refinements; l++)
   {
      mesh.UniformRefinement();
   }
   int dim = mesh.Dimension();
   
   auto u_func = [](const mfem::Vector &x, double, mfem::Vector &y) -> void
   {
      const double X = x.Elem(0);
      const double Y = x.Elem(1);
      const double Z = x.Elem(2);

      // u = (0, 0, X*Y)
      y.SetSize(3);
      y.Elem(0) = X*Y*Z;
      y.Elem(1) = X*X*Z;
      y.Elem(2) = X*Y*Y;
   };

   auto v_func = [](const mfem::Vector &x, double, mfem::Vector &y) -> void
   {
      const double X = x.Elem(0);
      const double Y = x.Elem(1);
      const double Z = x.Elem(2);

      // v = (0, 0, X + Y + X*Y)   (slightly more complex than linear)
      y.SetSize(3);
      y.Elem(0) = X*X+Y;
      y.Elem(1) = Y*Y+Z;
      y.Elem(2) = Z*Z+X;
   };

   mfem::VectorFunctionCoefficient u_coef(3, u_func);
   mfem::VectorFunctionCoefficient v_coef(3, v_func);

   mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order, dim);
   mfem::FiniteElementSpace ND(&mesh, fec_ND);
   mfem::GridFunction u(&ND), v(&ND);
   u.ProjectCoefficient(u_coef);
   v.ProjectCoefficient(v_coef);



   // mfem::ConstantCoefficient mass_coeff(1.), diff_coef(viscosity);
   mfem::BilinearForm blf_A(&ND);
   blf_A.AddBdrFaceIntegrator(new WouterIntegrator(0.));
   blf_A.Assemble();
   blf_A.Finalize();

   mfem::Vector A_u(ND.GetNDofs());
   blf_A.MultTranspose(u, A_u);
   ASSERT_FLOAT_EQ(-3. / 4., v * A_u);
}
