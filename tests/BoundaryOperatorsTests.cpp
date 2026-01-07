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
   blf_A.AddBdrFaceIntegrator(new WouterIntegrator(0.,0.));
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
   blf_A.AddBdrFaceIntegrator(new WouterIntegrator(-1.,0.));
   blf_A.Assemble();
   blf_A.Finalize();

   mfem::Vector A_u(ND.GetNDofs());
   blf_A.MultTranspose(u, A_u);
   ASSERT_FLOAT_EQ(-3. / 4.-1./3., v * A_u);
}


TEST(WouterIntegratorTest, ApproximationTest)
{
   // We computed <n x curl(u), v>_boundary for u = (xyz,xxz,xyy) and v=(xx+y,yy+z,zz+x). The exact solution is -3/4
   double last_err;
   double one_but_last_err;
   for(int order = 1; order < 3; ++order){
       for(int refinements = 0; refinements < 9-2*order; refinements++){
        
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
              y.Elem(0) = std::exp(X - 2*Y + Z) + std::sin(2*M_PI*X)*std::cos(M_PI*Z) + X*Y*(1 - Z);
              y.Elem(1) = X*X*std::sin(M_PI*Y) + std::cos(2*M_PI*Z)*(Y - Z) + std::exp(-X*Z);
              y.Elem(2) = std::sin(M_PI*X*Y) + Z*Z*std::cos(2*M_PI*Y) + (X - Y)*std::exp(Z);
        
           };
        
           auto v_func = [](const mfem::Vector &x, double, mfem::Vector &y) -> void
           {
              const double X = x.Elem(0);
              const double Y = x.Elem(1);
              const double Z = x.Elem(2);
        
              y.SetSize(3); 
              y.Elem(0) = std::cos(M_PI*X)*std::exp(Y - Z) + X*(1 - X)*Y + std::sin(2*M_PI*Z);
              y.Elem(1) = std::sin(2*M_PI*X*Z) + std::exp(-Y) + (Y - 1./2.)*(Y - 1./2.)*(Y - 1./2.);
              y.Elem(2) = std::cos(2*M_PI*Y*Z) + std::exp(X*Y) - Z*(1 - Z);
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
           blf_A.AddBdrFaceIntegrator(new WouterIntegrator(1.0,0.));
           blf_A.Assemble();
           blf_A.Finalize();
        
           mfem::Vector A_u(ND.GetNDofs());
           blf_A.MultTranspose(u, A_u);
           one_but_last_err = last_err;
           last_err = 4.4722583402915601 - (v * A_u);
           std::cout << "refinement: " << refinements << ", order: " << order << ", error: " << last_err << std::endl;
       }
       EXPECT_LT(last_err, (std::pow(0.5,order)+0.01)*one_but_last_err);
   }
}
TEST(WouterIntegratorTest, ApproximationTestAsymmetricPenalty)
{
   double last_err;
   double one_but_last_err;
   std::vector<double> exact_integrals{667.0180872213067,1330.817580069015,2658.416565764433,5313.614537155269,10624.01047993694,21244.80236550028}; // The exact integral is now dependent on h, because of the C/h <u x n, v x n> term in the integral.
   for(int order = 1; order < 3; ++order){
       for(int refinements = 0; refinements < exact_integrals.size(); refinements++){
        
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
              y.Elem(0) = std::exp(X - 2*Y + Z) + std::sin(2*M_PI*X)*std::cos(M_PI*Z) + X*Y*(1 - Z);
              y.Elem(1) = X*X*std::sin(M_PI*Y) + std::cos(2*M_PI*Z)*(Y - Z) + std::exp(-X*Z);
              y.Elem(2) = std::sin(M_PI*X*Y) + Z*Z*std::cos(2*M_PI*Y) + (X - Y)*std::exp(Z);
        
           };
        
           auto v_func = [](const mfem::Vector &x, double, mfem::Vector &y) -> void
           {
              const double X = x.Elem(0);
              const double Y = x.Elem(1);
              const double Z = x.Elem(2);
        
              y.SetSize(3); 
              y.Elem(0) = std::cos(M_PI*X)*std::exp(Y - Z) + X*(1 - X)*Y + std::sin(2*M_PI*Z);
              y.Elem(1) = std::sin(2*M_PI*X*Z) + std::exp(-Y) + (Y - 1./2.)*(Y - 1./2.)*(Y - 1./2.);
              y.Elem(2) = std::cos(2*M_PI*Y*Z) + std::exp(X*Y) - Z*(1 - Z);
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
           blf_A.AddBdrFaceIntegrator(new WouterIntegrator(-1.0,100.));
           blf_A.Assemble();
           blf_A.Finalize();
        
           mfem::Vector A_u(ND.GetNDofs());
           blf_A.MultTranspose(u, A_u);
           one_but_last_err = last_err;
           last_err = std::abs(exact_integrals.at(refinements) - (v * A_u));
           std::cout << "refinement: " << refinements << ", order: " << order << ", error: " << last_err << std::endl;
       }
       std::cout << last_err << ", " << one_but_last_err << "," << std::pow(0.5,order)+0.01 << (std::pow(0.5,order)+0.01)*one_but_last_err << std::endl;
       EXPECT_LT(last_err, (std::pow(0.5,order)+0.03)*one_but_last_err);
   }
}

TEST(WouterIntegratorTest, DefaultsHaveNoCoefficient3)
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
      y.Elem(0) =1;
      y.Elem(1) = 1;
      y.Elem(2) = 1;
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
   blf_A.AddBdrFaceIntegrator(new WouterIntegrator(0.,0.));
   blf_A.Assemble();
   blf_A.Finalize();

   mfem::Vector A_u(ND.GetNDofs());
   blf_A.MultTranspose(u, A_u);
   ASSERT_NEAR(0., v * A_u, 1e-12);
}

TEST(WouterIntegratorTest, DefaultsHaveNoCoefficient4)
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
      y.Elem(0) =1;
      y.Elem(1) = 1;
      y.Elem(2) = 1;
   };

   mfem::VectorFunctionCoefficient u_coef(3, u_func);

   mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order, dim);
   mfem::FiniteElementSpace ND(&mesh, fec_ND);
   mfem::GridFunction u(&ND), v(&ND);
   u.ProjectCoefficient(u_coef);



   // mfem::ConstantCoefficient mass_coeff(1.), diff_coef(viscosity);
   mfem::BilinearForm blf_A(&ND);
   blf_A.AddBdrFaceIntegrator(new WouterIntegrator(0.,0.));
   blf_A.Assemble();
   blf_A.Finalize();

   mfem::Vector A_u(ND.GetNDofs());
   blf_A.MultTranspose(u, A_u);
   for(auto val: A_u)
      ASSERT_NEAR(val,0.,1e-10);
}


