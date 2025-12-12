#include "BoundaryOperators.h"
#include "mfem.hpp"


void GradJumpIntegrator::AssembleFaceMatrix(
    const mfem::FiniteElement &el1, const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &Trans, mfem::DenseMatrix &elmat)
{
   MFEM_ASSERT(Trans.Elem2No >= 0, "GradJumpIntegrator needs an interior face");

   const int d1  = el1.GetDof();
   const int d2  = el2.GetDof();
   const int dim = Trans.GetSpaceDim();

   // quadrature: use a reasonable rule on the actual face geom
   const int p = std::max(el1.GetOrder(), el2.GetOrder());
   const mfem::IntegrationRule *ir =
      &mfem::IntRules.Get(static_cast<mfem::Geometry::Type>(Trans.FaceGeom),
                        2*p + 1);

   elmat.SetSize(d1 + d2);
   elmat = 0.0;

   auto weights = ir->GetWeights();

   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      const mfem::IntegrationPoint &ip_face = ir->IntPoint(i);

      // set face + both element IPs at once
      Trans.SetAllIntPoints(&ip_face);

      // face Jacobian and unit normal (2D: rotate tangent; 3D: cross of tangents)
      mfem::DenseMatrix J = Trans.Face->Jacobian();
      mfem::Vector nor(dim);
      double h;

      if (dim == 2)
      {
         mfem::Vector tan(2);
         J.GetColumn(0, tan);
         nor(0) =  tan(1);
         nor(1) = -tan(0);
         nor /= nor.Norml2();

         // local face size h_F ~ |t|
         h = tan.Norml2();
      }
      else // dim == 3
      {
         mfem::Vector t1(3), t2(3);
         J.GetColumn(0, t1);
         J.GetColumn(1, t2);

         // nor = t1 x t2
         nor.SetSize(3);
         nor(0) = t1(1)*t2(2) - t1(2)*t2(1);
         nor(1) = t1(2)*t2(0) - t1(0)*t2(2);
         nor(2) = t1(0)*t2(1) - t1(1)*t2(0);

         const double area = nor.Norml2();   // |t1 x t2| = local area scale
         nor /= area;                        // unit normal
         h = std::sqrt(area);                // length-like face scale
      }

      // physical ds weight
      const double wds = weights[i] * Trans.Face->Weight();
      const double fac = alpha * h * wds;  // gamma * h_F * ds

      // physical gradients on both sides
      mfem::DenseMatrix dshape1(d1, dim), dshape2(d2, dim);
      el1.CalcPhysDShape(*Trans.Elem1, dshape1);
      el2.CalcPhysDShape(*Trans.Elem2, dshape2);

      // normal derivatives arrays
      mfem::Vector dn1(d1), dn2(d2);
      for (int a = 0; a < d1; ++a)
      {
         double s = 0.0; for (int d = 0; d < dim; ++d) s += dshape1(a,d)*nor(d);
         dn1[a] = s;
      }
      for (int a = 0; a < d2; ++a)
      {
         double s = 0.0; for (int d = 0; d < dim; ++d) s += dshape2(a,d)*nor(d);
         dn2[a] = s;
      }

      // TL: + d1 d1^T
      for (int l = 0; l < d1; ++l)
         for (int k = 0; k < d1; ++k)
            elmat(l, k) += fac * dn1[l] * dn1[k];

      // TR: - d1 d2^T
      for (int l = 0; l < d1; ++l)
         for (int k = 0; k < d2; ++k)
            elmat(l, d1 + k) -= fac * dn1[l] * dn2[k];

      // BL: - d2 d1^T
      for (int l = 0; l < d2; ++l)
         for (int k = 0; k < d1; ++k)
            elmat(d1 + l, k) -= fac * dn2[l] * dn1[k];

      // BR: + d2 d2^T
      for (int l = 0; l < d2; ++l)
         for (int k = 0; k < d2; ++k)
            elmat(d1 + l, d1 + k) += fac * dn2[l] * dn2[k];
   }
}



void WouterIntegrator::AssembleElementMatrix(const mfem::FiniteElement &el, mfem::ElementTransformation &Trans,
                                             mfem::DenseMatrix &elmat)
{
   const mfem::IntegrationRule *ir = IntRule;
   ir = &mfem::IntRules.Get(el.GetGeomType(), 4);

   // Extract vertices in physical space
   mfem::IntegrationPoint ip;
   mfem::Vector v1, v2;
   ip.Set2(0., 0.);
   Trans.Transform(ip, v1);
   ip.Set2(1., 0.);
   Trans.Transform(ip, v2);
   // v1.Print(std::cout);
   // v2.Print(std::cout);

   // Compute the tangent vector
   mfem::Vector tan(2);
   tan.Elem(0) = v2.Elem(0) - v1.Elem(0);
   tan.Elem(1) = v2.Elem(1) - v1.Elem(1);
   tan /= tan.Norml2();
   // std::cout << "tan: ";
   // tan.Print(std::cout);
   // std::cout << std::endl;

   // Compute the normal vector
   mfem::Vector nor(2);
   nor.Elem(0) = tan.Elem(1);
   nor.Elem(1) = -tan.Elem(0);
   // std::cout << "nor: ";
   // nor.Print(std::cout);
   // std::cout << std::endl;

   // Approximate size of element
   mfem::real_t h = Trans.Jacobian().Det();

   elmat.SetSize(el.GetDof(), el.GetDof());
   elmat = 0.;
   auto weights = ir->GetWeights();
   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      ip = ir->IntPoint(i);

      // std::cout << "test\n";
      Trans.SetIntPoint(&ip);
      mfem::DenseMatrix shape(el.GetDof(), Trans.GetSpaceDim());
      mfem::DenseMatrix curl_shape(el.GetDof(), 1);
      // std::cout << "test\n";
      el.CalcVShape(Trans, shape);
      // std::cout << Trans.GetSpaceDim() << "test\n";
      el.CalcPhysCurlShape(Trans, curl_shape);

      // std::cout << "testq\n";
      for (int l = 0; l < el.GetDof(); l++)
         for (int k = 0; k < el.GetDof(); k++)
         {
            // Extract u and v
            mfem::Vector u, v;
            shape.GetRow(l, u);
            shape.GetRow(k, v);

            // std::cout << "test3\n";
            //  Extract curl(u) and curl(v)
            double curl_u = curl_shape.Elem(l, 0);
            double curl_v = curl_shape.Elem(k, 0);

            // std::cout << "test4\n";
            //  (n x u, n x v)
            elmat.Elem(l, k) += alpha * weights[i] * 1e6 / h * mfem::InnerProduct(u, tan) * mfem::InnerProduct(v, tan);

            // std::cout << "test5\n";
            //  (n x curl(u), v)
            elmat.Elem(l, k) += alpha * weights[i] * (nor[1] * curl_u * v.Elem(0) + -nor[0] * curl_u * v.Elem(1));

            // std::cout << "test6\n";
            //  (u, n x curl(v))
            elmat.Elem(l, k) += alpha * weights[i] * (nor[1] * curl_u * v.Elem(0) + -nor[0] * curl_u * v.Elem(1));
         }
   }
}

void WouterIntegrator::AssembleFaceMatrix(
    const mfem::FiniteElement &el1, const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &Trans, mfem::DenseMatrix &elmat)
{
   MFEM_ASSERT(Trans.Elem2No < 0,
               "support for interior faces is not implemented");
   mfem::IntegrationPoint ip, ip_face, ip_elem;
   ip.Set2(.5, 0.);
   mfem::Vector loc;
   // Get the face Jacobian and compute the normal vector
   Trans.SetIntPoint(&ip);
   Trans.Transform(ip, loc);

   // Compute the Jacobian of the face
   mfem::DenseMatrix J;
   J = Trans.Face->Jacobian();
   // J.Print(std::cout);

   // For 2D: Normal is orthogonal to the Jacobian rows
   mfem::Vector nor(2);
   nor(0) = -J(1, 0); // Swap and negate for orthogonal vector
   nor(1) = J(0, 0);

   // Normalize the normal vector
   double norm = nor.Norml2();
   nor /= -norm;

   mfem::Vector tan(2);
   J.GetColumn(0, tan);
   mfem::real_t h = tan.Norml2();
   tan /= h;

   // Print the normal vector
   // std::cout << "Center boundary: ";
   // loc.Print(std::cout);
   // std::cout << "Normal at boundary face ";
   // nor.Print(std::cout);
   // std::cout << "Tangent at boundary face ";
   // tan.Print(std::cout);

   const mfem::IntegrationRule *ir = IntRule;
   ir = &mfem::IntRules.Get(mfem::Geometry::Type::SEGMENT, 1);
   // Approximate size of element

   elmat.SetSize(el1.GetDof(), el1.GetDof());
   elmat = 0.;
   auto weights = ir->GetWeights();
   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      ip_face = ir->IntPoint(i);

      Trans.SetIntPoint(&ip_face);
      Trans.Loc1.Transform(ip_face, ip_elem);
      mfem::DenseMatrix shape(el1.GetDof(), Trans.GetSpaceDim());
      mfem::DenseMatrix curl_shape(el1.GetDof(), 1);

      mfem::ElementTransformation *tr1 = Trans.Elem1;
      tr1->SetIntPoint(&ip_elem);
      el1.CalcVShape(*tr1, shape);
      el1.CalcPhysCurlShape(*tr1, curl_shape);

      for (int l = 0; l < el1.GetDof(); l++)
         for (int k = 0; k < el1.GetDof(); k++)
         {
            // Extract u and v
            mfem::Vector u, v;
            shape.GetRow(l, u);
            shape.GetRow(k, v);

            // Extract curl(u) and curl(v)
            double curl_u = curl_shape.Elem(l, 0);
            double curl_v = curl_shape.Elem(k, 0);

            // (n x u, n x v)
            // std::cout
            //<< weights[i] << std::endl
            //<< h << std::endl
            //<< mfem::InnerProduct(u, tan) << std::endl
            //<< mfem::InnerProduct(v, tan) << std::endl<< std::endl;
            elmat.Elem(l, k) += weights[i] * alpha * h * (1e6 / h) * mfem::InnerProduct(u, tan) * mfem::InnerProduct(v, tan);

            // (n x curl(u), v)
            elmat.Elem(l, k) += weights[i] * alpha * h * (nor[1] * curl_u * v.Elem(0) + -nor[0] * curl_u * v.Elem(1));

            // (u, n x curl(v))
            elmat.Elem(l, k) += weights[i] * alpha * h * (nor[1] * curl_v * u.Elem(0) + -nor[0] * curl_v * u.Elem(1));
         }
   }
}

void WouterLFIntegrator::AssembleRHSElementVect(
    const mfem::FiniteElement &el, mfem::ElementTransformation &Tr, mfem::Vector &elvect)
{
   return;
}

void WouterLFIntegrator::AssembleRHSElementVect(
    const mfem::FiniteElement &el, mfem::FaceElementTransformations &Tr, mfem::Vector &elvect)
{

   mfem::IntegrationPoint ip, ip_face, ip_elem;
   ip.Set2(.5, 0.);
   mfem::Vector loc;
   // Get the face Jacobian and compute the normal vector
   Tr.SetIntPoint(&ip);
   Tr.Transform(ip, loc);

   // Compute the Jacobian of the face
   mfem::DenseMatrix J;
   J = Tr.Face->Jacobian();
   // J.Print(std::cout);

   // For 2D: Normal is orthogonal to the Jacobian rows
   mfem::Vector nor(2);
   nor(0) = -J(1, 0); // Swap and negate for orthogonal vector
   nor(1) = J(0, 0);

   // Normalize the normal vector
   double norm = nor.Norml2();
   nor /= -norm;

   mfem::Vector tan(2);
   J.GetColumn(0, tan);
   mfem::real_t h = tan.Norml2();
   tan /= h;

   int dof = el.GetDof();

   const mfem::IntegrationRule *ir = IntRule;
   ir = &mfem::IntRules.Get(mfem::Geometry::Type::SEGMENT, 1);
   // Approximate size of element

   elvect.SetSize(el.GetDof());
   elvect = 0.;
   auto weights = ir->GetWeights();
   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      ip_face = ir->IntPoint(i);

      Tr.SetIntPoint(&ip_face);
      Tr.Transform(ip_face, loc);



      mfem::Vector u;
      Q.Eval(u,Tr,ip_face);




      Tr.SetIntPoint(&ip_face);
      Tr.Loc1.Transform(ip_face, ip_elem);
      mfem::DenseMatrix shape(el.GetDof(), Tr.GetSpaceDim());
      mfem::DenseMatrix curl_shape(el.GetDof(), 1);

      mfem::ElementTransformation *tr1 = Tr.Elem1;
      tr1->SetIntPoint(&ip_elem);
      el.CalcVShape(*tr1, shape);
      el.CalcPhysCurlShape(*tr1, curl_shape);

      for (int l = 0; l < el.GetDof(); l++)
      {
         // Extract u and v
         mfem::Vector v;
         shape.GetRow(l, v);

         // Extract curl(u) and curl(v)
         double curl_v = curl_shape.Elem(l, 0);

         // (n x u, n x v)
         // std::cout
         //<< weights[i] << std::endl
         //<< h << std::endl
         //<< mfem::InnerProduct(u, tan) << std::endl
         //<< mfem::InnerProduct(v, tan) << std::endl<< std::endl;
         elvect.Elem(l) += weights[i] * eps * h * (1e6 / h) * mfem::InnerProduct(u, tan) * mfem::InnerProduct(v, tan);

         // (n x curl(u), v)
         //elvect.Elem(l) += weights[i] * h * (nor[1] * curl_u * v.Elem(0) + -nor[0] * curl_u * v.Elem(1));

         // (u, n x curl(v))
         elvect.Elem(l) += weights[i] * eps * h * (nor[1] * curl_v * u.Elem(0) + -nor[0] * curl_v * u.Elem(1));
      }
   }
}
