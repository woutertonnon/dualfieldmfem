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
   return;
}

void WouterIntegrator::AssembleFaceMatrix(
    const mfem::FiniteElement &el1, const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &Trans, mfem::DenseMatrix &elmat)
{
   MFEM_ASSERT(Trans.Elem2No < 0,
               "support for interior faces is not implemented");

   int dim = el1.GetDim();
   mfem::Vector normal(dim);

   mfem::IntegrationPoint ip_face;

   // Build a reasonable quadrature on the actual face geometry
   const mfem::IntegrationRule *ir = IntRule;
   ir = &mfem::IntRules.Get(static_cast<mfem::Geometry::Type>(Trans.FaceGeom),
                            2*el1.GetOrder()+2);

   elmat.SetSize(el1.GetDof(), el1.GetDof());
   elmat = 0.;
   auto weights = ir->GetWeights();
   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      ip_face = ir->IntPoint(i);

      // Sync face + element integration points. This ensures ip on the element
      // matches the face point orientation (important for tangential fields).
      Trans.SetAllIntPoints(&ip_face);
      const mfem::IntegrationPoint &ip_elem = Trans.Elem1->GetIntPoint();

      // Face normal at this quadrature point
      mfem::CalcOrtho(Trans.Face->Jacobian(), normal);

      mfem::DenseMatrix shape(el1.GetDof(), Trans.GetSpaceDim());
      mfem::DenseMatrix curl_shape(el1.GetDof(), 3);

      mfem::ElementTransformation *tr1 = Trans.Elem1;
      el1.CalcVShape(*tr1, shape);
      el1.CalcPhysCurlShape(*tr1, curl_shape);

      mfem::Vector temp_out(3);
      Trans.Transform(ip_face,temp_out);

      for (int l = 0; l < el1.GetDof(); l++)
         for (int k = 0; k < el1.GetDof(); k++)
         {
            // Extract u and v
            mfem::Vector u(dim), v(dim);
            shape.GetRow(l, u);
            shape.GetRow(k, v);

            // Extract curl(u) and curl(v)
            mfem::Vector curl_u(dim), curl_v(dim);
            curl_shape.GetRow(l, curl_u);
            curl_shape.GetRow(k, curl_v);

            mfem::Vector n_x_curl_u(dim), n_x_curl_v(dim);
            normal.cross3D(curl_u,n_x_curl_u);
            normal.cross3D(curl_v,n_x_curl_v);

            elmat.Elem(l,k) += weights[i] * (n_x_curl_u * v);

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
   int dim = el.GetDim();
   mfem::Vector normal(dim);
   mfem::CalcOrtho(Tr.Loc2.Transf.Jacobian(), normal);


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
