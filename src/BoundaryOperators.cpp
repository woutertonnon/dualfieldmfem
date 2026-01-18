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

//   const mfem::Mesh *mesh = Trans.Face ? Trans.Face->mesh : nullptr;
//   const int faceNo = Trans.Face ? Trans.Face->ElementNo : -1;
//
//   if (mesh && faceNo >= 0)
//   {
//      mfem::Array<int> fv;
//      mesh->GetFaceVertices(faceNo, fv);
//
//      mfem::out << "Boundary face #" << faceNo
//                << " (Elem1No=" << Trans.Elem1No
//                << ", Elem2No=" << Trans.Elem2No << ") vertices: ";
//
//      for (int j = 0; j < fv.Size(); ++j)
//      {
//         const double *X = mesh->GetVertex(fv[j]); // coordinates
//         mfem::out << fv[j] << "=("
//                   << X[0]
//                   << (mesh->SpaceDimension() > 1 ? (std::string(",") + std::to_string(X[1])) : "")
//                   << (mesh->SpaceDimension() > 2 ? (std::string(",") + std::to_string(X[2])) : "")
//                   << ") ";
//      }
//      mfem::out << "\n";
//   }
//   else
//   {
//      mfem::out << "Could not access mesh/faceNo from Trans.Face\n";
//   }
   // Build a reasonable quadrature on the actual face geometry
   const mfem::IntegrationRule *ir = IntRule;
   ir = &mfem::IntRules.Get(static_cast<mfem::Geometry::Type>(Trans.FaceGeom),
                            2*el1.GetOrder()+1);

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

      // Face normal atu this quadrature point
      Trans.Face->SetIntPoint(&ip_face);
      mfem::CalcOrtho(Trans.Face->Jacobian(), normal);
      //std::cout << "normal: " << std::endl;
      //normal.Print(std::cout);
      double h = sqrt(normal.Norml2());
      double area = normal.Norml2();

      //std::cout << "h = " << h << std::endl;
      mfem::DenseMatrix shape(el1.GetDof(), Trans.GetSpaceDim());
      mfem::DenseMatrix curl_shape(el1.GetDof(), 3);

      mfem::ElementTransformation *tr1 = Trans.Elem1;
      //tr1->SetIntPoint(&ip_face);
      el1.CalcVShape(*tr1, shape);
      el1.CalcPhysCurlShape(*tr1, curl_shape);

      mfem::Vector temp_out(3);
      Trans.Transform(ip_face,temp_out);

      for (int l = 0; l < el1.GetDof(); l++)
         for (int k = 0; k < el1.GetDof(); k++)
         {
            // Extract u and v
            mfem::Vector u(dim), v(dim);
            shape.GetRow(k, u);
            shape.GetRow(l, v);
//	    if(l==0){
//	        std::cout << "u[" << l << "] = ";
//	        u.Print(std::cout);
//	    }
//	    if(k==0){
//	        std::cout << "v[" << k << "] = ";
//	        v.Print(std::cout);
//	    }
            // Extract curl(u) and curl(v)
            mfem::Vector curl_u(dim), curl_v(dim);
            curl_shape.GetRow(k, curl_u);
            curl_shape.GetRow(l, curl_v);

            mfem::Vector n_x_curl_u(dim), n_x_curl_v(dim), n_x_u(dim), n_x_v(dim), n_x_u_x_n(dim), n_x_v_x_n(dim);
            normal.cross3D(curl_u,n_x_curl_u);
            normal.cross3D(curl_v,n_x_curl_v);
            normal.cross3D(u,n_x_u);
            normal.cross3D(v,n_x_v);

            elmat.Elem(l,k) += factor_ * weights[i] * (n_x_curl_u * v);
            elmat.Elem(l,k) += factor_ * theta_ * weights[i] * (u * n_x_curl_v);
            elmat.Elem(l,k) += factor_ * Cw_/(h*h*h) * weights[i]* (n_x_u * n_x_v);

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

   mfem::IntegrationPoint ip_face;

   // Build a reasonable quadrature on the actual face geometry
   const mfem::IntegrationRule *ir = IntRule;
   ir = &mfem::IntRules.Get(static_cast<mfem::Geometry::Type>(Tr.FaceGeom),
                            2*el.GetOrder()+12);

   elvect.SetSize(el.GetDof());
   elvect = 0.;
   auto weights = ir->GetWeights();
   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      ip_face = ir->IntPoint(i);

      // Sync face + element integration points. This ensures ip on the element
      // matches the face point orientation (important for tangential fields).
      Tr.SetAllIntPoints(&ip_face);
      const mfem::IntegrationPoint &ip_elem = Tr.Elem1->GetIntPoint();

      // Face normal at this quadrature point
      mfem::CalcOrtho(Tr.Face->Jacobian(), normal);
      double h = sqrt(normal.Norml2());
      mfem::DenseMatrix shape(el.GetDof(), Tr.GetSpaceDim());
      mfem::DenseMatrix curl_shape(el.GetDof(), 3);

      mfem::ElementTransformation *tr1 = Tr.Elem1;
      el.CalcVShape(*tr1, shape);
      el.CalcPhysCurlShape(*tr1, curl_shape);

      mfem::Vector temp_out(3);
      Tr.Transform(ip_face,temp_out);


      mfem::Vector u(3);
      Q.Eval(u,Tr,ip_face);

      for (int k = 0; k < el.GetDof(); k++)
      {
         // Extract u and v
         mfem::Vector v(dim);
         shape.GetRow(k, v);

         // Extract curl(u) and curl(v)
         mfem::Vector curl_v(dim);
         curl_shape.GetRow(k, curl_v);

         mfem::Vector n_x_curl_v(dim), n_x_v(dim), n_x_u(dim);
         normal.cross3D(curl_v,n_x_curl_v);
         normal.cross3D(v,n_x_v);
         normal.cross3D(u,n_x_u);

         elvect.Elem(k) += factor_ * theta_ * weights[i] * (u * n_x_curl_v);
         elvect.Elem(k) += factor_ * Cw_/(h*h*h) * weights[i]* (n_x_u * n_x_v);

      } 
   }


}
