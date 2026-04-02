#ifndef VELOCITY_SMOOTHING_H
#define VELOCITY_SMOOTHING_H

#include "FindElementBFS.h"
#include "mfem.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

/// Provides a Lipschitz-continuous velocity field from an ND₂ GridFunction
/// via the box-averaging procedure of Eq. 32 in Tonnon & Hiptmair (2023):
///
///   ū^i(x) = (1/h) ∫_{x_i - h/2}^{x_i + h/2} u^i(x₁,...,ξ,...,x_d) dξ
///
/// where h = h_min is the minimum edge length in the mesh. Each velocity
/// component is averaged along the corresponding coordinate direction.
///
/// The smoothed velocity is computed on-the-fly at each query point using
/// 2-point Gauss-Legendre quadrature per component direction. This is only
/// called during characteristic tracing (a few points per small edge), so
/// the cost of 2*dim element lookups per evaluation is negligible.
class SmoothedVelocityField
{
public:
   using VelocityFunc =
       std::function<void(const mfem::Vector &, double, int, mfem::Vector &)>;

   SmoothedVelocityField(mfem::Mesh &mesh)
       : mesh_(mesh), dim_(mesh.SpaceDimension()), u_gf_(nullptr)
   {
      MFEM_VERIFY(dim_ == 2 || dim_ == 3,
                  "SmoothedVelocityField: only 2D and 3D.");
      ComputeMinEdgeLength();
   }

   /// Set the current velocity GridFunction for box averaging.
   /// Call at the beginning of each timestep.
   void Update(mfem::GridFunction &u_gf) { u_gf_ = &u_gf; }

   /// Evaluate smoothed velocity at a physical point.
   ///
   /// For each component i, computes the 1D box average:
   ///   ū^i(x) = (1/h) ∫_{x_i - h/2}^{x_i + h/2} u^i(x₁,...,ξ,...,x_d) dξ
   ///
   /// @a x          physical point
   /// @a t          time (unused, for VelocityFunc interface compatibility)
   /// @a elem_hint  starting element for BFS point location
   /// @a v          output velocity vector
   void Evaluate(const mfem::Vector &x, double t,
                 int elem_hint, mfem::Vector &v) const
   {
      (void)t;
      MFEM_VERIFY(u_gf_ != nullptr,
                  "SmoothedVelocityField::Evaluate: call Update() first.");
      v.SetSize(dim_);

      const double half_h = 0.5 * h_min_;

      for (int comp = 0; comp < dim_; ++comp)
      {
         v[comp] = IntegrateBoxComponent(x, comp, half_h, elem_hint);
      }
   }

   /// Create a VelocityFunc callback for use with TraceDeparture.
   VelocityFunc MakeVelocityCallback() const
   {
      return [this](const mfem::Vector &x, double t, int elem_hint,
                    mfem::Vector &v) {
         this->Evaluate(x, t, elem_hint, v);
      };
   }

   /// Access minimum edge length.
   double GetHMin() const { return h_min_; }

private:
   void ComputeMinEdgeLength()
   {
      h_min_ = std::numeric_limits<double>::max();
      mfem::Array<int> verts;
      for (int e = 0; e < mesh_.GetNEdges(); ++e)
      {
         mesh_.GetEdgeVertices(e, verts);
         const double *v0 = mesh_.GetVertex(verts[0]);
         const double *v1 = mesh_.GetVertex(verts[1]);
         double len2 = 0.0;
         for (int d = 0; d < dim_; ++d)
         {
            double diff = v1[d] - v0[d];
            len2 += diff * diff;
         }
         h_min_ = std::min(h_min_, std::sqrt(len2));
      }
      MFEM_VERIFY(h_min_ > 1e-15,
                  "SmoothedVelocityField: degenerate mesh (h_min ≈ 0).");
   }

   /// Compute the box-averaged component of velocity.
   ///
   /// Returns (1/h) ∫_{x_comp - h/2}^{x_comp + h/2} u^comp(x') dξ
   /// using 2-point Gauss-Legendre quadrature on [0,1] mapped to the interval.
   double IntegrateBoxComponent(const mfem::Vector &center,
                                int comp, double half_h,
                                int elem_hint) const
   {
      // 2-point GL on [0,1]
      static const double gl_x[2] = {0.21132486540518712, 0.78867513459481288};
      static const double gl_w[2] = {0.5, 0.5};

      double result = 0.0;
      const double h = 2.0 * half_h;

      mfem::Vector pt(dim_), vel(dim_);
      mfem::IntegrationPoint ip;
      mfem::IsoparametricTransformation eltrans;

      for (int q = 0; q < 2; ++q)
      {
         // Quadrature point: shift center by ξ in direction comp
         for (int d = 0; d < dim_; ++d)
         {
            pt[d] = center[d];
         }
         pt[comp] += -half_h + gl_x[q] * h;

         // Find element containing this point
         int elem = FindElementBFS(mesh_, elem_hint, pt, ip);
         if (elem < 0)
         {
            // Point outside mesh — evaluate at center instead
            for (int d = 0; d < dim_; ++d) { pt[d] = center[d]; }
            elem = FindElementBFS(mesh_, elem_hint, pt, ip);
            if (elem < 0) { continue; }
         }

         // Evaluate velocity at this point
         mesh_.GetElementTransformation(elem, &eltrans);
         eltrans.SetIntPoint(&ip);
         u_gf_->GetVectorValue(eltrans, ip, vel);

         result += gl_w[q] * vel[comp];
      }

      // GL weights sum to 1 on [0,1]; we integrate over length h then
      // divide by h for the average. The factors cancel.
      return result;
   }

   mfem::Mesh &mesh_;
   int dim_;
   double h_min_;
   mfem::GridFunction *u_gf_;
};

#endif // VELOCITY_SMOOTHING_H
