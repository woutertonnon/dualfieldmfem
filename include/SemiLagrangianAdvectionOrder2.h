#ifndef SEMI_LAGRANGIAN_ADVECTION_ORDER2_H
#define SEMI_LAGRANGIAN_ADVECTION_ORDER2_H

#include "IntegrateLineCellSplit.h"
#include "RapettiProjection.h"
#include "SemiLagrangianAdvection.h"
#include "SmallEdgeEnumeration.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

/// Second-order semi-Lagrangian advection operator for discrete 1-forms
/// using Rapetti small-edge interpolation.
///
/// Implements the pullback-based advection from Tonnon & Hiptmair (2023),
/// Equation 34.  Given ω_h^{n-1} ∈ ND₂, computes:
///
///   ω̃ = I_{h,2}( X̄*_{τ} ω_h^{n-1} )
///
/// by evaluating line integrals along transported small edges (Rapetti DOFs),
/// projecting via the I_{h,2} operator, and converting to MFEM ND₂ DOFs.
///
/// The algorithm for each element K:
///   1. Enumerate all N_small small edges of K (SmallEdgeTopology)
///   2. Trace each small edge's 2 endpoints backward via Heun
///      (using dihedral-averaged velocity at arrival points)
///   3. Compute line integral along each transported small edge using
///      ComputeTransportedDOF() (reused from first-order code)
///   4. Apply I_{h,2} projection: RapettiProjector
///   5. Convert to ND₂ DOFs: RapettiToND2Converter
///   6. Scatter to global GridFunction (with MFEM orientation signs)
///
/// Template parameter N_gauss: Gauss-Legendre order for sub-segment
/// quadrature.  N_gauss=2 is recommended for second-order accuracy.
template <unsigned N_gauss = 2>
class SemiLagrangianAdvection1FormOrder2
{
public:
   using VelocityFunc =
       std::function<void(const mfem::Vector &, double, int, mfem::Vector &)>;

   using BoundaryFunc =
       std::function<void(const mfem::Vector &, double, int, mfem::Vector &)>;

   /// Construct the operator for a given ND₂ finite element space.
   SemiLagrangianAdvection1FormOrder2(mfem::FiniteElementSpace &fes)
       : mesh_(*fes.GetMesh()), fes_(fes),
         dim_(mesh_.SpaceDimension()),
         topo_(mesh_),
         projector_(dim_),
         converter_(dim_),
         first_order_op_(fes)
   {
      // Rapetti small-edge framework requires simplicial elements
      mfem::Geometry::Type expected = (dim_ == 2) ? mfem::Geometry::TRIANGLE
                                                   : mfem::Geometry::TETRAHEDRON;
      MFEM_VERIFY(mesh_.GetNE() > 0 &&
                  mesh_.GetElementBaseGeometry(0) == expected,
                  "SemiLagrangianAdvection1FormOrder2 requires simplicial "
                  "meshes (triangles in 2D, tetrahedra in 3D). "
                  "Hexahedral/quad meshes are not supported.");

      n_small_ = topo_.NumSmallEdgesPerElement();
      n_nd2_local_ = converter_.NumND2();

      // Build element-to-element table for BFS point location
      mesh_.ElementToElementTable();
      if (dim_ == 3)
      {
         mfem::Array<int> faces, ori;
         for (int el = 0; el < mesh_.GetNE(); ++el)
         {
            mesh_.GetElementFaces(el, faces, ori);
         }
      }
      mesh_.GetEdgeVertexTable();
   }

   /// BDF1 semi-Lagrangian step (first-order in time, second-order in space):
   ///   omega_new = I_{h,2}( X̄*_{dt} omega_old )
   ///
   /// @a velocity     velocity callback for Heun corrector at departure point.
   /// @a boundary     boundary 1-form value for outside portions.
   /// @a t            current time (arrival time t^n).
   /// @a dt           timestep size τ.
   /// @a omega_old    input solution ω^{n-1} in ND₂ space.
   /// @a omega_new    output solution (may NOT alias omega_old).
   /// @a trace_order  ODE order for characteristic tracing (2=Heun recommended).
   /// @a velocity_gf  optional velocity GridFunction for dihedral averaging
   ///                 at arrival points. If nullptr, velocity callback is used
   ///                 directly (no dihedral averaging). In self-advection
   ///                 (hcurl Navier-Stokes), pass &omega_old.
   void Apply(const VelocityFunc &velocity,
              const BoundaryFunc &boundary,
              double t, double dt,
              mfem::GridFunction &omega_old,
              mfem::GridFunction &omega_new,
              int trace_order = 2,
              mfem::GridFunction *velocity_gf = nullptr)
   {
      omega_new = 0.0;
      ComputePullbackND2DOFs(velocity, boundary, t, dt,
                             omega_old, omega_new, trace_order, velocity_gf);
   }

   /// BDF2 semi-Lagrangian step (second-order in time and space).
   ///
   /// Computes two pullbacks and combines them for BDF2:
   ///   ω̃₁ = I_{h,2}(X̄*_{τ} ω^{n-1})
   ///   ω̃₂ = I_{h,2}(X̄*_{2τ} ω^{n-2})
   ///
   /// The caller uses these for the BDF2 RHS:
   ///   RHS = (4/(2τ))M·ω̃₁ - (1/(2τ))M·ω̃₂ + f
   ///
   /// @a omega_n1       input ω^{n-1}
   /// @a omega_n2       input ω^{n-2}
   /// @a omega_tilde_1  output I_{h,2}(X̄*_{τ} ω^{n-1})
   /// @a omega_tilde_2  output I_{h,2}(X̄*_{2τ} ω^{n-2})
   /// @a velocity_gf    velocity GF for dihedral averaging (both pullbacks
   ///                   use the same velocity field, typically ω^{n-1}).
   void ApplyBDF2(const VelocityFunc &velocity,
                  const BoundaryFunc &boundary,
                  double t, double dt,
                  mfem::GridFunction &omega_n1,
                  mfem::GridFunction &omega_n2,
                  mfem::GridFunction &omega_tilde_1,
                  mfem::GridFunction &omega_tilde_2,
                  int trace_order = 2,
                  mfem::GridFunction *velocity_gf = nullptr)
   {
      omega_tilde_1 = 0.0;
      omega_tilde_2 = 0.0;

      ComputePullbackND2DOFs(velocity, boundary, t, dt,
                             omega_n1, omega_tilde_1, trace_order, velocity_gf);
      ComputePullbackND2DOFs(velocity, boundary, t, 2.0 * dt,
                             omega_n2, omega_tilde_2, trace_order, velocity_gf);
   }

   /// Access the underlying first-order operator (for ComputeTransportedDOF).
   SemiLagrangianAdvection1Form<N_gauss> &GetFirstOrderOp()
   {
      return first_order_op_;
   }

private:
   /// Compute the pullback of a 1-form via transported small edges and project
   /// into ND₂ DOF space.
   ///
   /// Uses dihedral-angle-weighted velocity averaging at arrival points
   /// (small edge endpoints on mesh edges) for the Heun predictor, and
   /// direct grid function evaluation at the predicted departure point
   /// for the Heun corrector.
   ///
   /// Parallelized over elements with OpenMP. For each element:
   /// 1. Compute physical coordinates of all small edge endpoints
   /// 2. Look up dihedral-averaged velocity at each endpoint
   /// 3. Trace departure points backward along characteristics
   /// 4. Compute line integrals along transported small edges
   /// 5. Apply I_{h,2} projection
   /// 6. Convert Rapetti → ND₂ DOFs
   /// 7. Scatter to global GridFunction
   ///
   /// @a velocity_gf  GridFunction for dihedral averaging at arrival
   ///                 points (typically ω^{n-1}, the velocity 1-form).
   ///                 If nullptr, the velocity callback is used directly
   ///                 (no dihedral averaging).
   void ComputePullbackND2DOFs(const VelocityFunc &velocity,
                               const BoundaryFunc &boundary,
                               double t, double dt,
                               mfem::GridFunction &gf_old,
                               mfem::GridFunction &omega_new,
                               int trace_order,
                               mfem::GridFunction *velocity_gf)
   {
      const int n_elem = mesh_.GetNE();
      const double t_departure = t - dt;

      // Precompute dihedral-averaged velocities at all mesh edge
      // vertices and midpoints.  Row layout per edge e:
      //   [vel_v0[0..dim), vel_mid[0..dim), vel_v1[0..dim)]
      // where v0 < v1 in global vertex numbering.
      const bool use_dihedral = (velocity_gf != nullptr);
      mfem::DenseMatrix edge_vel;
      if (use_dihedral)
      {
         first_order_op_.PrecomputeEdgeDihedralVelocities(*velocity_gf, edge_vel);
         // Force-initialize the vertex-pair → edge lookup (not thread-safe
         // on first call, so must happen before the parallel region).
         first_order_op_.GetVertexPairToEdgeMap();
      }

#pragma omp parallel
      {
         CellSplitWorkspace ws;

         // Per-thread local storage
         std::vector<mfem::Vector> se_a(n_small_), se_b(n_small_);
         std::vector<mfem::Vector> dep_a(n_small_), dep_b(n_small_);
         for (int k = 0; k < n_small_; ++k)
         {
            se_a[k].SetSize(dim_);
            se_b[k].SetSize(dim_);
            dep_a[k].SetSize(dim_);
            dep_b[k].SetSize(dim_);
         }

         std::vector<double> raw_integrals(n_small_);
         std::vector<double> rapetti_dofs(n_small_);
         std::vector<double> nd2_dofs(n_nd2_local_);

         mfem::Array<int> elem_dofs;
         mfem::Array<int> elem_verts;
         mfem::DofTransformation doftrans_local;
         mfem::Vector vel_a(dim_), vel_b(dim_);

#pragma omp for schedule(runtime)
         for (int el = 0; el < n_elem; ++el)
         {
            // Step 1: Get physical endpoints of all small edges
            mesh_.GetElementVertices(el, elem_verts);
            for (int k = 0; k < n_small_; ++k)
            {
               topo_.GetSmallEdgeEndpoints(el, k, se_a[k], se_b[k]);
            }

            // Step 2: Trace departure points backward along characteristics.
            // If dihedral averaging is enabled, use precomputed edge velocities
            // at arrival points (Heun predictor); the velocity callback handles
            // the corrector step at the predicted departure point.
            int start_hint = el;
            for (int k = 0; k < n_small_; ++k)
            {
               const mfem::Vector *vel_a_ptr = nullptr;
               const mfem::Vector *vel_b_ptr = nullptr;

               if (use_dihedral)
               {
                  const LocalSmallEdge &se = topo_.GetLocalSmallEdge(k);
                  const int g_i  = elem_verts[se.local_vertex];
                  const int g_a  = elem_verts[se.edge_v0];
                  const int g_b  = elem_verts[se.edge_v1];

                  LookupDihedralVelocity(g_i, g_a, g_b, edge_vel, vel_a);
                  LookupDihedralVelocity(g_i, g_b, g_a, edge_vel, vel_b);
                  vel_a_ptr = &vel_a;
                  vel_b_ptr = &vel_b;
               }

               SemiLagrangianAdvection1Form<N_gauss>::TraceDeparture(
                   velocity, nullptr, t, dt,
                   se_a[k], dep_a[k], trace_order, start_hint, 2, vel_a_ptr);
               SemiLagrangianAdvection1Form<N_gauss>::TraceDeparture(
                   velocity, nullptr, t, dt,
                   se_b[k], dep_b[k], trace_order, start_hint, 2, vel_b_ptr);
            }

            // Step 3: Compute line integrals along transported small edges
            for (int k = 0; k < n_small_; ++k)
            {
               raw_integrals[k] = first_order_op_.ComputeTransportedDOF(
                   mesh_, gf_old,
                   dep_a[k], dep_b[k],
                   start_hint, boundary, t_departure, ws);
            }

            // Step 4: I_{h,2} projection
            projector_.ProjectElement(raw_integrals.data(), rapetti_dofs.data());

            // Step 5: Convert Rapetti → ND₂ DOFs
            converter_.Convert(rapetti_dofs.data(), nd2_dofs.data());

            // Step 6: Scatter to global GridFunction with orientation signs
            // Use the thread-local DofTransformation overload to avoid
            // racing on the shared mutable FiniteElementSpace::DoFTrans.
            fes_.GetElementDofs(el, elem_dofs, doftrans_local);
            assert(elem_dofs.Size() == n_nd2_local_);

            // Apply DofTransformation: maps reference-local DOFs to the
            // element's actual DOF orientation (matching MFEM's convention
            // for Project() → TransformPrimal() → SetSubVector()).
            mfem::Vector nd2_vec(nd2_dofs.data(), n_nd2_local_);
            if (!doftrans_local.IsIdentity())
            {
               doftrans_local.TransformPrimal(nd2_vec);
            }

            for (int i = 0; i < elem_dofs.Size(); ++i)
            {
               int gi = elem_dofs[i];
               double sign = 1.0;
               if (gi < 0)
               {
                  gi = -1 - gi;
                  sign = -1.0;
               }
               // Last-write-wins: shared DOFs on element boundaries produce
               // identical values from adjacent elements (the transported
               // small edges on shared mesh edges are deterministic).
               omega_new(gi) = sign * nd2_vec(i);
            }
         }
      }
   }

   /// Look up the dihedral-averaged velocity at point 0.5*(v_{g1} + v_{g2}).
   ///
   /// If g1 != g2: point is midpoint of mesh edge (g1, g2).
   /// If g1 == g2: point is vertex g1; use the big edge (g2, g_other)
   ///              for the dihedral average lookup.
   ///
   /// @a g1, g2    global vertex indices whose midpoint is the query point.
   /// @a g_other   the other big-edge endpoint (used only when g1 == g2).
   /// @a edge_vel  precomputed dihedral velocities (from PrecomputeEdgeDihedralVelocities).
   /// @a vel_out   output velocity vector.
   void LookupDihedralVelocity(int g1, int g2, int g_other,
                                const mfem::DenseMatrix &edge_vel,
                                mfem::Vector &vel_out)
   {
      if (g1 != g2)
      {
         // Point is midpoint of mesh edge (g1, g2)
         int eid = first_order_op_.FindGlobalEdge(g1, g2);
         assert(eid >= 0);
         for (int d = 0; d < dim_; ++d)
         {
            vel_out[d] = edge_vel(eid, dim_ + d);
         }
      }
      else
      {
         // Point is vertex g1; use big edge (g2, g_other) for lookup
         int eid = first_order_op_.FindGlobalEdge(g2, g_other);
         assert(eid >= 0);
         // Determine if g1 is v0 or v1 of the global edge
         // (v0 = min(g2, g_other), v1 = max(g2, g_other))
         if (g1 == std::min(g2, g_other))
         {
            for (int d = 0; d < dim_; ++d)
            {
               vel_out[d] = edge_vel(eid, d);
            }
         }
         else
         {
            for (int d = 0; d < dim_; ++d)
            {
               vel_out[d] = edge_vel(eid, 2 * dim_ + d);
            }
         }
      }
   }

   mfem::Mesh &mesh_;
   mfem::FiniteElementSpace &fes_;
   int dim_;
   int n_small_;
   int n_nd2_local_;

   SmallEdgeTopology topo_;
   RapettiProjector projector_;
   RapettiToND2Converter converter_;

   /// First-order operator, used for ComputeTransportedDOF and TraceDeparture.
   SemiLagrangianAdvection1Form<N_gauss> first_order_op_;
};

#endif // SEMI_LAGRANGIAN_ADVECTION_ORDER2_H
