#ifndef SEMI_LAGRANGIAN_ADVECTION_H
#define SEMI_LAGRANGIAN_ADVECTION_H

#include "IntegrateLineCellSplit.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <ctime>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct SemiLagrangianStepStats
{
   bool enable_breakdown = true;
   bool enable_thread_balance = false;

   double trace_departure_s = 0.0;
   double split_line_s = 0.0;
   double interior_integral_s = 0.0;
   double boundary_integral_s = 0.0;
   long long split_calls = 0;
   long long total_segments = 0;

   double edge_thread_min_s = 0.0;
   double edge_thread_avg_s = 0.0;
   double edge_thread_max_s = 0.0;
   double edge_thread_imbalance = 0.0;
   int edge_threads_active = 0;
   long long edge_thread_edges_min = 0;
   long long edge_thread_edges_max = 0;
   double edge_thread_cpu_min_s = 0.0;
   double edge_thread_cpu_avg_s = 0.0;
   double edge_thread_cpu_max_s = 0.0;
   double edge_thread_cpu_util_min = 0.0;
   double edge_thread_cpu_util_avg = 0.0;
   double edge_thread_cpu_util_max = 0.0;

   void Reset()
   {
      trace_departure_s = 0.0;
      split_line_s = 0.0;
      interior_integral_s = 0.0;
      boundary_integral_s = 0.0;
      split_calls = 0;
      total_segments = 0;

      edge_thread_min_s = 0.0;
      edge_thread_avg_s = 0.0;
      edge_thread_max_s = 0.0;
      edge_thread_imbalance = 0.0;
      edge_threads_active = 0;
      edge_thread_edges_min = 0;
      edge_thread_edges_max = 0;
      edge_thread_cpu_min_s = 0.0;
      edge_thread_cpu_avg_s = 0.0;
      edge_thread_cpu_max_s = 0.0;
      edge_thread_cpu_util_min = 0.0;
      edge_thread_cpu_util_avg = 0.0;
      edge_thread_cpu_util_max = 0.0;
   }
};

/// Semi-Lagrangian advection operator for discrete 1-forms (Nédélec elements).
///
/// Implements the pullback-based advection from Tonnon & Hiptmair, 2023
/// (arXiv:2301.04923).  Given ω_h^{n-1} ∈ ND_1, computes
///
///   ω_h^n = I_h( X̄^*_{t,t-τ} ω_h^{n-1} )
///
/// by evaluating line integrals along transported edges:
///
///   DOF(e) = ∫_{X̄(e)} ω_h^{n-1} · dl
///
/// where X̄(e) is the straight line from X(v0) to X(v1), the departure
/// points of the edge vertices obtained by tracing characteristics backward.
///
/// Boundary handling: when the transported edge partially or fully exits the
/// domain, the outside portion is filled using a user-supplied boundary
/// value function g(x, t), integrated along the outside part of the
/// transported edge.
///
/// Template parameter N_gauss: Gauss-Legendre order for sub-segment
/// quadrature.  For ND1 on affine elements, N_gauss=1 (midpoint) is exact.
template <unsigned N_gauss = 1>
class SemiLagrangianAdvection1Form
{
public:
   /// Velocity field: u(x, t) → v, with a BFS start-element hint.
   using VelocityFunc =
       std::function<void(const mfem::Vector &, double, int, mfem::Vector &)>;

   /// Boundary value: g(x, t, bdr_attr) → v.  Evaluated at physical points
   /// outside the mesh domain to supply DOF contributions for the exited
   /// portion of transported edges.  The boundary attribute @a bdr_attr
   /// identifies which part of ∂Ω the edge exited through (0 if unknown).
   using BoundaryFunc =
       std::function<void(const mfem::Vector &, double, int, mfem::Vector &)>;

   /// Construct the operator for a given ND finite element space.
   /// Precomputes mesh adjacency maps and GL quadrature rule.
   /// The FE space must use ND_FECollection of order 1.
   SemiLagrangianAdvection1Form(mfem::FiniteElementSpace &fes)
       : mesh_(*fes.GetMesh()), fes_(fes),
         dim_(mesh_.SpaceDimension())
   {
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
      BuildEdgeToElementMap();
      BuildFaceBdrAttrMap();
   }

   /// BDF1 semi-Lagrangian step (first-order in time):
   ///   omega_new = I_h( X̄^*_{dt} omega_old )
   ///
   /// @a velocity     maps (x, t) → velocity vector.
   /// @a boundary     maps (x, t) → boundary 1-form value for outside portions.
   /// @a t            current time (arrival time t^n).
   /// @a dt           timestep size τ.
   /// @a trace_order  ODE order for characteristic tracing (1=Euler, 2=Heun).
   /// @a omega_new may NOT alias @a omega_old.
   void Apply(const VelocityFunc &velocity,
              const BoundaryFunc &boundary,
              double t, double dt,
              mfem::GridFunction &omega_old,
              mfem::GridFunction &omega_new,
              int trace_order = 1,
              SemiLagrangianStepStats *step_stats = nullptr)
   {
      if (step_stats) { step_stats->Reset(); }
      mfem::Vector dofs_pulled(fes_.GetNDofs());
      ComputePullbackDOFs(velocity, boundary, t, dt,
                          omega_old, dofs_pulled, trace_order, step_stats);

      for (int i = 0; i < fes_.GetNDofs(); ++i)
      {
         omega_new(i) = dofs_pulled(i);
      }
   }

   /// BDF2 semi-Lagrangian step (second-order in time):
   ///   omega_new = 4/3 I_h(X̄^*_{dt} omega_n1) - 1/3 I_h(X̄^*_{2dt} omega_n2)
   ///
   /// @a omega_n1  is ω_h^{n-1}, @a omega_n2  is ω_h^{n-2}.
   void ApplyBDF2(const VelocityFunc &velocity,
                  const BoundaryFunc &boundary,
                  double t, double dt,
                  mfem::GridFunction &omega_n1,
                  mfem::GridFunction &omega_n2,
                  mfem::GridFunction &omega_new,
                  int trace_order = 1,
                  SemiLagrangianStepStats *step_stats = nullptr)
   {
      if (step_stats) { step_stats->Reset(); }
      const int ndofs = fes_.GetNDofs();
      mfem::Vector dofs1(ndofs), dofs2(ndofs);

      ComputePullbackDOFs(velocity, boundary, t, dt,
                          omega_n1, dofs1, trace_order, nullptr);
      ComputePullbackDOFs(velocity, boundary, t, 2.0 * dt,
                          omega_n2, dofs2, trace_order, nullptr);

      for (int i = 0; i < ndofs; ++i)
      {
         omega_new(i) = (4.0 / 3.0) * dofs1(i) - (1.0 / 3.0) * dofs2(i);
      }
   }

   /// Compute the transported DOF value for a single edge.  Useful for
   /// external OpenMP parallelization with per-thread Mesh/GF copies.
   ///
   /// @a d0, @a d1       departure points of the edge vertices.
   /// @a start_elem      BFS starting element.
    /// @a boundary        boundary value function for outside portions
    ///                    (receives boundary attribute for dispatch).
    /// @a t_departure     time at the departure (t^n - dt).
     double ComputeTransportedDOF(
         mfem::Mesh &mesh,
         mfem::GridFunction &gf,
         const mfem::Vector &d0,
         const mfem::Vector &d1,
         int start_elem,
         const BoundaryFunc &boundary,
         double t_departure,
         CellSplitWorkspace &ws,
         double *split_line_s = nullptr,
         double *interior_integral_s = nullptr,
         double *boundary_integral_s = nullptr,
         long long *split_calls = nullptr,
         long long *total_segments = nullptr) const
   {
      const int dim = d0.Size();

      // Split the transported edge into sub-segments within mesh elements.
      // Track which face the line exits through (boundary attribute source).
      int exit_face_fwd = -1;
      std::chrono::steady_clock::time_point split_start;
      if (split_line_s)
      {
         split_start = std::chrono::steady_clock::now();
      }
      auto &segments = ws.split_segments;
      SplitLineIntoSegments(mesh, start_elem, d0, d1, segments,
                            &exit_face_fwd,
                            ws.split_verts,
                            &ws.eltrans, &ws.inv_tr);
      if (split_line_s)
      {
         *split_line_s += std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - split_start)
                              .count();
      }
      if (split_calls) { (*split_calls)++; }
      if (total_segments) { *total_segments += static_cast<long long>(segments.size()); }

      // If no segments found (departure entirely outside), try reversed
      // direction in case d1 is inside but d0 is not
      bool reversed = false;
      int exit_face_rev = -1;
      if (segments.empty())
      {
         if (split_line_s)
         {
            split_start = std::chrono::steady_clock::now();
         }
         SplitLineIntoSegments(mesh, start_elem, d1, d0, segments,
                                &exit_face_rev,
                                ws.split_verts,
                                &ws.eltrans, &ws.inv_tr);
         if (split_line_s)
         {
            *split_line_s += std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - split_start)
                                 .count();
         }
         if (split_calls) { (*split_calls)++; }
         if (total_segments) { *total_segments += static_cast<long long>(segments.size()); }
         reversed = true;
      }

      int exit_face = reversed ? exit_face_rev : exit_face_fwd;

      // Direction vector for the line from d0 to d1
      mfem::Vector direction(dim);
      subtract(d1, d0, direction);

      if (segments.empty())
      {
         // Both endpoints outside — determine boundary attribute via ray
         // intersection from an interior element point toward departure midpoint.
         mfem::Vector dep_mid(dim);
         add(d0, d1, dep_mid);
         dep_mid *= 0.5;
         mfem::Vector origin_inside = ElementCenter(start_elem);
         int bdr_attr = FindNearestBoundaryAttribute(
             origin_inside, dep_mid, start_elem);
         if (bdr_attr == 0)
         {
            bdr_attr = FindNearestBoundaryAttribute(
                origin_inside, d0, start_elem);
         }
         if (bdr_attr == 0)
         {
            bdr_attr = FindNearestBoundaryAttribute(
                origin_inside, d1, start_elem);
         }
         std::chrono::steady_clock::time_point bdr_start;
         if (boundary_integral_s)
         {
            bdr_start = std::chrono::steady_clock::now();
         }
         const double bdr_val = IntegrateBoundaryPortion(boundary, t_departure, bdr_attr,
                                                         d0, direction, 0.0, 1.0);
         if (boundary_integral_s)
         {
            *boundary_integral_s += std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - bdr_start)
                                      .count();
         }
         return bdr_val;
      }

      // Look up boundary attribute from exit face
      int bdr_attr = 0;
      if (exit_face >= 0 && exit_face < (int)face_bdr_attr_.size())
      {
         bdr_attr = face_bdr_attr_[exit_face];
      }

      // Compute coverage: fraction of [0,1] covered by segments
      double coverage = 0.0;
      for (const auto &seg : segments)
      {
         coverage += seg.s_end - seg.s_start;
      }

      // Compute line integral over the interior (covered) portion
      double integral;
      std::chrono::steady_clock::time_point interior_start;
      if (interior_integral_s)
      {
         interior_start = std::chrono::steady_clock::now();
      }
      if (reversed)
      {
         integral = -IntegrateLineTangentialCellSplit(
             mesh, gf, d1, d0, segments, gl_rule_, ws);
      }
      else
      {
         integral = IntegrateLineTangentialCellSplit(
             mesh, gf, d0, d1, segments, gl_rule_, ws);
      }
      if (interior_integral_s)
      {
         *interior_integral_s += std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - interior_start)
                                     .count();
      }

      // For the portion(s) outside the domain, integrate the boundary
      // value function along the uncovered sub-intervals of [0,1].
      if (coverage < 1.0 - 1e-12)
      {
         const double endpoint_tol = 1e-10;
         const bool has_endpoint_gap =
             (segments.front().s_start > endpoint_tol) ||
             (segments.back().s_end < 1.0 - endpoint_tol);

         // If coverage loss is only due to tiny internal split gaps
         // (e.g. line passing through mesh vertices), there is no true
         // outside interval and no boundary contribution to add.
         if (bdr_attr == 0 && !has_endpoint_gap)
         {
            return integral;
         }

         // Robust fallback: if the line left the domain but no valid
         // boundary attribute was recovered from the exit face, infer it
         // from an interior point of a covered segment toward an uncovered
         // endpoint.  This avoids using boundary points as ray origins.
         if (bdr_attr == 0)
         {
            auto MapToOriginalInterval = [reversed](const LineSegment &seg,
                                                    double &s0,
                                                    double &s1)
            {
               if (reversed)
               {
                  s0 = 1.0 - seg.s_end;
                  s1 = 1.0 - seg.s_start;
               }
               else
               {
                  s0 = seg.s_start;
                  s1 = seg.s_end;
               }
            };

            double covered_start = 1.0;
            double covered_end = 0.0;
            int idx_start = -1;
            int idx_end = -1;
            for (int i = 0; i < (int)segments.size(); ++i)
            {
               double s0, s1;
               MapToOriginalInterval(segments[i], s0, s1);
               if (s0 < covered_start)
               {
                  covered_start = s0;
                  idx_start = i;
               }
               if (s1 > covered_end)
               {
                  covered_end = s1;
                  idx_end = i;
               }
            }

            const double start_gap = covered_start;
            const double end_gap = 1.0 - covered_end;
            const bool has_start_gap = start_gap > 1e-12;
            const bool has_end_gap = end_gap > 1e-12;

            mfem::Vector origin_inside(dim), target(dim);
            bool target_is_d0 = false;
            bool target_is_mid = false;
            if (has_start_gap || has_end_gap)
            {
               bool use_start = has_start_gap;
               if (has_start_gap && has_end_gap)
               {
                  use_start = (start_gap >= end_gap);
               }

               int seg_idx = use_start ? idx_start : idx_end;
               double s0, s1;
               MapToOriginalInterval(segments[seg_idx], s0, s1);
               const double s_inside = 0.5 * (s0 + s1);
               for (int d = 0; d < dim; ++d)
               {
                  origin_inside[d] = d0[d] + s_inside * direction[d];
                  target[d] = use_start ? d0[d] : d1[d];
               }
               target_is_d0 = use_start;
            }
            else
            {
               origin_inside = ElementCenter(start_elem);
               add(d0, d1, target);
               target *= 0.5;
               target_is_mid = true;
            }

            bdr_attr = FindNearestBoundaryAttribute(
                origin_inside, target, start_elem);
            if (bdr_attr == 0)
            {
               if (target_is_mid)
               {
                  bdr_attr = FindNearestBoundaryAttribute(
                      origin_inside, d0, start_elem);
                  if (bdr_attr == 0)
                  {
                     bdr_attr = FindNearestBoundaryAttribute(
                         origin_inside, d1, start_elem);
                  }
               }
               else
               {
                  const mfem::Vector &other_target = target_is_d0 ? d1 : d0;
                  bdr_attr = FindNearestBoundaryAttribute(
                      origin_inside, other_target, start_elem);
               }
            }
         }

         // If boundary attribution is still unknown, avoid applying an
         // incorrect boundary value on uncovered intervals.
         if (bdr_attr == 0)
         {
            return integral;
         }

         // Walk through [0,1] and integrate boundary over gaps
         double s_covered = 0.0;
         for (size_t i = 0; i < segments.size(); ++i)
         {
            // Gap before this segment
            if (segments[i].s_start > s_covered + 1e-14)
            {
               double s_gap_start = s_covered;
               double s_gap_end = segments[i].s_start;
               // Map gap back to d0→d1 parameterization
               double s0, s1;
               if (reversed)
               {
                  s0 = 1.0 - s_gap_end;
                  s1 = 1.0 - s_gap_start;
               }
               else
               {
                  s0 = s_gap_start;
                  s1 = s_gap_end;
               }
               std::chrono::steady_clock::time_point bdr_start;
               if (boundary_integral_s)
               {
                  bdr_start = std::chrono::steady_clock::now();
               }
               integral += IntegrateBoundaryPortion(
                   boundary, t_departure, bdr_attr, d0, direction, s0, s1);
               if (boundary_integral_s)
               {
                  *boundary_integral_s += std::chrono::duration<double>(
                                            std::chrono::steady_clock::now() - bdr_start)
                                            .count();
               }
            }
            s_covered = segments[i].s_end;
         }
         // Gap after the last segment
         if (s_covered < 1.0 - 1e-14)
         {
            double s0, s1;
            if (reversed)
            {
               s0 = 0.0;
               s1 = 1.0 - s_covered;
            }
            else
            {
               s0 = s_covered;
               s1 = 1.0;
            }
            std::chrono::steady_clock::time_point bdr_start;
            if (boundary_integral_s)
            {
               bdr_start = std::chrono::steady_clock::now();
            }
            integral += IntegrateBoundaryPortion(
                boundary, t_departure, bdr_attr, d0, direction, s0, s1);
            if (boundary_integral_s)
            {
               *boundary_integral_s += std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() - bdr_start)
                                         .count();
            }
         }
      }

      return integral;
   }

   /// Access the precomputed edge-to-element map (returns first adjacent element).
   int EdgeStartElement(int edge) const { return edge_to_elems_.GetRow(edge)[0]; }

   /// Access the GL quadrature rule.
   const GaussLegendreRule<N_gauss> &GetRule() const { return gl_rule_; }

private:
   /// Build edge → {elements} table by transposing the element → edge table.
   void BuildEdgeToElementMap()
   {
      const mfem::Table &el2edge = mesh_.ElementToEdgeTable();
      const int n_edges = mesh_.GetNEdges();
      mfem::Transpose(el2edge, edge_to_elems_, n_edges);
   }

   /// Build a map from mesh face index to boundary attribute.
   /// Interior faces map to 0.  Boundary faces map to their attribute (≥ 1).
   void BuildFaceBdrAttrMap()
   {
      mfem::Array<int> face_to_bdr = mesh_.GetFaceToBdrElMap();
      int nfaces = (dim_ == 3) ? mesh_.GetNFaces() : mesh_.GetNEdges();
      face_bdr_attr_.assign(nfaces, 0);
      for (int i = 0; i < face_to_bdr.Size(); ++i)
      {
         if (face_to_bdr[i] >= 0)
         {
            face_bdr_attr_[i] = mesh_.GetBdrAttribute(face_to_bdr[i]);
         }
      }
   }

   /// Physical-space center of element @a elem.
   mfem::Vector ElementCenter(int elem) const
   {
      mfem::IsoparametricTransformation eltrans;
      mesh_.GetElementTransformation(elem, &eltrans);
      const mfem::IntegrationPoint &ip =
          mfem::Geometries.GetCenter(mesh_.GetElementGeometry(elem));
      mfem::Vector center(dim_);
      eltrans.Transform(ip, center);
      return center;
   }

   /// Determine boundary attribute by shooting a ray from @a origin (inside
   /// the mesh) toward @a target (outside).  Returns the attribute of the
   /// first boundary face hit, or 0 if none found.
   int FindNearestBoundaryAttribute(
       const mfem::Vector &origin,
       const mfem::Vector &target,
       int start_elem) const
   {
      int exit_face = -1;
      SplitLineIntoSegments(mesh_, start_elem, origin, target, &exit_face);
      if (exit_face >= 0 && exit_face < (int)face_bdr_attr_.size())
      {
         return face_bdr_attr_[exit_face];
      }
      return 0;
   }

   /// Integrate the boundary value function along the sub-interval
   /// [s0, s1] of the line x(s) = base + s * dir, using the same
   /// Gauss-Legendre rule as for interior segments.
   ///
   /// @a bdr_attr is the boundary attribute of the exit face (passed
   /// through to the BoundaryFunc for dispatch).
   ///
   /// Returns ∫_{s0}^{s1} g(x(s), t, bdr_attr) · dir ds.
   double IntegrateBoundaryPortion(
       const BoundaryFunc &boundary, double t, int bdr_attr,
       const mfem::Vector &base, const mfem::Vector &dir,
       double s0, double s1) const
   {
      const int dim = base.Size();
      double len = s1 - s0;
      if (len < 1e-15) { return 0.0; }

      double result = 0.0;
      mfem::Vector pt(dim), val(dim);

      for (int q = 0; q < gl_rule_.num_points; ++q)
      {
         double s = s0 + gl_rule_.nodes[q] * len;
         for (int d = 0; d < dim; ++d)
         {
            pt[d] = base[d] + s * dir[d];
         }
         boundary(pt, t, bdr_attr, val);
         double dot = 0.0;
         for (int d = 0; d < dim; ++d)
         {
            dot += val[d] * dir[d];
         }
         result += gl_rule_.weights[q] * len * dot;
      }
      return result;
   }

   /// Trace a point backward along the characteristic curve by time dt.
   /// @a order = 1: explicit Euler,  d = x - dt * u(x, t).
   /// @a order = 2: Heun's method,
   ///   d = x - dt/2 * (u(x, t) + u(x - dt*u(x, t), t - dt)).
   static void TraceDeparture(const VelocityFunc &velocity,
                               double t, double dt,
                               const mfem::Vector &x, mfem::Vector &d,
                               int order,
                               int start_elem_hint)
   {
      const int dim = x.Size();
      d.SetSize(dim);

      mfem::Vector vel(dim);
      velocity(x, t, start_elem_hint, vel);

      if (order == 1)
      {
         for (int i = 0; i < dim; ++i)
         {
            d[i] = x[i] - dt * vel[i];
         }
      }
      else
      {
         // Predictor
         for (int i = 0; i < dim; ++i)
         {
            d[i] = x[i] - dt * vel[i];
         }
         // Corrector: evaluate velocity at departure time t - dt
         mfem::Vector vel2(dim);
         velocity(d, t - dt, start_elem_hint, vel2);
         for (int i = 0; i < dim; ++i)
         {
            d[i] = x[i] - 0.5 * dt * (vel[i] + vel2[i]);
         }
      }
   }

   static double ThreadCPUSeconds()
   {
      struct timespec ts;
      if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0)
      {
         return 0.0;
      }
      return static_cast<double>(ts.tv_sec) +
             1e-9 * static_cast<double>(ts.tv_nsec);
   }

   /// Compute the pullback DOF values for all edges.
   void ComputePullbackDOFs(const VelocityFunc &velocity,
                            const BoundaryFunc &boundary,
                            double t, double dt,
                            mfem::GridFunction &gf_old,
                            mfem::Vector &dof_values,
                            int trace_order,
                            SemiLagrangianStepStats *step_stats)
   {
      const int n_edges = mesh_.GetNEdges();
      double t_departure = t - dt;
      const bool collect_breakdown =
          (step_stats != nullptr && step_stats->enable_breakdown);
      const bool collect_thread_balance =
          (step_stats != nullptr && step_stats->enable_thread_balance);

      double trace_departure_s = 0.0;
      double split_line_s = 0.0;
      double interior_integral_s = 0.0;
      double boundary_integral_s = 0.0;
      long long split_calls = 0;
      long long total_segments = 0;

#ifdef _OPENMP
      const int max_threads = omp_get_max_threads();
#else
      const int max_threads = 1;
#endif
      std::vector<double> edge_loop_time_s;
      std::vector<double> edge_loop_cpu_s;
      std::vector<long long> edge_loop_edges;
      if (collect_thread_balance)
      {
         edge_loop_time_s.assign(max_threads, 0.0);
         edge_loop_cpu_s.assign(max_threads, 0.0);
         edge_loop_edges.assign(max_threads, 0);
      }

#pragma omp parallel reduction(+:trace_departure_s,split_line_s,interior_integral_s,boundary_integral_s,split_calls,total_segments)
      {
         CellSplitWorkspace ws;
         mfem::Array<int> verts;
         mfem::Array<int> dofs;
         mfem::Vector v0(dim_), v1(dim_);
         mfem::Vector dep0(dim_), dep1(dim_);

         int tid = 0;
#ifdef _OPENMP
         tid = omp_get_thread_num();
#endif

         std::chrono::steady_clock::time_point edge_loop_start;
         double edge_cpu_start_s = 0.0;
         if (collect_thread_balance)
         {
            edge_loop_start = std::chrono::steady_clock::now();
            edge_cpu_start_s = ThreadCPUSeconds();
         }
         long long local_edge_count = 0;

#pragma omp for schedule(runtime)
         for (int e = 0; e < n_edges; ++e)
         {
            // Get edge vertices (v0 < v1 in MFEM convention)
            mesh_.GetEdgeVertices(e, verts);
            const int v0_id = verts[0];
            const int v1_id = verts[1];

            const double *v0_ptr = mesh_.GetVertex(v0_id);
            const double *v1_ptr = mesh_.GetVertex(v1_id);

            for (int d = 0; d < dim_; ++d)
            {
               v0[d] = v0_ptr[d];
               v1[d] = v1_ptr[d];
            }

            // Trace vertices backward along characteristics.
            // Use edge-local element hint instead of a global seed.
            const int start_hint = EdgeStartElement(e);
            std::chrono::steady_clock::time_point trace_start;
            if (collect_breakdown)
            {
               trace_start = std::chrono::steady_clock::now();
            }
            TraceDeparture(velocity, t, dt, v0, dep0, trace_order, start_hint);
            TraceDeparture(velocity, t, dt, v1, dep1, trace_order, start_hint);
            if (collect_breakdown)
            {
               trace_departure_s += std::chrono::duration<double>(
                                        std::chrono::steady_clock::now() - trace_start)
                                        .count();
            }

            // Get DOF index for this edge
            fes_.GetEdgeDofs(e, dofs);

            // Compute transported DOF
            double new_dof = ComputeTransportedDOF(
                mesh_, gf_old, dep0, dep1, start_hint,
                boundary, t_departure, ws,
                collect_breakdown ? &split_line_s : nullptr,
                collect_breakdown ? &interior_integral_s : nullptr,
                collect_breakdown ? &boundary_integral_s : nullptr,
                collect_breakdown ? &split_calls : nullptr,
                collect_breakdown ? &total_segments : nullptr);

            dof_values(dofs[0]) = new_dof;
            ++local_edge_count;
         }

         if (collect_thread_balance)
         {
            edge_loop_time_s[tid] = std::chrono::duration<double>(
                                        std::chrono::steady_clock::now() - edge_loop_start)
                                        .count();
            edge_loop_cpu_s[tid] = ThreadCPUSeconds() - edge_cpu_start_s;
            edge_loop_edges[tid] = local_edge_count;
         }
      }

      if (collect_breakdown)
      {
         step_stats->trace_departure_s = trace_departure_s;
         step_stats->split_line_s = split_line_s;
         step_stats->interior_integral_s = interior_integral_s;
         step_stats->boundary_integral_s = boundary_integral_s;
         step_stats->split_calls = split_calls;
         step_stats->total_segments = total_segments;
      }

      if (collect_thread_balance)
      {
         double t_min = std::numeric_limits<double>::infinity();
         double t_max = 0.0;
         double t_sum = 0.0;
         double cpu_min = std::numeric_limits<double>::infinity();
         double cpu_max = 0.0;
         double cpu_sum = 0.0;
         double util_min = std::numeric_limits<double>::infinity();
         double util_max = 0.0;
         double util_sum = 0.0;
         long long e_min = std::numeric_limits<long long>::max();
         long long e_max = 0;
         int active_threads = 0;

         for (int i = 0; i < max_threads; ++i)
         {
            if (edge_loop_edges[i] <= 0) { continue; }
            ++active_threads;
            t_min = std::min(t_min, edge_loop_time_s[i]);
            t_max = std::max(t_max, edge_loop_time_s[i]);
            t_sum += edge_loop_time_s[i];
            cpu_min = std::min(cpu_min, edge_loop_cpu_s[i]);
            cpu_max = std::max(cpu_max, edge_loop_cpu_s[i]);
            cpu_sum += edge_loop_cpu_s[i];
            const double util = (edge_loop_time_s[i] > 0.0)
                                    ? (edge_loop_cpu_s[i] / edge_loop_time_s[i])
                                    : 0.0;
            util_min = std::min(util_min, util);
            util_max = std::max(util_max, util);
            util_sum += util;
            e_min = std::min(e_min, edge_loop_edges[i]);
            e_max = std::max(e_max, edge_loop_edges[i]);
         }

         if (active_threads > 0)
         {
            step_stats->edge_thread_min_s = t_min;
            step_stats->edge_thread_max_s = t_max;
            step_stats->edge_thread_avg_s = t_sum / static_cast<double>(active_threads);
            step_stats->edge_thread_imbalance =
                (step_stats->edge_thread_avg_s > 0.0)
                    ? (step_stats->edge_thread_max_s / step_stats->edge_thread_avg_s)
                    : 0.0;
            step_stats->edge_threads_active = active_threads;
            step_stats->edge_thread_edges_min = e_min;
            step_stats->edge_thread_edges_max = e_max;
            step_stats->edge_thread_cpu_min_s = cpu_min;
            step_stats->edge_thread_cpu_avg_s =
                cpu_sum / static_cast<double>(active_threads);
            step_stats->edge_thread_cpu_max_s = cpu_max;
            step_stats->edge_thread_cpu_util_min = util_min;
            step_stats->edge_thread_cpu_util_avg =
                util_sum / static_cast<double>(active_threads);
            step_stats->edge_thread_cpu_util_max = util_max;
         }
      }
   }

   mfem::Mesh &mesh_;
   mfem::FiniteElementSpace &fes_;
   int dim_;
   GaussLegendreRule<N_gauss> gl_rule_;
   mfem::Table edge_to_elems_;
   std::vector<int> face_bdr_attr_;  ///< face index → boundary attribute (0=interior)
};

#endif // SEMI_LAGRANGIAN_ADVECTION_H
