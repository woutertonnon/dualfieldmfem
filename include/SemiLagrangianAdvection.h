#ifndef SEMI_LAGRANGIAN_ADVECTION_H
#define SEMI_LAGRANGIAN_ADVECTION_H

#include "IntegrateLineCellSplit.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <vector>

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
   /// Velocity field: u(x, t) → v.
   using VelocityFunc =
       std::function<void(const mfem::Vector &, double, mfem::Vector &)>;

   /// Boundary value: g(x, t, bdr_attr) → v.  Evaluated at physical points
   /// outside the mesh domain to supply DOF contributions for the exited
   /// portion of transported edges.  The boundary attribute @a bdr_attr
   /// identifies which part of ∂Ω the edge exited through (0 if unknown).
   using BoundaryFunc =
       std::function<void(const mfem::Vector &, double, int, mfem::Vector &)>;

   /// Construct the operator for a given ND finite element space.
   /// Precomputes the edge-to-element map, dihedral angles, and GL quadrature
   /// rule.  The FE space must use ND_FECollection of order 1.
   SemiLagrangianAdvection1Form(mfem::FiniteElementSpace &fes)
       : mesh_(*fes.GetMesh()), fes_(fes),
         dim_(mesh_.SpaceDimension())
   {
      mesh_.ElementToElementTable();
      BuildEdgeToElementMap();
      BuildDihedralAngles();
      BuildEdgeBdrAttrMap();
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
   /// @a velocity_gf  if non-null, vertex velocities are computed as
   ///                 dihedral-angle-weighted averages over adjacent elements
   ///                 instead of using the VelocityFunc callback.
   /// @a omega_new may NOT alias @a omega_old.
   void Apply(const VelocityFunc &velocity,
              const BoundaryFunc &boundary,
              double t, double dt,
              mfem::GridFunction &omega_old,
              mfem::GridFunction &omega_new,
              int trace_order = 1,
              mfem::GridFunction *velocity_gf = nullptr)
   {
      mfem::Vector dofs_pulled(fes_.GetNDofs());
      ComputePullbackDOFs(velocity, boundary, t, dt,
                          omega_old, dofs_pulled, trace_order, velocity_gf);

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
                  mfem::GridFunction *velocity_gf = nullptr)
   {
      const int ndofs = fes_.GetNDofs();
      mfem::Vector dofs1(ndofs), dofs2(ndofs);

      ComputePullbackDOFs(velocity, boundary, t, dt,
                          omega_n1, dofs1, trace_order, velocity_gf);
      ComputePullbackDOFs(velocity, boundary, t, 2.0 * dt,
                          omega_n2, dofs2, trace_order, velocity_gf);

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
        CellSplitWorkspace &ws) const
   {
      const int dim = d0.Size();

      // Split the transported edge into sub-segments within mesh elements.
      // Track which face the line exits through (boundary attribute source).
      int exit_face_fwd = -1;
      auto segments = SplitLineIntoSegments(mesh, start_elem, d0, d1,
                                            &exit_face_fwd);

      // If no segments found (departure entirely outside), try reversed
      // direction in case d1 is inside but d0 is not
      bool reversed = false;
      int exit_face_rev = -1;
      if (segments.empty())
      {
         segments = SplitLineIntoSegments(mesh, start_elem, d1, d0,
                                          &exit_face_rev);
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
         return IntegrateBoundaryPortion(boundary, t_departure, bdr_attr,
                                         d0, direction, 0.0, 1.0);
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
      if (reversed)
      {
         integral = -IntegrateLineTangentialCellSplit(
             mesh, gf, start_elem, d1, d0, gl_rule_, ws);
      }
      else
      {
         integral = IntegrateLineTangentialCellSplit(
             mesh, gf, start_elem, d0, d1, gl_rule_, ws);
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
               integral += IntegrateBoundaryPortion(
                   boundary, t_departure, bdr_attr, d0, direction, s0, s1);
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
            integral += IntegrateBoundaryPortion(
                boundary, t_departure, bdr_attr, d0, direction, s0, s1);
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

   /// Precompute the dihedral angle (3D) or equal weight (2D) for every
   /// (edge, adjacent-element) pair.  The flat array is indexed in parallel
   /// with the connections of edge_to_elems_.
   void BuildDihedralAngles()
   {
      const int n_edges = mesh_.GetNEdges();
      const int n_conn = edge_to_elems_.Size_of_connections();
      edge_elem_dihedral_angles_.resize(n_conn);

      if (dim_ == 2)
      {
         // In 2D the dihedral angle concept does not apply; use equal weights.
         std::fill(edge_elem_dihedral_angles_.begin(),
                   edge_elem_dihedral_angles_.end(), 1.0);
         return;
      }

      // 3D: compute the dihedral angle at each edge within each element.
      for (int e = 0; e < n_edges; ++e)
      {
         mfem::Array<int> edge_verts;
         mesh_.GetEdgeVertices(e, edge_verts);
         const int gv0 = edge_verts[0], gv1 = edge_verts[1];

         // Physical coords of edge endpoints
         const double *p0 = mesh_.GetVertex(gv0);
         const double *p1 = mesh_.GetVertex(gv1);
         mfem::Vector edge_dir(3);
         for (int d = 0; d < 3; ++d) { edge_dir[d] = p1[d] - p0[d]; }

         const int row_off = edge_to_elems_.GetI()[e];
         const int n_adj = edge_to_elems_.RowSize(e);
         const int *elems = edge_to_elems_.GetRow(e);

         for (int k = 0; k < n_adj; ++k)
         {
            edge_elem_dihedral_angles_[row_off + k] =
                ComputeDihedralAngle3D(elems[k], gv0, gv1, p0, edge_dir);
         }
      }
   }

   /// Compute the dihedral angle at an edge (gv0–gv1) within a 3D element.
   /// @a p0  physical coordinates of gv0.
   /// @a edge_dir  = p1 - p0 (physical edge direction).
   double ComputeDihedralAngle3D(int elem, int gv0, int gv1,
                                 const double *p0,
                                 const mfem::Vector &edge_dir) const
   {
      // Find the two element vertices NOT on the edge.
      mfem::Array<int> ev;
      mesh_.GetElementVertices(elem, ev);
      int opp[2];
      int n_opp = 0;
      for (int j = 0; j < ev.Size(); ++j)
      {
         if (ev[j] != gv0 && ev[j] != gv1) { opp[n_opp++] = ev[j]; }
      }

      // Project both opposite vertices onto the plane ⊥ to the edge through p0.
      mfem::Vector pa(3), pb(3);
      const double *a = mesh_.GetVertex(opp[0]);
      const double *b = mesh_.GetVertex(opp[1]);
      const double e_dot_e = edge_dir * edge_dir;
      for (int d = 0; d < 3; ++d)
      {
         pa[d] = a[d] - p0[d];
         pb[d] = b[d] - p0[d];
      }
      const double proj_a = (pa * edge_dir) / e_dot_e;
      const double proj_b = (pb * edge_dir) / e_dot_e;
      for (int d = 0; d < 3; ++d)
      {
         pa[d] -= proj_a * edge_dir[d];
         pb[d] -= proj_b * edge_dir[d];
      }

      const double na = pa.Norml2(), nb = pb.Norml2();
      if (na < 1e-15 || nb < 1e-15) { return 0.0; }
      double cos_angle = (pa * pb) / (na * nb);
      cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
      return std::acos(cos_angle);
   }

   /// Build a map from edge index to all adjacent boundary attributes.
   /// Interior edges have an empty list.  Boundary edges collect every
   /// distinct attribute of the boundary faces that contain them.
   void BuildEdgeBdrAttrMap()
   {
      edge_bdr_attrs_.resize(mesh_.GetNEdges());
      for (int be = 0; be < mesh_.GetNBE(); ++be)
      {
         const int bdr_attr = mesh_.GetBdrAttribute(be);
         mfem::Array<int> edges, ori;
         mesh_.GetBdrElementEdges(be, edges, ori);
         for (int j = 0; j < edges.Size(); ++j)
         {
            auto &attrs = edge_bdr_attrs_[edges[j]];
            if (std::find(attrs.begin(), attrs.end(), bdr_attr) == attrs.end())
            {
               attrs.push_back(bdr_attr);
            }
         }
      }
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

   /// Evaluate boundary velocity at a point, trying each boundary attribute
   /// until one gives a non-zero result.  This ensures that ridge edges
   /// (shared by e.g. a driven lid and a no-slip wall) use the driven
   /// velocity rather than zero.
   void EvaluateBoundaryVelocity(
       const BoundaryFunc &boundary,
       const mfem::Vector &x, double t,
       const std::vector<int> &attrs,
       mfem::Vector &vel) const
   {
      for (int attr : attrs)
      {
         boundary(x, t, attr, vel);
         if (vel.Norml2() > 1e-15) { return; }
      }
      // All attributes gave zero — vel already holds the last (zero) result.
   }

   /// Evaluate velocity at a vertex as a dihedral-angle-weighted average
   /// over all elements sharing a given edge.
   /// @a edge_idx    mesh edge index.
   /// @a vert_global global vertex index (must be one of the edge's vertices).
   /// @a vel         [out] weighted-average velocity (sized to dim_).
   void EvaluateVertexVelocityWeighted(
       mfem::GridFunction &velocity_gf,
       int edge_idx,
       int vert_global,
       mfem::Vector &vel) const
   {
      vel.SetSize(dim_);
      vel = 0.0;
      double total_weight = 0.0;

      const int row_off = edge_to_elems_.GetI()[edge_idx];
      const int n_adj = edge_to_elems_.RowSize(edge_idx);
      const int *elems = edge_to_elems_.GetRow(edge_idx);

      mfem::Vector v_elem(dim_);

      for (int k = 0; k < n_adj; ++k)
      {
         const int elem = elems[k];
         const double w = edge_elem_dihedral_angles_[row_off + k];

         // Find local vertex index within the element
         mfem::Array<int> elem_verts;
         mesh_.GetElementVertices(elem, elem_verts);
         int local_idx = -1;
         for (int j = 0; j < elem_verts.Size(); ++j)
         {
            if (elem_verts[j] == vert_global) { local_idx = j; break; }
         }

         // Get reference coordinates for this local vertex
         const int geom = mesh_.GetElementGeometry(elem);
         const mfem::IntegrationRule *ir = mfem::Geometries.GetVertices(geom);
         const mfem::IntegrationPoint &ip = ir->IntPoint(local_idx);

         velocity_gf.GetVectorValue(elem, ip, v_elem);

         for (int d = 0; d < dim_; ++d) { vel[d] += w * v_elem[d]; }
         total_weight += w;
      }

      if (total_weight > 1e-15)
      {
         vel /= total_weight;
      }
   }

   /// Trace a point backward using a precomputed vertex velocity for the
   /// predictor step, falling back to @a velocity for the Heun corrector.
   static void TraceDepartureWeighted(
       const mfem::Vector &vertex_vel,
       const VelocityFunc &velocity,
       double t, double dt,
       const mfem::Vector &x, mfem::Vector &d,
       int order)
   {
      const int dim = x.Size();
      d.SetSize(dim);

      if (order == 1)
      {
         for (int i = 0; i < dim; ++i)
         {
            d[i] = x[i] - dt * vertex_vel[i];
         }
      }
      else
      {
         // Predictor with precomputed vertex velocity
         for (int i = 0; i < dim; ++i)
         {
            d[i] = x[i] - dt * vertex_vel[i];
         }
         // Corrector: evaluate velocity at departure point via callback
         mfem::Vector vel2(dim);
         velocity(d, t - dt, vel2);
         for (int i = 0; i < dim; ++i)
         {
            d[i] = x[i] - 0.5 * dt * (vertex_vel[i] + vel2[i]);
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
                              int order)
   {
      const int dim = x.Size();
      d.SetSize(dim);

      mfem::Vector vel(dim);
      velocity(x, t, vel);

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
         velocity(d, t - dt, vel2);
         for (int i = 0; i < dim; ++i)
         {
            d[i] = x[i] - 0.5 * dt * (vel[i] + vel2[i]);
         }
      }
   }

   /// Compute the pullback DOF values for all edges.
   /// When @a velocity_gf is non-null, vertex velocities are computed as
   /// dihedral-angle-weighted averages over adjacent elements.
   void ComputePullbackDOFs(const VelocityFunc &velocity,
                            const BoundaryFunc &boundary,
                            double t, double dt,
                            mfem::GridFunction &gf_old,
                            mfem::Vector &dof_values,
                            int trace_order,
                            mfem::GridFunction *velocity_gf = nullptr)
   {
      const int n_edges = mesh_.GetNEdges();
      CellSplitWorkspace ws;
      double t_departure = t - dt;

      for (int e = 0; e < n_edges; ++e)
      {
         // Get edge vertices (v0 < v1 in MFEM convention)
         mfem::Array<int> verts;
         mesh_.GetEdgeVertices(e, verts);

         const double *v0_ptr = mesh_.GetVertex(verts[0]);
         const double *v1_ptr = mesh_.GetVertex(verts[1]);

         mfem::Vector v0(dim_), v1(dim_);
         for (int d = 0; d < dim_; ++d)
         {
            v0[d] = v0_ptr[d];
            v1[d] = v1_ptr[d];
         }

         // Trace vertices backward along characteristics
         mfem::Vector dep0(dim_), dep1(dim_);
         if (velocity_gf && !edge_bdr_attrs_[e].empty())
         {
            // Boundary edge: use exact BC velocity for tracing
            mfem::Vector vel0(dim_), vel1(dim_);
            EvaluateBoundaryVelocity(boundary, v0, t, edge_bdr_attrs_[e], vel0);
            EvaluateBoundaryVelocity(boundary, v1, t, edge_bdr_attrs_[e], vel1);
            TraceDepartureWeighted(vel0, velocity, t, dt, v0, dep0,
                                   trace_order);
            TraceDepartureWeighted(vel1, velocity, t, dt, v1, dep1,
                                   trace_order);
         }
         else if (velocity_gf)
         {
            // Interior edge: dihedral-angle-weighted GF average
            mfem::Vector vel0(dim_), vel1(dim_);
            EvaluateVertexVelocityWeighted(*velocity_gf, e, verts[0], vel0);
            EvaluateVertexVelocityWeighted(*velocity_gf, e, verts[1], vel1);
            TraceDepartureWeighted(vel0, velocity, t, dt, v0, dep0,
                                   trace_order);
            TraceDepartureWeighted(vel1, velocity, t, dt, v1, dep1,
                                   trace_order);
         }
         else
         {
            TraceDeparture(velocity, t, dt, v0, dep0, trace_order);
            TraceDeparture(velocity, t, dt, v1, dep1, trace_order);
         }

          // Get DOF index for this edge
          mfem::Array<int> dofs;
          fes_.GetEdgeDofs(e, dofs);

          // Compute transported DOF
          double new_dof = ComputeTransportedDOF(
              mesh_, gf_old, dep0, dep1, EdgeStartElement(e),
              boundary, t_departure, ws);

         dof_values(dofs[0]) = new_dof;
      }
   }

   mfem::Mesh &mesh_;
   mfem::FiniteElementSpace &fes_;
   int dim_;
   GaussLegendreRule<N_gauss> gl_rule_;
   mfem::Table edge_to_elems_;
   std::vector<double> edge_elem_dihedral_angles_;
   std::vector<std::vector<int>> edge_bdr_attrs_;  ///< edge → boundary attributes (empty=interior)
   std::vector<int> face_bdr_attr_;  ///< face index → boundary attribute (0=interior)
};

#endif // SEMI_LAGRANGIAN_ADVECTION_H
