#ifndef INTEGRATE_LINE_CELL_SPLIT_H
#define INTEGRATE_LINE_CELL_SPLIT_H

#include "FindElementBFS.h"
#include <boost/math/quadrature/gauss.hpp>
#include <limits>

/// A sub-segment of a line that lies entirely within one mesh element.
struct LineSegment
{
   int elem_id;
   double s_start;
   double s_end;
};

/// Find where the line x(s) = pos1 + s*dir exits element @a elem_id.
/// Checks all faces of the element for intersection with the line.
/// Ignores @a entry_face to avoid re-entering through the same face.
/// Returns true if an exit was found, with @a s_exit and @a exit_face set.
inline bool FindExitFace(mfem::Mesh &mesh, int elem_id,
                         const mfem::Vector &pos1, const mfem::Vector &dir,
                         double s_current, int entry_face,
                         double &s_exit, int &exit_face)
{
   const int dim = mesh.SpaceDimension();
   mfem::Array<int> faces, ori;
   if (dim == 2)
   {
      mesh.GetElementEdges(elem_id, faces, ori);
   }
   else
   {
      mesh.GetElementFaces(elem_id, faces, ori);
   }

   s_exit = std::numeric_limits<double>::max();
   exit_face = -1;

   for (int f = 0; f < faces.Size(); ++f)
   {
      if (faces[f] == entry_face) { continue; }

      mfem::Array<int> verts;
      if (dim == 2)
      {
         mesh.GetEdgeVertices(faces[f], verts);
      }
      else
      {
         mesh.GetFaceVertices(faces[f], verts);
      }

      double s_hit = -1.0;
      bool hit = false;

      if (dim == 2)
      {
         // Line-edge intersection in 2D
         const double *v0 = mesh.GetVertex(verts[0]);
         const double *v1 = mesh.GetVertex(verts[1]);

         double ex = v1[0] - v0[0], ey = v1[1] - v0[1];
         double det = dir[0] * (-ey) - dir[1] * (-ex);
         if (std::abs(det) < 1e-14) { continue; }

         double rx = v0[0] - pos1[0], ry = v0[1] - pos1[1];
         s_hit = ((-ey) * rx + ex * ry) / det;
         double t = (-dir[1] * rx + dir[0] * ry) / det;

         hit = (t >= -1e-10 && t <= 1.0 + 1e-10);
         if (hit && s_hit > s_current + 1e-12 && s_hit < s_exit)
         {
            s_exit = s_hit;
            exit_face = faces[f];
         }
      }
      else // dim == 3
      {
         // Line-face intersection in 3D.
         // Split the face into triangles: for a triangle (3 verts) test once,
         // for a quad (4 verts) test two triangles (0-1-2) and (0-2-3).
         int n_tris = (verts.Size() == 4) ? 2 : 1;
         int tri_idx[2][3] = {{0, 1, 2}, {0, 2, 3}};

         for (int t = 0; t < n_tris; ++t)
         {
            const double *tv0 = mesh.GetVertex(verts[tri_idx[t][0]]);
            const double *tv1 = mesh.GetVertex(verts[tri_idx[t][1]]);
            const double *tv2 = mesh.GetVertex(verts[tri_idx[t][2]]);

            double e1[3] = {tv1[0]-tv0[0], tv1[1]-tv0[1], tv1[2]-tv0[2]};
            double e2[3] = {tv2[0]-tv0[0], tv2[1]-tv0[1], tv2[2]-tv0[2]};
            double n[3]  = {e1[1]*e2[2]-e1[2]*e2[1],
                            e1[2]*e2[0]-e1[0]*e2[2],
                            e1[0]*e2[1]-e1[1]*e2[0]};

            double denom = n[0]*dir[0] + n[1]*dir[1] + n[2]*dir[2];
            if (std::abs(denom) < 1e-14) { continue; }

            double dd[3] = {tv0[0]-pos1[0], tv0[1]-pos1[1], tv0[2]-pos1[2]};
            double s_tri = (n[0]*dd[0] + n[1]*dd[1] + n[2]*dd[2]) / denom;

            double p[3] = {pos1[0]+s_tri*dir[0]-tv0[0],
                           pos1[1]+s_tri*dir[1]-tv0[1],
                           pos1[2]+s_tri*dir[2]-tv0[2]};

            double d00 = e1[0]*e1[0]+e1[1]*e1[1]+e1[2]*e1[2];
            double d01 = e1[0]*e2[0]+e1[1]*e2[1]+e1[2]*e2[2];
            double d11 = e2[0]*e2[0]+e2[1]*e2[1]+e2[2]*e2[2];
            double d20 = p[0]*e1[0]+p[1]*e1[1]+p[2]*e1[2];
            double d21 = p[0]*e2[0]+p[1]*e2[1]+p[2]*e2[2];

            double inv = 1.0 / (d00*d11 - d01*d01);
            double bv = (d11*d20 - d01*d21) * inv;
            double bw = (d00*d21 - d01*d20) * inv;

            if (bv >= -1e-10 && bw >= -1e-10 && bv + bw <= 1.0 + 1e-10 &&
                s_tri > s_current + 1e-12 && s_tri < s_exit)
            {
               s_exit = s_tri;
               exit_face = faces[f];
               hit = true;
            }
         }
      }

   }

   return exit_face >= 0;
}

/// Walk the line from pos1 to pos2 through the mesh, splitting it at
/// element boundaries.  Returns a list of sub-segments, each lying
/// entirely within one element.
inline std::vector<LineSegment> SplitLineIntoSegments(
    mfem::Mesh &mesh, int start_elem_id,
    const mfem::Vector &pos1, const mfem::Vector &pos2,
    int *out_exit_face = nullptr)
{
   std::vector<LineSegment> segments;
   if (out_exit_face) { *out_exit_face = -1; }

   const int dim = mesh.SpaceDimension();
   mfem::Vector dir(dim);
   subtract(pos2, pos1, dir);

   // Zero or near-zero length line: nothing to integrate
   double dir_norm_sq = 0.0;
   for (int d = 0; d < dim; ++d) { dir_norm_sq += dir[d] * dir[d]; }
   if (dir_norm_sq < 1e-18) { return segments; }

   mfem::IntegrationPoint ip;
   int elem = FindElementBFS(mesh, start_elem_id, pos1, ip);
   if (elem < 0) { return segments; }

   double s = 0.0;
   int entry_face = -1;
   constexpr double nudge_step = 1e-10;

   while (s < 1.0 - 1e-12)
   {
      double s_exit;
      int exit_face;

      bool found = FindExitFace(mesh, elem, pos1, dir,
                                s, entry_face, s_exit, exit_face);

      if (!found)
      {
         // No exit face found.  This typically happens when the line
         // passes exactly through a mesh vertex or lies on a face:
         // all face intersections land at s_current and are rejected.
         // Nudge forward and use BFS to find the next element.
         double s_nudge = s + nudge_step;
         if (s_nudge >= 1.0 - 1e-12)
         {
            segments.push_back({elem, s, 1.0});
            break;
         }

         segments.push_back({elem, s, s_nudge});

         mfem::Vector nudged_pt(dim);
         add(pos1, s_nudge, dir, nudged_pt);

         mfem::IntegrationPoint nudged_ip;
         int next = FindElementBFS(mesh, elem, nudged_pt, nudged_ip);
         if (next < 0) { break; }

         s = s_nudge;
         elem = next;
         entry_face = -1;
         continue;
      }

      if (s_exit >= 1.0 - 1e-12)
      {
         segments.push_back({elem, s, 1.0});
         break;
      }

      segments.push_back({elem, s, s_exit});
      s = s_exit;

      // Cross to the neighbour through exit_face
      int elem1, elem2;
      mesh.GetFaceElements(exit_face, &elem1, &elem2);
      int next = (elem1 == elem) ? elem2 : elem1;

      if (next < 0) // hit the domain boundary
      {
         if (out_exit_face) { *out_exit_face = exit_face; }
         break;
      }

      elem = next;
      entry_face = exit_face;
   }

   return segments;
}

/// Scratch space for the cell-split integrator.
struct CellSplitWorkspace
{
   int dim = 0;
   mfem::Vector point;
   mfem::Vector val;
   mfem::Vector direction;
   mfem::IsoparametricTransformation eltrans;
   mfem::InverseElementTransformation inv_tr;

   CellSplitWorkspace()
   {
      inv_tr.SetInitialGuessType(mfem::InverseElementTransformation::Center);
      inv_tr.SetSolverType(
          mfem::InverseElementTransformation::NewtonElementProject);
   }

   void Init(int d)
   {
      dim = d;
      point.SetSize(d);
      val.SetSize(d);
      direction.SetSize(d);
   }
};

/// Pre-computed Gauss-Legendre rule on [0, 1] for integrating polynomial
/// sub-segments exactly.  For ND1 on affine simplices, N_gauss=1 (midpoint)
/// suffices.  For ND2, N_gauss=2 is exact.
template <unsigned N_gauss>
struct GaussLegendreRule
{
   int num_points;
   std::vector<double> nodes;
   std::vector<double> weights;

   GaussLegendreRule()
   {
      const auto &abs =
          boost::math::quadrature::gauss<double, N_gauss>::abscissa();
      const auto &wts =
          boost::math::quadrature::gauss<double, N_gauss>::weights();

      const int half = static_cast<int>(abs.size());
      nodes.reserve(N_gauss);
      weights.reserve(N_gauss);

      for (int i = half - 1; i >= 0; --i)
      {
         if (abs[i] == 0.0) { continue; }
         nodes.push_back((-abs[i] + 1.0) / 2.0);
         weights.push_back(wts[i] / 2.0);
      }
      for (int i = 0; i < half; ++i)
      {
         nodes.push_back((abs[i] + 1.0) / 2.0);
         weights.push_back(wts[i] / 2.0);
      }

      num_points = static_cast<int>(nodes.size());
   }
};

/// Integrate the tangential component of a vector GridFunction along the
/// line from @a pos1 to @a pos2 using the cell-splitting algorithm.
///
/// The line is split at element boundaries (Algorithm 1 from Tonnon &
/// Hiptmair, 2023).  Each sub-segment is integrated with a Gauss-Legendre
/// rule of order @a N_gauss, which is exact for polynomial integrands.
template <unsigned N_gauss>
inline double IntegrateLineTangentialCellSplit(
    mfem::Mesh &mesh,
    mfem::GridFunction &gf,
    int start_elem_id,
    const mfem::Vector &pos1,
    const mfem::Vector &pos2,
    const GaussLegendreRule<N_gauss> &rule,
    CellSplitWorkspace &ws)
{
   ws.Init(mesh.SpaceDimension());
   subtract(pos2, pos1, ws.direction);

   auto segments = SplitLineIntoSegments(mesh, start_elem_id, pos1, pos2);

   double result = 0.0;
   for (const auto &seg : segments)
   {
      double len = seg.s_end - seg.s_start;

      mesh.GetElementTransformation(seg.elem_id, &ws.eltrans);
      ws.inv_tr.SetTransformation(ws.eltrans);

      for (int q = 0; q < rule.num_points; ++q)
      {
         double s = seg.s_start + rule.nodes[q] * len;
         add(pos1, s, ws.direction, ws.point);

         mfem::IntegrationPoint ip;
         ws.inv_tr.Transform(ws.point, ip);
         ws.eltrans.SetIntPoint(&ip);

         gf.GetVectorValue(ws.eltrans, ip, ws.val);
         result += rule.weights[q] * len * (ws.val * ws.direction);
      }
   }
   return result;
}

/// Convenience overload without workspace.
template <unsigned N_gauss>
inline double IntegrateLineTangentialCellSplit(
    mfem::Mesh &mesh,
    mfem::GridFunction &gf,
    int start_elem_id,
    const mfem::Vector &pos1,
    const mfem::Vector &pos2,
    const GaussLegendreRule<N_gauss> &rule)
{
   CellSplitWorkspace ws;
   return IntegrateLineTangentialCellSplit(
       mesh, gf, start_elem_id, pos1, pos2, rule, ws);
}

#endif // INTEGRATE_LINE_CELL_SPLIT_H
