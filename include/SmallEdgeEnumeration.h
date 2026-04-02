#ifndef SMALL_EDGE_ENUMERATION_H
#define SMALL_EDGE_ENUMERATION_H

#include "mfem.hpp"
#include <array>
#include <cassert>
#include <vector>

/// Describes a small edge on the reference simplex.
/// A small edge {v_i, e_j} is defined by a vertex v_i and a big edge e_j.
/// Its physical endpoints are the midpoints of v_i with each endpoint of e_j:
///   a = 0.5*(v_i + v_{e_j,0}),  b = 0.5*(v_i + v_{e_j,1})
struct LocalSmallEdge
{
   int local_vertex;     ///< Local vertex index in element (0..d)
   int local_big_edge;   ///< Local edge index in element
   int edge_v0;          ///< Local index of first endpoint of the big edge
   int edge_v1;          ///< Local index of second endpoint of the big edge
};

/// Topology and coordinate computation for Rapetti "small edges" on
/// simplicial meshes.
///
/// For a d-simplex with (d+1) vertices and d(d+1)/2 edges, there are
/// (d+1) * d(d+1)/2 = 3(d+1)(d-1) small edges per element:
///   - 2D (triangle):    9 small edges
///   - 3D (tetrahedron): 24 small edges
///
/// Small edges are indexed as k = local_vertex * num_edges + local_edge,
/// i.e., vertex-major ordering.
///
/// Classification:
///   - Edge-local: vertex v_i is an endpoint of big edge e_j. These 2 small
///     edges per big edge have linearly independent shape functions.
///   - Face-interior (2D) / Face-local non-edge (3D): vertex v_i is NOT an
///     endpoint of big edge e_j, but both lie on the same face. Shape
///     functions are linearly dependent within each face group.
class SmallEdgeTopology
{
public:
   explicit SmallEdgeTopology(mfem::Mesh &mesh)
       : mesh_(mesh), dim_(mesh.SpaceDimension())
   {
      MFEM_VERIFY(dim_ == 2 || dim_ == 3,
                  "SmallEdgeTopology: only 2D and 3D supported.");
      InitReferenceData();
   }

   /// Number of small edges per element.
   int NumSmallEdgesPerElement() const { return n_small_; }

   /// Number of big edges per element.
   int NumBigEdgesPerElement() const { return n_big_edges_; }

   /// Number of vertices per element.
   int NumVerticesPerElement() const { return dim_ + 1; }

   /// Access local small edge data for small edge k (0-based).
   const LocalSmallEdge &GetLocalSmallEdge(int k) const
   {
      return local_small_edges_[k];
   }

   /// Compute physical endpoints of small edge k of element @a elem.
   /// Small edge {v_i, e_j} with e_j=(v_a, v_b):
   ///   a = 0.5*(v_i + v_a),  b = 0.5*(v_i + v_b)
   void GetSmallEdgeEndpoints(int elem, int k,
                              mfem::Vector &a, mfem::Vector &b) const
   {
      const LocalSmallEdge &se = local_small_edges_[k];
      mfem::Array<int> verts;
      mesh_.GetElementVertices(elem, verts);

      const double *vi = mesh_.GetVertex(verts[se.local_vertex]);
      const double *va = mesh_.GetVertex(verts[se.edge_v0]);
      const double *vb = mesh_.GetVertex(verts[se.edge_v1]);

      a.SetSize(dim_);
      b.SetSize(dim_);
      for (int d = 0; d < dim_; ++d)
      {
         a[d] = 0.5 * (vi[d] + va[d]);
         b[d] = 0.5 * (vi[d] + vb[d]);
      }
   }

   /// Get the 2 small edge indices (in [0..N_small)) that lie on big edge
   /// @a local_big_edge. These are the edge-local small edges formed by
   /// contracting the edge toward each of its own vertices.
   void GetEdgeLocalSmallEdges(int local_big_edge, int idx[2]) const
   {
      idx[0] = edge_local_[local_big_edge][0];
      idx[1] = edge_local_[local_big_edge][1];
   }

   /// Get the face-interior small edge indices for a given face.
   /// In 2D, there is one "face" (the element itself) with 3 interior
   /// small edges. In 3D, each triangular face has 3 face-interior
   /// small edges.
   int NumFaces() const { return n_faces_; }
   int NumFaceInteriorSmallEdges(int face) const
   {
      return static_cast<int>(face_interior_[face].size());
   }
   const std::vector<int> &GetFaceInteriorSmallEdges(int face) const
   {
      return face_interior_[face];
   }

   /// Dimension.
   int Dim() const { return dim_; }

   /// MFEM local edge vertex table: edge_verts_[e] = {v0, v1}.
   const std::array<int, 2> &EdgeVertices(int local_edge) const
   {
      return edge_verts_[local_edge];
   }

   /// Reference vertex coordinates (barycentric corners).
   /// 2D: (0,0), (1,0), (0,1)
   /// 3D: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
   const double *RefVertex(int v) const
   {
      return ref_verts_[v].data();
   }

private:
   void InitReferenceData()
   {
      const int nv = dim_ + 1;

      // Get edge-vertex connectivity from MFEM geometry constants.
      if (dim_ == 2)
      {
         // Triangle: 3 edges
         // MFEM: e0=(0,1), e1=(1,2), e2=(2,0)
         n_big_edges_ = 3;
         edge_verts_ = {{{{0, 1}}, {{1, 2}}, {{2, 0}}}};

         ref_verts_ = {{{0.0, 0.0}}, {{1.0, 0.0}}, {{0.0, 1.0}}};

         n_faces_ = 1; // the triangle itself
      }
      else
      {
         // Tetrahedron: 6 edges
         // MFEM: e0=(0,1), e1=(0,2), e2=(0,3), e3=(1,2), e4=(1,3), e5=(2,3)
         n_big_edges_ = 6;
         edge_verts_ = {{{{0, 1}}, {{0, 2}}, {{0, 3}},
                         {{1, 2}}, {{1, 3}}, {{2, 3}}}};

         ref_verts_ = {{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}},
                        {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}};

         // MFEM tet faces: f0=(1,2,3), f1=(0,3,2), f2=(0,1,3), f3=(0,2,1)
         n_faces_ = 4;
      }

      // Enumerate all small edges: vertex-major ordering.
      n_small_ = nv * n_big_edges_;
      local_small_edges_.resize(n_small_);
      for (int v = 0; v < nv; ++v)
      {
         for (int e = 0; e < n_big_edges_; ++e)
         {
            int k = v * n_big_edges_ + e;
            local_small_edges_[k].local_vertex = v;
            local_small_edges_[k].local_big_edge = e;
            local_small_edges_[k].edge_v0 = edge_verts_[e][0];
            local_small_edges_[k].edge_v1 = edge_verts_[e][1];
         }
      }

      // Classify: edge-local (vertex is an endpoint of the big edge).
      edge_local_.resize(n_big_edges_);
      for (int e = 0; e < n_big_edges_; ++e)
      {
         int v0 = edge_verts_[e][0];
         int v1 = edge_verts_[e][1];
         // Small edge from v0: k = v0 * n_big_edges_ + e
         // Small edge from v1: k = v1 * n_big_edges_ + e
         edge_local_[e][0] = v0 * n_big_edges_ + e;
         edge_local_[e][1] = v1 * n_big_edges_ + e;
      }

      // Classify: face-interior (vertex NOT on the big edge).
      if (dim_ == 2)
      {
         // One face (the triangle), 3 interior small edges.
         face_interior_.resize(1);
         for (int k = 0; k < n_small_; ++k)
         {
            int v = local_small_edges_[k].local_vertex;
            int e = local_small_edges_[k].local_big_edge;
            int ev0 = edge_verts_[e][0];
            int ev1 = edge_verts_[e][1];
            if (v != ev0 && v != ev1)
            {
               face_interior_[0].push_back(k);
            }
         }
      }
      else
      {
         // 4 faces. For each face, find small edges {v, e} where v is on the
         // face but NOT an endpoint of e, and e is an edge of the face.
         // MFEM tet face vertices:
         static const int face_v[4][3] = {
            {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}
         };

         // For each face, find which big edges lie on it.
         // A big edge lies on a face if both its vertices are face vertices.
         face_interior_.resize(4);
         for (int f = 0; f < 4; ++f)
         {
            std::array<int, 3> fv = {face_v[f][0], face_v[f][1], face_v[f][2]};

            for (int k = 0; k < n_small_; ++k)
            {
               int v = local_small_edges_[k].local_vertex;
               int e = local_small_edges_[k].local_big_edge;
               int ev0 = edge_verts_[e][0];
               int ev1 = edge_verts_[e][1];

               // Check: v is on this face, e is on this face, but v is NOT
               // an endpoint of e (which would make it edge-local).
               bool v_on_face = (v == fv[0] || v == fv[1] || v == fv[2]);
               bool ev0_on_face = (ev0 == fv[0] || ev0 == fv[1] || ev0 == fv[2]);
               bool ev1_on_face = (ev1 == fv[0] || ev1 == fv[1] || ev1 == fv[2]);
               bool e_on_face = ev0_on_face && ev1_on_face;
               bool is_edge_local = (v == ev0 || v == ev1);

               if (v_on_face && e_on_face && !is_edge_local)
               {
                  face_interior_[f].push_back(k);
               }
            }
         }
      }
   }

   mfem::Mesh &mesh_;
   int dim_;
   int n_small_;
   int n_big_edges_;
   int n_faces_;

   std::vector<LocalSmallEdge> local_small_edges_;

   // edge_local_[e] = {k0, k1}: the two small edge indices on big edge e
   std::vector<std::array<int, 2>> edge_local_;

   // face_interior_[f] = {k...}: small edge indices in face-interior group f
   std::vector<std::vector<int>> face_interior_;

   // Reference element data
   std::vector<std::array<int, 2>> edge_verts_;
   std::vector<std::array<double, 3>> ref_verts_;
};

#endif // SMALL_EDGE_ENUMERATION_H
