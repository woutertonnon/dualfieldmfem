#ifndef RAPETTI_PROJECTION_H
#define RAPETTI_PROJECTION_H

#include "SmallEdgeEnumeration.h"
#include "mfem.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <vector>

/// Evaluates the k-th Rapetti shape function w^{v_i, e_j} = λ_i · w^{e_j}
/// at a point in the reference simplex.
///
/// The shape function is a vector-valued polynomial 1-form proxy:
///   w^{v_i, e_j}(x) = λ_i(x) · w^{e_j}(x)
/// where λ_i is the barycentric coordinate for vertex i and w^{e_j} is the
/// Whitney 1-form for edge e_j.
///
/// Whitney 1-form for edge e = (v_a, v_b):
///   w^e = λ_a ∇λ_b - λ_b ∇λ_a
class RapettiShapeFunctions
{
public:
   explicit RapettiShapeFunctions(int dim) : dim_(dim)
   {
      MFEM_VERIFY(dim == 2 || dim == 3,
                  "RapettiShapeFunctions: only 2D and 3D.");
      if (dim == 2)
      {
         // Triangle reference: v0=(0,0), v1=(1,0), v2=(0,1)
         // λ₀ = 1-x-y, λ₁ = x, λ₂ = y
         // ∇λ₀ = (-1,-1), ∇λ₁ = (1,0), ∇λ₂ = (0,1)
         n_verts_ = 3;
         n_edges_ = 3;
         // MFEM edges: e0=(0,1), e1=(1,2), e2=(2,0)
      }
      else
      {
         // Tet reference: v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), v3=(0,0,1)
         // λ₀ = 1-x-y-z, λ₁ = x, λ₂ = y, λ₃ = z
         // ∇λ₀ = (-1,-1,-1), ∇λ₁ = (1,0,0), ∇λ₂ = (0,1,0), ∇λ₃ = (0,0,1)
         n_verts_ = 4;
         n_edges_ = 6;
      }
      n_small_ = n_verts_ * n_edges_;
   }

   int NumSmall() const { return n_small_; }

   /// Evaluate Rapetti shape function k at reference point x.
   /// Returns a dim-dimensional vector.
   void Eval(int k, const mfem::Vector &x, mfem::Vector &val) const
   {
      val.SetSize(dim_);
      val = 0.0;

      int vi = k / n_edges_;
      int ej = k % n_edges_;

      // Barycentric coordinates
      double lam[4];
      ComputeBarycentric(x, lam);

      // Edge vertices
      int ea, eb;
      GetEdgeVerts(ej, ea, eb);

      // ∇λ for each vertex
      double grad_a[3], grad_b[3];
      GradLambda(ea, grad_a);
      GradLambda(eb, grad_b);

      // Whitney: w^e = λ_a ∇λ_b - λ_b ∇λ_a
      // Rapetti: λ_i * w^e
      double coef = lam[vi];
      for (int d = 0; d < dim_; ++d)
      {
         val[d] = coef * (lam[ea] * grad_b[d] - lam[eb] * grad_a[d]);
      }
   }

   /// Integrate Rapetti shape function k along a line segment from a to b
   /// in the reference simplex, using N-point Gauss-Legendre quadrature.
   double IntegrateAlongSegment(int k, const double *a, const double *b,
                                int n_quad = 3) const
   {
      // Direction vector
      double dir[3] = {0, 0, 0};
      for (int d = 0; d < dim_; ++d)
      {
         dir[d] = b[d] - a[d];
      }

      // Gauss-Legendre on [0,1]
      static const double gl2_x[] = {0.21132486540518712, 0.78867513459481288};
      static const double gl2_w[] = {0.5, 0.5};
      static const double gl3_x[] = {0.11270166537925831, 0.5,
                                     0.88729833462074169};
      static const double gl3_w[] = {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0};

      const double *qx, *qw;
      int nq;
      if (n_quad <= 2)
      {
         qx = gl2_x;
         qw = gl2_w;
         nq = 2;
      }
      else
      {
         qx = gl3_x;
         qw = gl3_w;
         nq = 3;
      }

      double result = 0.0;
      mfem::Vector pt(dim_), val(dim_);
      for (int q = 0; q < nq; ++q)
      {
         double t = qx[q];
         for (int d = 0; d < dim_; ++d)
         {
            pt[d] = a[d] + t * dir[d];
         }
         Eval(k, pt, val);
         double dot = 0.0;
         for (int d = 0; d < dim_; ++d)
         {
            dot += val[d] * dir[d];
         }
         result += qw[q] * dot;
      }
      return result;
   }

private:
   void ComputeBarycentric(const mfem::Vector &x, double *lam) const
   {
      if (dim_ == 2)
      {
         lam[1] = x[0];
         lam[2] = x[1];
         lam[0] = 1.0 - x[0] - x[1];
      }
      else
      {
         lam[1] = x[0];
         lam[2] = x[1];
         lam[3] = x[2];
         lam[0] = 1.0 - x[0] - x[1] - x[2];
      }
   }

   void GradLambda(int v, double *grad) const
   {
      // ∇λ_v on the reference simplex
      grad[0] = grad[1] = grad[2] = 0.0;
      if (v == 0)
      {
         for (int d = 0; d < dim_; ++d) { grad[d] = -1.0; }
      }
      else
      {
         grad[v - 1] = 1.0;
      }
   }

   void GetEdgeVerts(int ej, int &va, int &vb) const
   {
      if (dim_ == 2)
      {
         // MFEM triangle: e0=(0,1), e1=(1,2), e2=(2,0)
         static const int ev[3][2] = {{0, 1}, {1, 2}, {2, 0}};
         va = ev[ej][0];
         vb = ev[ej][1];
      }
      else
      {
         // MFEM tet: e0=(0,1), e1=(0,2), e2=(0,3), e3=(1,2), e4=(1,3), e5=(2,3)
         static const int ev[6][2] = {
            {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
         };
         va = ev[ej][0];
         vb = ev[ej][1];
      }
   }

   int dim_;
   int n_verts_;
   int n_edges_;
   int n_small_;
};


/// Computes the I_{h,2} projection operator on the reference simplex.
///
/// Given raw line integrals over all small edges of one element, produces
/// the Rapetti DOF coefficients via the two-phase algorithm from
/// Tonnon & Hiptmair (2023), Section 3.1.3:
///
///   Phase 1: Solve 2×2 invertible systems for edge-local small edges
///   Phase 2: Solve face-interior systems in least-squares sense
class RapettiProjector
{
public:
   explicit RapettiProjector(int dim)
       : dim_(dim), topo_helper_(dim), shapes_(dim)
   {
      n_small_ = shapes_.NumSmall();
      n_verts_ = (dim == 2) ? 3 : 4;
      n_edges_ = (dim == 2) ? 3 : 6;

      ComputeMMatrix();
      PrecomputeEdgeInverses();
      PrecomputeFaceInteriorPseudoinverses();
   }

   int NumSmall() const { return n_small_; }

   /// Project raw small-edge integrals to Rapetti DOF coefficients.
   ///
   /// @a raw_integrals: length n_small_, the integral of a 1-form over each
   ///   transported small edge.
   /// @a rapetti_dofs: length n_small_, output Rapetti coefficients.
   void ProjectElement(const double *raw_integrals,
                       double *rapetti_dofs) const
   {
      // Phase 1: edge-local solves
      for (int e = 0; e < n_edges_; ++e)
      {
         int k0 = edge_local_idx_[e][0];
         int k1 = edge_local_idx_[e][1];
         double r0 = raw_integrals[k0];
         double r1 = raw_integrals[k1];
         rapetti_dofs[k0] = edge_inv_[e][0][0] * r0 + edge_inv_[e][0][1] * r1;
         rapetti_dofs[k1] = edge_inv_[e][1][0] * r0 + edge_inv_[e][1][1] * r1;
      }

      // Phase 2: face-interior least-squares solves
      for (int f = 0; f < static_cast<int>(face_groups_.size()); ++f)
      {
         const auto &fg = face_groups_[f];
         int n_int = static_cast<int>(fg.interior_idx.size());
         if (n_int == 0) { continue; }

         // Build modified RHS: r_interior - coupling * eta_edge
         std::vector<double> rhs(n_int);
         for (int i = 0; i < n_int; ++i)
         {
            int ki = fg.interior_idx[i];
            rhs[i] = raw_integrals[ki];

            // Subtract contributions from already-determined edge-local DOFs
            for (int j = 0; j < static_cast<int>(fg.edge_idx.size()); ++j)
            {
               int kj = fg.edge_idx[j];
               rhs[i] -= fg.coupling(i, j) * rapetti_dofs[kj];
            }
         }

         // Apply pseudoinverse
         for (int i = 0; i < n_int; ++i)
         {
            int ki = fg.interior_idx[i];
            rapetti_dofs[ki] = 0.0;
            for (int j = 0; j < n_int; ++j)
            {
               rapetti_dofs[ki] += fg.pseudoinverse(i, j) * rhs[j];
            }
         }
      }
   }

   /// Access the full M matrix (for testing).
   const mfem::DenseMatrix &GetMMatrix() const { return M_; }

private:
   /// Helper that provides the same topology info as SmallEdgeTopology but
   /// without requiring a mesh (works on the reference element only).
   struct ReferenceTopology
   {
      int dim, n_verts, n_edges, n_small;
      std::vector<std::array<int, 2>> edge_verts;
      std::vector<std::array<int, 2>> edge_local; // per big edge: 2 small edges
      std::vector<std::vector<int>> face_interior; // per face: interior small edges

      explicit ReferenceTopology(int d) : dim(d)
      {
         n_verts = d + 1;
         if (d == 2)
         {
            n_edges = 3;
            edge_verts = {{{0, 1}}, {{1, 2}}, {{2, 0}}};
         }
         else
         {
            n_edges = 6;
            edge_verts = {{{0, 1}}, {{0, 2}}, {{0, 3}},
                          {{1, 2}}, {{1, 3}}, {{2, 3}}};
         }
         n_small = n_verts * n_edges;

         // Edge-local
         edge_local.resize(n_edges);
         for (int e = 0; e < n_edges; ++e)
         {
            int v0 = edge_verts[e][0];
            int v1 = edge_verts[e][1];
            edge_local[e] = {v0 * n_edges + e, v1 * n_edges + e};
         }

         // Face-interior
         if (d == 2)
         {
            face_interior.resize(1);
            for (int k = 0; k < n_small; ++k)
            {
               int v = k / n_edges;
               int e = k % n_edges;
               int ev0 = edge_verts[e][0], ev1 = edge_verts[e][1];
               if (v != ev0 && v != ev1)
               {
                  face_interior[0].push_back(k);
               }
            }
         }
         else
         {
            static const int fv[4][3] = {
               {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}
            };
            face_interior.resize(4);
            for (int f = 0; f < 4; ++f)
            {
               for (int k = 0; k < n_small; ++k)
               {
                  int v = k / n_edges;
                  int e = k % n_edges;
                  int ev0 = edge_verts[e][0], ev1 = edge_verts[e][1];
                  bool v_on = (v == fv[f][0] || v == fv[f][1] || v == fv[f][2]);
                  bool e_on = false;
                  {
                     bool a = (ev0 == fv[f][0] || ev0 == fv[f][1] || ev0 == fv[f][2]);
                     bool b = (ev1 == fv[f][0] || ev1 == fv[f][1] || ev1 == fv[f][2]);
                     e_on = a && b;
                  }
                  bool is_edge_local = (v == ev0 || v == ev1);
                  if (v_on && e_on && !is_edge_local)
                  {
                     face_interior[f].push_back(k);
                  }
               }
            }
         }
      }
   };

   /// Compute the M matrix on the reference element.
   /// M_{ij} = ∫_{s_i} w^{s_j} where s_i is the i-th small edge and
   /// w^{s_j} is the j-th Rapetti shape function.
   void ComputeMMatrix()
   {
      M_.SetSize(n_small_, n_small_);
      M_ = 0.0;

      // Reference vertices
      double rv[4][3] = {};
      if (dim_ == 2)
      {
         rv[0][0] = 0; rv[0][1] = 0;
         rv[1][0] = 1; rv[1][1] = 0;
         rv[2][0] = 0; rv[2][1] = 1;
      }
      else
      {
         rv[0][0] = 0; rv[0][1] = 0; rv[0][2] = 0;
         rv[1][0] = 1; rv[1][1] = 0; rv[1][2] = 0;
         rv[2][0] = 0; rv[2][1] = 1; rv[2][2] = 0;
         rv[3][0] = 0; rv[3][1] = 0; rv[3][2] = 1;
      }

      // For each small edge s_i, compute its reference endpoints.
      // Then integrate each shape function w^{s_j} along s_i.
      for (int i = 0; i < n_small_; ++i)
      {
         int vi = i / (dim_ == 2 ? 3 : 6);
         int ei = i % (dim_ == 2 ? 3 : 6);
         int ea_i, eb_i;
         GetEdgeVerts(ei, ea_i, eb_i);

         // Endpoints of small edge i on reference element
         double a[3] = {}, b[3] = {};
         for (int d = 0; d < dim_; ++d)
         {
            a[d] = 0.5 * (rv[vi][d] + rv[ea_i][d]);
            b[d] = 0.5 * (rv[vi][d] + rv[eb_i][d]);
         }

         for (int j = 0; j < n_small_; ++j)
         {
            M_(i, j) = shapes_.IntegrateAlongSegment(j, a, b, 3);
         }
      }
   }

   void GetEdgeVerts(int ej, int &va, int &vb) const
   {
      if (dim_ == 2)
      {
         static const int ev[3][2] = {{0, 1}, {1, 2}, {2, 0}};
         va = ev[ej][0]; vb = ev[ej][1];
      }
      else
      {
         static const int ev[6][2] = {
            {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
         };
         va = ev[ej][0]; vb = ev[ej][1];
      }
   }

   void PrecomputeEdgeInverses()
   {
      edge_local_idx_.resize(n_edges_);
      edge_inv_.resize(n_edges_);

      for (int e = 0; e < n_edges_; ++e)
      {
         int ea, eb;
         GetEdgeVerts(e, ea, eb);
         int k0 = ea * n_edges_ + e;
         int k1 = eb * n_edges_ + e;
         edge_local_idx_[e] = {k0, k1};

         // Extract 2×2 block M[{k0,k1}, {k0,k1}]
         double a00 = M_(k0, k0), a01 = M_(k0, k1);
         double a10 = M_(k1, k0), a11 = M_(k1, k1);
         double det = a00 * a11 - a01 * a10;
         MFEM_VERIFY(std::abs(det) > 1e-14,
                     "RapettiProjector: edge block is singular.");
         double inv_det = 1.0 / det;
         edge_inv_[e][0][0] = a11 * inv_det;
         edge_inv_[e][0][1] = -a01 * inv_det;
         edge_inv_[e][1][0] = -a10 * inv_det;
         edge_inv_[e][1][1] = a00 * inv_det;
      }
   }

   void PrecomputeFaceInteriorPseudoinverses()
   {
      ReferenceTopology ref(dim_);
      int n_faces = static_cast<int>(ref.face_interior.size());
      face_groups_.resize(n_faces);

      for (int f = 0; f < n_faces; ++f)
      {
         auto &fg = face_groups_[f];
         fg.interior_idx = ref.face_interior[f];
         int n_int = static_cast<int>(fg.interior_idx.size());
         if (n_int == 0) { continue; }

         // Collect all edge-local small edges that couple to these interior ones.
         // These are the edge-local small edges on edges of this face.
         std::vector<int> edge_idx;
         if (dim_ == 2)
         {
            // All 6 edge-local small edges
            for (int e = 0; e < n_edges_; ++e)
            {
               edge_idx.push_back(edge_local_idx_[e][0]);
               edge_idx.push_back(edge_local_idx_[e][1]);
            }
         }
         else
         {
            // Edges of this face
            static const int fv[4][3] = {
               {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}
            };
            for (int e = 0; e < n_edges_; ++e)
            {
               int ea, eb;
               GetEdgeVerts(e, ea, eb);
               bool a_on = (ea == fv[f][0] || ea == fv[f][1] || ea == fv[f][2]);
               bool b_on = (eb == fv[f][0] || eb == fv[f][1] || eb == fv[f][2]);
               if (a_on && b_on)
               {
                  edge_idx.push_back(edge_local_idx_[e][0]);
                  edge_idx.push_back(edge_local_idx_[e][1]);
               }
            }
         }
         fg.edge_idx = edge_idx;
         int n_edge = static_cast<int>(edge_idx.size());

         // Coupling matrix: M[interior_idx, edge_idx]
         fg.coupling.SetSize(n_int, n_edge);
         for (int i = 0; i < n_int; ++i)
         {
            for (int j = 0; j < n_edge; ++j)
            {
               fg.coupling(i, j) = M_(fg.interior_idx[i], edge_idx[j]);
            }
         }

         // Interior-interior block: M[interior_idx, interior_idx]
         mfem::DenseMatrix M_int(n_int, n_int);
         for (int i = 0; i < n_int; ++i)
         {
            for (int j = 0; j < n_int; ++j)
            {
               M_int(i, j) = M_(fg.interior_idx[i], fg.interior_idx[j]);
            }
         }

         // Compute pseudoinverse of M_int via SVD
         fg.pseudoinverse.SetSize(n_int, n_int);
         ComputePseudoinverse(M_int, fg.pseudoinverse);
      }
   }

   /// Jacobi eigendecomposition of a symmetric matrix (LAPACK-free).
   /// On output, evals contains eigenvalues and evecs the corresponding
   /// orthonormal eigenvectors as columns.  Works for small matrices
   /// (n ≤ ~30) which is all we need here.
   static void JacobiEigen(const mfem::DenseMatrix &S,
                            mfem::Vector &evals,
                            mfem::DenseMatrix &evecs)
   {
      const int n = S.Height();
      assert(n == S.Width());

      // Work on a copy
      mfem::DenseMatrix A(S);
      evecs.SetSize(n, n);
      evecs = 0.0;
      for (int i = 0; i < n; ++i) { evecs(i, i) = 1.0; }

      const int max_iter = 100 * n * n;
      const double tol = 1e-15;

      for (int iter = 0; iter < max_iter; ++iter)
      {
         // Find largest off-diagonal element
         double max_off = 0.0;
         int p = 0, q = 1;
         for (int i = 0; i < n; ++i)
         {
            for (int j = i + 1; j < n; ++j)
            {
               double aij = std::abs(A(i, j));
               if (aij > max_off) { max_off = aij; p = i; q = j; }
            }
         }
         if (max_off < tol) { break; }

         // Compute rotation
         double app = A(p, p), aqq = A(q, q), apq = A(p, q);
         double theta_val;
         if (std::abs(app - aqq) < tol * (std::abs(app) + std::abs(aqq) + tol))
         {
            theta_val = M_PI / 4.0;
         }
         else
         {
            theta_val = 0.5 * std::atan2(2.0 * apq, app - aqq);
         }
         double c = std::cos(theta_val), s = std::sin(theta_val);

         // Apply Givens rotation: A <- G^T A G
         for (int i = 0; i < n; ++i)
         {
            double aip = A(i, p), aiq = A(i, q);
            A(i, p) =  c * aip + s * aiq;
            A(i, q) = -s * aip + c * aiq;
         }
         for (int j = 0; j < n; ++j)
         {
            double apj = A(p, j), aqj = A(q, j);
            A(p, j) =  c * apj + s * aqj;
            A(q, j) = -s * apj + c * aqj;
         }

         // Update eigenvectors
         for (int i = 0; i < n; ++i)
         {
            double vip = evecs(i, p), viq = evecs(i, q);
            evecs(i, p) =  c * vip + s * viq;
            evecs(i, q) = -s * vip + c * viq;
         }
      }

      evals.SetSize(n);
      for (int i = 0; i < n; ++i) { evals(i) = A(i, i); }
   }

   /// Compute the Moore-Penrose pseudoinverse of A (LAPACK-free).
   /// Uses Jacobi eigendecomposition of A^T A.
   static void ComputePseudoinverse(const mfem::DenseMatrix &A,
                                     mfem::DenseMatrix &A_pinv)
   {
      const int m = A.Height();
      const int n = A.Width();

      // A_pinv = (A^T A)^+ A^T
      mfem::DenseMatrix AtA(n, n);
      MultAtB(A, A, AtA);

      // Eigendecomposition of symmetric AtA
      mfem::DenseMatrix evecs(n, n);
      mfem::Vector evals(n);
      JacobiEigen(AtA, evals, evecs);

      // Find threshold from max eigenvalue
      double max_eval = 0.0;
      for (int i = 0; i < n; ++i)
      {
         max_eval = std::max(max_eval, std::abs(evals(i)));
      }
      double tol = 1e-12 * max_eval;

      // Build (A^T A)^+ = V diag(1/lambda_i for lambda_i > tol) V^T
      mfem::DenseMatrix D_inv(n, n);
      D_inv = 0.0;
      for (int i = 0; i < n; ++i)
      {
         D_inv(i, i) = (evals(i) > tol) ? (1.0 / evals(i)) : 0.0;
      }

      mfem::DenseMatrix temp(n, n), AtA_pinv(n, n);
      Mult(evecs, D_inv, temp);
      MultABt(temp, evecs, AtA_pinv);

      // A^+ = (A^T A)^+ A^T
      A_pinv.SetSize(n, m);
      MultABt(AtA_pinv, A, A_pinv);
   }

   struct FaceGroup
   {
      std::vector<int> interior_idx;
      std::vector<int> edge_idx;
      mfem::DenseMatrix coupling;      ///< M[interior, edge]
      mfem::DenseMatrix pseudoinverse;  ///< (M[interior, interior])^+
   };

   int dim_;
   int n_small_;
   int n_verts_;
   int n_edges_;
   ReferenceTopology topo_helper_;
   RapettiShapeFunctions shapes_;
   mfem::DenseMatrix M_;

   std::vector<std::array<int, 2>> edge_local_idx_;
   std::vector<std::array<std::array<double, 2>, 2>> edge_inv_;
   std::vector<FaceGroup> face_groups_;
};


/// Converts between Rapetti small-edge DOFs and MFEM ND₂ DOFs.
///
/// The transformation matrix T has entries T_{ij} = DOF^{ND₂}_i(w^{s_j})
/// where DOF^{ND₂}_i is the i-th MFEM ND₂ DOF functional and w^{s_j} is
/// the j-th Rapetti shape function.
///
///   - 2D: T is 8×9 (8 ND₂ DOFs, 9 Rapetti shape functions), rank 8
///   - 3D: T is 20×24, rank 20
class RapettiToND2Converter
{
public:
   explicit RapettiToND2Converter(int dim)
       : dim_(dim), shapes_(dim)
   {
      n_small_ = shapes_.NumSmall();
      ComputeTransformationMatrix();
   }

   int NumND2() const { return n_nd2_; }
   int NumSmall() const { return n_small_; }

   /// Convert Rapetti DOFs to MFEM ND₂ DOFs.
   void Convert(const double *rapetti_dofs, double *nd2_dofs) const
   {
      for (int i = 0; i < n_nd2_; ++i)
      {
         double sum = 0.0;
         for (int j = 0; j < n_small_; ++j)
         {
            sum += T_(i, j) * rapetti_dofs[j];
         }
         nd2_dofs[i] = sum;
      }
   }

   /// Convert MFEM ND₂ DOFs to Rapetti DOFs (pseudoinverse, for testing).
   void ConvertInverse(const double *nd2_dofs, double *rapetti_dofs) const
   {
      for (int i = 0; i < n_small_; ++i)
      {
         double sum = 0.0;
         for (int j = 0; j < n_nd2_; ++j)
         {
            sum += T_pinv_(i, j) * nd2_dofs[j];
         }
         rapetti_dofs[i] = sum;
      }
   }

   /// Access T matrix (for testing).
   const mfem::DenseMatrix &GetTMatrix() const { return T_; }

private:
   void ComputeTransformationMatrix()
   {
      // Create a reference ND₂ finite element.
      mfem::ND_FECollection fec(2, dim_);
      mfem::Geometry::Type geom = (dim_ == 2) ? mfem::Geometry::TRIANGLE
                                               : mfem::Geometry::TETRAHEDRON;
      const mfem::FiniteElement *fe = fec.FiniteElementForGeometry(geom);
      n_nd2_ = fe->GetDof();

      T_.SetSize(n_nd2_, n_small_);
      T_ = 0.0;

      // For each ND₂ DOF, evaluate each Rapetti shape function.
      // ND₂ DOFs are defined by a set of integration points and
      // directions (weights). We use MFEM's ProjectCurl/CalcVShape
      // infrastructure indirectly.
      //
      // The DOF functionals for Nédélec elements are:
      //   DOF_i(ω) = Σ_q w_{i,q} · ω(x_{i,q}) · d_{i,q}
      // where x_{i,q} are integration points and d_{i,q} directions.
      //
      // We can evaluate T_{ij} by using the element's interpolation:
      // if ω = w^{s_j} (a known polynomial), then DOF_i(ω) can be
      // computed by evaluating w^{s_j} at the DOF points.

      // Use MFEM's Project mechanism: create a VectorCoefficient for each
      // Rapetti shape function and project it.
      mfem::IsoparametricTransformation ref_trans;
      if (dim_ == 2)
      {
         ref_trans.SetIdentityTransformation(mfem::Geometry::TRIANGLE);
      }
      else
      {
         ref_trans.SetIdentityTransformation(mfem::Geometry::TETRAHEDRON);
      }

      for (int j = 0; j < n_small_; ++j)
      {
         // Create a lambda-based coefficient for Rapetti shape function j
         auto eval_func = [this, j](const mfem::Vector &x,
                                     mfem::Vector &val) {
            shapes_.Eval(j, x, val);
         };
         mfem::VectorFunctionCoefficient coef(dim_, eval_func);

         mfem::Vector dofs(n_nd2_);
         fe->Project(coef, ref_trans, dofs);

         for (int i = 0; i < n_nd2_; ++i)
         {
            T_(i, j) = dofs(i);
         }
      }

      // Compute pseudoinverse T^+ = (T^T T)^{-1} T^T
      // T is n_nd2 × n_small with n_nd2 < n_small, full row rank.
      // T^+ = T^T (T T^T)^{-1}
      mfem::DenseMatrix TTt(n_nd2_, n_nd2_);
      MultABt(T_, T_, TTt);

      mfem::DenseMatrixInverse TTt_inv(TTt);
      mfem::DenseMatrix TTt_inv_mat(n_nd2_, n_nd2_);
      TTt_inv.GetInverseMatrix(TTt_inv_mat);

      T_pinv_.SetSize(n_small_, n_nd2_);
      MultAtB(T_, TTt_inv_mat, T_pinv_);
   }

   int dim_;
   int n_small_;
   int n_nd2_;
   RapettiShapeFunctions shapes_;
   mfem::DenseMatrix T_;       ///< n_nd2 × n_small
   mfem::DenseMatrix T_pinv_;  ///< n_small × n_nd2
};

#endif // RAPETTI_PROJECTION_H
