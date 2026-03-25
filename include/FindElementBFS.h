#ifndef FIND_ELEMENT_BFS_H
#define FIND_ELEMENT_BFS_H

#include "mfem.hpp"
#include <queue>
#include <vector>

/// Search for the element containing @a point using a BFS starting from
/// @a start_elem_id.  Expands outward through face-neighbours until the
/// containing element is found or the whole mesh has been visited.
///
/// @return element id containing the point, or -1 if not found.
///         On success, @a ip is set to the reference-space coordinates.
inline int FindElementBFS(mfem::Mesh &mesh, int start_elem_id,
                          const mfem::Vector &point,
                          mfem::IntegrationPoint &ip,
                          int *visited_count = nullptr)
{
   const int ne = mesh.GetNE();
   if (start_elem_id < 0 || start_elem_id >= ne)
   {
      if (visited_count) { *visited_count = 0; }
      return -1;
   }

   const mfem::Table &el2el = mesh.ElementToElementTable();

   std::vector<bool> visited(ne, false);
   std::queue<int> queue;

   queue.push(start_elem_id);
   visited[start_elem_id] = true;

   int count = 0;

   mfem::IsoparametricTransformation eltrans;
   mfem::InverseElementTransformation inv_tr;
   inv_tr.SetInitialGuessType(mfem::InverseElementTransformation::Center);
   inv_tr.SetSolverType(
       mfem::InverseElementTransformation::NewtonElementProject);

   while (!queue.empty())
   {
      int elem_id = queue.front();
      queue.pop();
      ++count;

      mesh.GetElementTransformation(elem_id, &eltrans);
      inv_tr.SetTransformation(eltrans);

      int result = inv_tr.Transform(point, ip);
      if (result == mfem::InverseElementTransformation::Inside)
      {
         if (visited_count) { *visited_count = count; }
         return elem_id;
      }

      const int *nbrs = el2el.GetRow(elem_id);
      const int n_nbrs = el2el.RowSize(elem_id);
      for (int i = 0; i < n_nbrs; ++i)
      {
         if (nbrs[i] >= 0 && !visited[nbrs[i]])
         {
            visited[nbrs[i]] = true;
            queue.push(nbrs[i]);
         }
      }
   }

   if (visited_count) { *visited_count = count; }
   return -1;
}

#endif // FIND_ELEMENT_BFS_H
