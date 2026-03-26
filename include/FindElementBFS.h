#ifndef FIND_ELEMENT_BFS_H
#define FIND_ELEMENT_BFS_H

#include "mfem.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>
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

   thread_local std::vector<int> visited_stamp;
   thread_local std::vector<int> bfs_queue;
   thread_local int stamp = 1;

   if ((int)visited_stamp.size() != ne)
   {
      visited_stamp.assign(ne, 0);
      stamp = 1;
   }
   if (stamp == std::numeric_limits<int>::max())
   {
      std::fill(visited_stamp.begin(), visited_stamp.end(), 0);
      stamp = 1;
   }
   const int current_stamp = stamp++;

   bfs_queue.clear();
   bfs_queue.push_back(start_elem_id);
   visited_stamp[start_elem_id] = current_stamp;

   int count = 0;

   thread_local mfem::IsoparametricTransformation eltrans;
   thread_local mfem::InverseElementTransformation inv_tr;
   thread_local bool inv_tr_initialized = false;
   if (!inv_tr_initialized)
   {
      inv_tr.SetInitialGuessType(mfem::InverseElementTransformation::Center);
      inv_tr.SetSolverType(
          mfem::InverseElementTransformation::NewtonElementProject);
      inv_tr_initialized = true;
   }

   std::size_t queue_head = 0;
   while (queue_head < bfs_queue.size())
   {
      int elem_id = bfs_queue[queue_head++];
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
         if (nbrs[i] >= 0 && visited_stamp[nbrs[i]] != current_stamp)
         {
            visited_stamp[nbrs[i]] = current_stamp;
            bfs_queue.push_back(nbrs[i]);
         }
      }
   }

   if (visited_count) { *visited_count = count; }
   return -1;
}

#endif // FIND_ELEMENT_BFS_H
