set(MFEM_USE_METIS_5 ON CACHE BOOL "Enable METIS 5")
set(MFEM_USE_SUITESPARSE ON CACHE BOOL "Use SuiteSparse")
set(MFEM_USE_MPI  ON CACHE BOOL "Force MFEM to enable MPI (required for Hypre AMS/ADS)" FORCE)
set(MFEM_FETCH_HYPRE ON CACHE BOOL "Fetch/build Hypre when enabling MPI" FORCE)
set(MFEM_USE_OPENMP ON CACHE BOOL "Enable OpenMP backend for parallel mat-vec" FORCE)
set(MFEM_USE_LEGACY_OPENMP OFF CACHE BOOL "Enable legacy OpenMP in SparseMatrix::Mult" FORCE)
set(MFEM_THREAD_SAFE ON CACHE BOOL "Thread-safe mode (required for OpenMP)" FORCE)
message(STATUS "Forcing MFEM configuration: MFEM_USE_MPI=${MFEM_USE_MPI}, MFEM_USESUITESPARSE=${MFEM_USE_SUITESPARSE}, MFEM_USE_METIS_5=${MFEM_USE_METIS_5}")

add_subdirectory(mfem)

# The boundary-integral library is normally the lowercase git submodule
# (extern/boundaryintegrallib). Some checkouts instead have it as a manual
# clone under the GitHub-cased name (extern/BoundaryIntegralLib, e.g. on Euler).
# Use whichever is populated so this file never needs a per-machine edit.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/boundaryintegrallib/CMakeLists.txt")
    add_subdirectory(boundaryintegrallib)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegralLib/CMakeLists.txt")
    add_subdirectory(BoundaryIntegralLib)
else()
    message(FATAL_ERROR
        "boundary-integral library not found under extern/. Run "
        "'git submodule update --init extern/boundaryintegrallib', or place a "
        "clone at extern/BoundaryIntegralLib.")
endif()
