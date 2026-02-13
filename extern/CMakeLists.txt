set(MFEM_USE_METIS_5 ON CACHE BOOL "Enable METIS 5")
set(MFEM_USE_SUITESPARSE ON CACHE BOOL "Use SuiteSparse")
set(MFEM_USE_MPI  OFF CACHE BOOL "Force MFEM to enable MPI (required for MUMPS)" FORCE)
message(STATUS "Forcing MFEM configuration: MFEM_USE_MPI=${MFEM_USE_MPI}, MFEM_USESUITESPARSE=${MFEM_USE_SUITESPARSE}, MFEM_USE_METIS_5=${MFEM_USE_METIS_5}")

add_subdirectory(mfem)
add_subdirectory(boundaryintegrallib)
