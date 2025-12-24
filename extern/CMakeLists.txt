# Prefer the system-wide OpenMPI installation when searching for MPI
# libraries so FindMPI does not pick up stray user-local MPI libs (e.g. in
# ~/.local). These are hints only; we avoid forcing the Fortran MPI wrapper
# or enabling MUMPS unconditionally because that forces FindMPI Fortran
# checks which can fail on systems with mixed MPI runtimes.


# Enable MPI in MFEM by default, but do NOT force MUMPS/Fortran unless the
# user explicitly requests it (via a cached MFEM_USE_MUMPS or by setting
# MFEM_USE_MUMPS on the command line). Forcing MUMPS causes FindMPI to run
# Fortran link tests which may fail when different MPI runtimes are mixed.
# Enable fetching for selected TPLs; we want MFEM to fetch HYPRE but avoid
# fetching METIS (to prevent ABI mismatches) and avoid optional packages that
# would require additional system de$pendencies such as STRUMPACK/SuiteSparse.
#set(MFEM_FETCH_TPLS ON CACHE BOOL "Fetch MFEM TPLs")
#set(MFEM_FETCH_HYPRE ON CACHE BOOL "Fetch HYPRE")
#set(MFEM_FETCH_METIS ON CACHE BOOL "Do not fetch METIS")
# Do not force SuiteSparse off here; allow the top-level caller or command
# line to enable `-DMFEM_USE_SUITESPARSE=ON` when system SuiteSparse is
# available. Only set a default if the variable was not provided.
#set(MFEM_USE_MPI ON CACHE BOOL "Enable MPI in MFEM")
#set(MFEM_USE_METIS ON CACHE BOOL "Enable METIS")
set(MFEM_USE_METIS_5 ON CACHE BOOL "Enable METIS 5")
#set(METIS_INCLUDE_DIRS "/home/wtonnon/local/include"
#    CACHE PATH "METIS include directory")
#set(METIS_LIBRARIES
#    "/home/wtonnon/local/lib/libmetis.a;/home/wtonnon/VisualStudioProjects/GKlib/lib/libGKlib.so"
#    CACHE STRING "METIS libraries")
set(MFEM_USE_SUITESPARSE ON CACHE BOOL "Use SuiteSparse)")
#set(MFEM_USE_STRUMPACK ON CACHE BOOL "Enable STRUMPACK" )
#set(MFEM_USE_SUPERLU NO CACHE BOOL "Enable SUPERLU")
# cmake ..   -DMFEM_USE_MPI=YES   -DMFEM_USE_METIS=YES   -DMFEM_USE_METIS_5=YES   -DMETIS_INCLUDE_DIRS=/home/wtonnon/local/include   -DMETIS_LIBRARIES="/home/wtonnon/local/lib/libmetis.a;/home/wtonnon/VisualStudioProjects/GKlib/lib/libGKlib.so"



add_subdirectory(mfem)
