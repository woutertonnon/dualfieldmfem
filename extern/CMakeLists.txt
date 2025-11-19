# Prefer the system-wide OpenMPI installation when searching for MPI
# libraries so FindMPI does not pick up stray user-local MPI libs (e.g. in
# ~/.local). These are hints only; we avoid forcing the Fortran MPI wrapper
# or enabling MUMPS unconditionally because that forces FindMPI Fortran
# checks which can fail on systems with mixed MPI runtimes.


# Enable MPI in MFEM by default, but do NOT force MUMPS/Fortran unless the
# user explicitly requests it (via a cached MFEM_USE_MUMPS or by setting
# MFEM_USE_MUMPS on the command line). Forcing MUMPS causes FindMPI to run
# Fortran link tests which may fail when different MPI runtimes are mixed.
set(MFEM_USE_MPI YES CACHE BOOL "Enable MPI in MFEM")
# Enable fetching for selected TPLs; we want MFEM to fetch HYPRE but avoid
# fetching METIS (to prevent ABI mismatches) and avoid optional packages that
# would require additional system de$pendencies such as STRUMPACK/SuiteSparse.
set(MFEM_FETCH_TPLS NO CACHE BOOL "Fetch MFEM TPLs")
set(MFEM_FETCH_METIS NO CACHE BOOL "Do not fetch METIS")
set(MFEM_FETCH_HYPRE YES CACHE BOOL "Do not fetch METIS")
# Do not force SuiteSparse off here; allow the top-level caller or command
# line to enable `-DMFEM_USE_SUITESPARSE=ON` when system SuiteSparse is
# available. Only set a default if the variable was not provided.
set(MFEM_USE_SUITESPARSE NO CACHE BOOL "Enable SuiteSparse")
set(MFEM_USE_STRUMPACK OFF CACHE BOOL "Enable STRUMPACK" )
set(MFEM_USE_SUPERLU YES CACHE BOOL "Enable SUPERLU")


add_subdirectory(mfem)
