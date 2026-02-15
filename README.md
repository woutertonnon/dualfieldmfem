# dualfieldmfem (in active development)

A discretization of the incompressible Navier--Stokes problem, 
implemented with MFEM and inspired by the MEHC dual-field formulation 
described in:

https://arxiv.org/abs/2104.13023

------------------------------------------------------------------------

## What's in this branch

-   CMake-based build (CMake ≥ 3.20, C++17)
-   Bundled dependencies via submodules:
    -   MFEM (extern/mfem)
    -   BoundaryIntegralLib (extern/boundaryintegrallib)
-   Application target:
    -   singlefield_navierstokes_nitsche
-   Optional tests using GoogleTest

------------------------------------------------------------------------

## Requirements

### Build tools

-   CMake 3.20+
-   C++ compiler with C++17 support

### Required system libraries

-   Boost (program_options, filesystem)
-   SuiteSparse
-   Metis 5

### Optional

-   GoogleTest (for tests)

------------------------------------------------------------------------

## Clone (with submodules, non-recursive)

Clone the repository and initialize submodules:

    git clone --branch cleanup --recurse-submodules https://github.com/woutertonnon/dualfieldmfem.git

If you already cloned without submodules:

    git submodule update --init

------------------------------------------------------------------------

## Build

From the repository root:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j

------------------------------------------------------------------------

## Run the application

    ./build/singlefield_navierstokes_nitsche --help

------------------------------------------------------------------------

## Tests

Enable and run tests:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
    cmake --build build -j
    ctest --test-dir build --output-on-failure

If GoogleTest is not installed, tests are skipped.

------------------------------------------------------------------------

## MFEM configuration

MFEM is configured via CMake cache variables in extern/CMakeLists.txt.

Defaults:

-   MFEM_USE_METIS_5 = ON
-   MFEM_USE_SUITESPARSE = ON
-   MFEM_USE_MPI = OFF

Note that Metis 5 and SuiteSparse are used and can thus not be disabled.

------------------------------------------------------------------------

## Documentation

For now, only documentation for the python libraries controlling the "testbench" and the IO with the C++ code are documented:

[https://woutertonnon.github.io/dualfieldmfem/api.html#](https://woutertonnon.github.io/dualfieldmfem/api.html#)

------------------------------------------------------------------------

## Project layout

-   apps/ -- application executables
-   extern/ -- submodules (MFEM, BoundaryIntegralLib)
-   include/ -- project headers
-   tests/ -- GTest files
-   scripts/ -- benchmark and IO library (python)

------------------------------------------------------------------------
