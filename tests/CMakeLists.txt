if(NOT BUILD_TESTING)
  return()
endif()

include(CTest)

# GoogleTest
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  message(WARNING "GoogleTest not found. Tests will be skipped. "
                  "Install GTest (e.g. libgtest-dev) to enable tests.")
  return()
endif()

find_package(Threads REQUIRED)

add_executable(stokes_tests
  StokesTests.cpp
  SemiLagrangianAdvectionTests.cpp
)

target_link_libraries(stokes_tests
  PRIVATE
    mfem
    boundaryoperatorslib
    Threads::Threads
    dualfield_project_options
)

target_link_libraries(stokes_tests PRIVATE GTest::gtest GTest::gtest_main)


add_test(NAME stokes_tests COMMAND stokes_tests)

# MPI semi-Lagrangian tests (separate executable with custom main)
add_executable(semilag_mpi_tests
  SemiLagrangianMPITests.cpp
)
target_link_libraries(semilag_mpi_tests
  PRIVATE
    mfem
    boundaryoperatorslib
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    Threads::Threads
    dualfield_project_options
    GTest::gtest
)
# Run with 1, 2, and 4 MPI ranks
add_test(NAME semilag_mpi_np1 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 $<TARGET_FILE:semilag_mpi_tests>)
add_test(NAME semilag_mpi_np2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:semilag_mpi_tests>)
add_test(NAME semilag_mpi_np4 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:semilag_mpi_tests>)

add_test(
  NAME dualfield_smoke_umfpack
  COMMAND $<TARGET_FILE:dualfieldnavierstokes_nitsche>
          -c data/config/ConstantField/ConstantField_smoke_umfpack.json
)
set_tests_properties(dualfield_smoke_umfpack PROPERTIES
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_test(
  NAME dualfield_smoke_amsads
  COMMAND $<TARGET_FILE:dualfieldnavierstokes_nitsche>
          -c data/config/ConstantField/ConstantField_smoke_amsads.json
)
set_tests_properties(dualfield_smoke_amsads PROPERTIES
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# MG tests
include_directories(../include)

add_executable(incidence_tests
    IncidenceTest.cpp
)
add_executable(stokesoperator_tests
    StokesOperatorTest.cpp
)
add_executable(stokesdgs_tests
    StokesDGSTest.cpp
)
add_executable(stokesmg_tests
    StokesMGTest.cpp
)

target_link_libraries(incidence_tests
    PRIVATE
        incidence
        mfem
        # Prefer imported targets when available (CMake >= 3.5)
        $<$<TARGET_EXISTS:GTest::gtest>:GTest::gtest>
        $<$<TARGET_EXISTS:GTest::gtest_main>:GTest::gtest_main>
        # Fallback for older FindGTest modules
        ${GTEST_LIBRARIES}
        ${GTEST_MAIN_LIBRARIES}
        pthread
)

target_link_libraries(boundaryoperators_tests
    PRIVATE
        boundaryoperatorslib
        mfem
        # Prefer imported targets when available (CMake >= 3.5)
        $<$<TARGET_EXISTS:GTest::gtest>:GTest::gtest>
        $<$<TARGET_EXISTS:GTest::gtest_main>:GTest::gtest_main>
        # Fallback for older FindGTest modules
        ${GTEST_LIBRARIES}
        ${GTEST_MAIN_LIBRARIES}
        pthread
)

target_link_libraries(stokesoperator_tests
    PRIVATE
        boundaryoperatorslib
        mfem
        incidence
        stokesoperator
        # Prefer imported targets when available (CMake >= 3.5)
        $<$<TARGET_EXISTS:GTest::gtest>:GTest::gtest>
        $<$<TARGET_EXISTS:GTest::gtest_main>:GTest::gtest_main>
        # Fallback for older FindGTest modules
        ${GTEST_LIBRARIES}
        ${GTEST_MAIN_LIBRARIES}
        pthread
)

target_link_libraries(stokesdgs_tests
    PRIVATE
        mfem
        stokesdgs
        # Prefer imported targets when available (CMake >= 3.5)
        $<$<TARGET_EXISTS:GTest::gtest>:GTest::gtest>
        $<$<TARGET_EXISTS:GTest::gtest_main>:GTest::gtest_main>
        # Fallback for older FindGTest modules
        ${GTEST_LIBRARIES}
        ${GTEST_MAIN_LIBRARIES}
        pthread
)

target_link_libraries(stokesmg_tests
    PRIVATE
        stokesoperator
        stokesdgs
        stokesmg
        mfem
        # Prefer imported targets when available (CMake >= 3.5)
        $<$<TARGET_EXISTS:GTest::gtest>:GTest::gtest>
        $<$<TARGET_EXISTS:GTest::gtest_main>:GTest::gtest_main>
        # Fallback for older FindGTest modules
        ${GTEST_LIBRARIES}
        ${GTEST_MAIN_LIBRARIES}
        pthread
)

add_test(NAME incidence_tests COMMAND incidence_tests)
add_test(NAME stokesoperator_tests COMMAND stokesoperator_tests)
add_test(NAME stokesdgs_tests COMMAND stokesdgs_tests)
add_test(NAME stokesmg_tests COMMAND stokesmg_tests)
