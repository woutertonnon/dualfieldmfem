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
