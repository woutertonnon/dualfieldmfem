find_package(GTest QUIET)
if (NOT GTEST_FOUND)
    message(WARNING "GoogleTest not found. Install libgtest-dev (and build the libs) to enable the test target.")
    return()
endif()

include_directories(${GTEST_INCLUDE_DIRS})

add_executable(boundaryoperators_tests
    BoundaryOperatorsTests.cpp
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

add_test(NAME boundaryoperators_tests COMMAND boundaryoperators_tests)

add_executable(stokes_tests
    StokesTests.cpp
)

target_link_libraries(stokes_tests
    PRIVATE
        boundaryoperatorslib
        mfem
        $<$<TARGET_EXISTS:GTest::gtest>:GTest::gtest>
        $<$<TARGET_EXISTS:GTest::gtest_main>:GTest::gtest_main>
        ${GTEST_LIBRARIES}
        ${GTEST_MAIN_LIBRARIES}
        pthread
)

add_test(NAME stokes_tests COMMAND stokes_tests)
