cmake_minimum_required (VERSION 4.2)

project(dualfieldmfem VERSION 0.1.0 LANGUAGES C CXX)

# Put all binaries in the build dir root (keeps your original behavior).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Testing
include(CTest)  # defines BUILD_TESTING option and calls enable_testing() when BUILD_TESTING=ON

# Dependencies
find_package(Boost 1.65 REQUIRED COMPONENTS program_options filesystem)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(dualfield_project_options INTERFACE)

# Project-wide include dirs used by your own targets.
target_include_directories(dualfield_project_options
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "/usr/include/suitesparse"   # consider making this a find_package later
    # "${CMAKE_SOURCE_DIR}/extern/spectra/include"
    # ${EIGEN3_INCLUDE_DIR}
)

include_directories(
    "${PROJECT_SOURCE_DIR}/extern/spectra/include"
    ${EIGEN3_INCLUDE_DIR}
)

target_compile_features(dualfield_project_options INTERFACE cxx_std_23) # Same as MFEM

# Boost usage requirements (modern: prefer imported targets if available)
# If your Boost provides Boost::program_options, Boost::filesystem targets, use them:
if(TARGET Boost::program_options AND TARGET Boost::filesystem)
  target_link_libraries(dualfield_project_options
    INTERFACE
      Boost::program_options
      Boost::filesystem
  )
else()
  # Fallback for older Boost / older CMake setups
  target_include_directories(dualfield_project_options INTERFACE "${Boost_INCLUDE_DIRS}")
  target_link_libraries(dualfield_project_options INTERFACE "${Boost_LIBRARIES}")
endif()


add_subdirectory(extern)
add_subdirectory(apps)
add_subdirectory(src)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
