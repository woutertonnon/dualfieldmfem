cmake_minimum_required (VERSION 2.9)
project(MEHCscheme)



set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
find_package(Git QUIET)
FIND_PACKAGE(Boost COMPONENTS program_options filesystem REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
# Use the bundled extern/mfem headers/build instead of an external ../mfem/build
include_directories("${PROJECT_SOURCE_DIR}/include")
link_directories("${CMAKE_BINARY_DIR}/src")


add_subdirectory(extern)
add_subdirectory(apps)


