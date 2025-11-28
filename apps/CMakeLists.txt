



# dirichlet problem
#add_executable(${PROJECT_NAME} dualfieldnavierstokes.cpp)
add_executable(${PROJECT_NAME} MEHC_matrixfree.cpp)







# links MFEM directory
target_include_directories(${PROJECT_NAME} PUBLIC extern/mfem)
target_link_directories(${PROJECT_NAME} PRIVATE extern/mfem)

# Good defaults for perf + perfanno
target_compile_options(${PROJECT_NAME} PRIVATE
#    -O2                     # optimize
    -g                      # debug symbols
    -fno-omit-frame-pointer # keep frame pointers for better stack traces
    -fno-optimize-sibling-calls # helps with nicer call stacks
)

target_link_options(${PROJECT_NAME} PRIVATE
    -fno-omit-frame-pointer
)


# link libraries
target_link_libraries(${PROJECT_NAME} mfem)




