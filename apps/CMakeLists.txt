



# dirichlet problem
#add_executable(${PROJECT_NAME} dualfieldnavierstokes.cpp)
add_executable(${PROJECT_NAME} mumpstest.cpp)







# links MFEM directory
target_include_directories(${PROJECT_NAME} PUBLIC extern/mfem)
target_link_directories(${PROJECT_NAME} PRIVATE extern/mfem)


# link libraries
target_link_libraries(${PROJECT_NAME} mfem)




