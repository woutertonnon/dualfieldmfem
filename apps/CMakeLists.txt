



# dirichlet problem
add_executable(${PROJECT_NAME} dualfieldnavierstokes.cpp)







# links MFEM directory
target_include_directories(${PROJECT_NAME} PUBLIC extern/mfem)
target_link_directories(${PROJECT_NAME} PRIVATE extern/mfem)

# link utils file
#add_library(UTILS utils.cpp)
#target_include_directories(UTILS PUBLIC extern/mfem)
#target_link_directories(UTILS PRIVATE extern/mfem)
#target_link_libraries(UTILS mfem)

# link libraries
#target_link_libraries(${PROJECT_NAME} UTILS)
target_link_libraries(${PROJECT_NAME} mfem)




