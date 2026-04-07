# --- Base Libraries ---
include_directories(../include)

add_library(incidence incidence.cpp)
target_link_libraries(incidence PUBLIC mfem)

add_library(stokesoperator StokesOperator.cpp)
target_link_libraries(stokesoperator PUBLIC boundaryoperatorslib incidence mfem)

add_library(stokesdgs StokesDGS.cpp)
target_link_libraries(stokesdgs PUBLIC stokesoperator mfem)

add_library(stokesmg
    StokesMG.cpp StokesDGS.cpp StokesOperator.cpp
    incidence.cpp)
target_link_libraries(stokesmg PUBLIC boundaryoperatorslib mfem)
