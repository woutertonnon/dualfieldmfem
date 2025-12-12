# Make an automatic library - will be static or dynamic based on user setting
add_library(boundaryoperatorslib BoundaryOperators.cpp "../include/BoundaryOperators.h")

# We need this directory, and users of our library will need it too
target_include_directories(boundaryoperatorslib PUBLIC ../include /home/wtonnon/VisualStudioProjects/dualfieldmfem/build/extern/mfem)
target_link_directories(boundaryoperatorslib PRIVATE /home/wtonnon/VisualStudioProjects/dualfieldmfem/build/extern/mfem)