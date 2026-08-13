// cube_void.geo
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 1};
Box(2) = {0.25, 0.25, 0.25, 0.5, 0.5, 0.5};

BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

Physical Surface("outer_boundary") = Surface In BoundingBox{-0.1, -0.1, -0.1, 1.1, 1.1, 1.1};
Physical Surface("inner_boundary") = Surface In BoundingBox{0.2, 0.2, 0.2, 0.8, 0.8, 0.8};
Physical Volume("domain") = {3};

Mesh.CharacteristicLengthMin = 1;
