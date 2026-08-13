// cube_two_voids.geo
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 1};
Box(2) = {0.125, 0.125, 0.125, 0.25, 0.25, 0.25};
Box(3) = {0.625, 0.625, 0.625, 0.25, 0.25, 0.25};

BooleanDifference(4) = { Volume{1}; Delete; }{ Volume{2, 3}; Delete; };

Physical Surface("boundary") = Surface In BoundingBox{-0.1, -0.1, -0.1, 1.1, 1.1, 1.1};
Physical Volume("domain") = {4};

Mesh.CharacteristicLengthMin = 0.5;
