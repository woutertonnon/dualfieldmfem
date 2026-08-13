// cube.geo
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 1};

Physical Surface("boundary") = {1, 2, 3, 4, 5, 6};
Physical Volume("domain") = {1};

Mesh.CharacteristicLengthMin = 1;
