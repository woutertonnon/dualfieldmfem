// cylinder.geo
SetFactory("OpenCASCADE");
// Cylinder(tag) = {X, Y, Z,  DX, DY, DZ,  Radius};
Cylinder(1) = {0, 0, 0, 0, 0, 1, 0.5};

Physical Surface("boundary") = {1, 2, 3};
Physical Volume("domain") = {1};

Mesh.CharacteristicLengthMin = 0.5;
