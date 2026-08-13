// cube_hole.geo
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 1};
// Z-range made artificially larger to ensure a clean boolean cut through the faces
Box(2) = {0.25, 0.25, -0.1, 0.5, 0.5, 1.2}; 

BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

Physical Surface("boundary") = Surface In BoundingBox{-0.1, -0.1, -0.1, 1.1, 1.1, 1.1};
Physical Volume("domain") = {3};

Mesh.CharacteristicLengthMin = 0.5;
