// corner.geo
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 1};
Box(2) = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

Physical Surface("boundary") = Surface In BoundingBox{-0.1, -0.1, -0.1, 1.1, 1.1, 1.1};
Physical Volume("domain") = {3};

Mesh.CharacteristicLengthMin = 10.;
