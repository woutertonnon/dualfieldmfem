// ball.geo
SetFactory("OpenCASCADE");
Sphere(1) = {0, 0, 0, 0.5};

Physical Surface("boundary") = {1};
Physical Volume("domain") = {1};

Mesh.Algorithm = 1;
Mesh.CharacteristicLengthMin = 0.5;
