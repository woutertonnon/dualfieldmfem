// tetra.geo
lc = 1.0;
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {0.5, Sqrt(3)/2, 0, lc};
Point(4) = {0.5, Sqrt(3)/6, Sqrt(2/3), lc};

Line(1) = {1, 2}; Line(2) = {2, 3}; Line(3) = {3, 1};
Line(4) = {1, 4}; Line(5) = {2, 4}; Line(6) = {3, 4};

Curve Loop(1) = {1, 2, 3};      Plane Surface(1) = {1};
Curve Loop(2) = {1, 5, -4};     Plane Surface(2) = {2};
Curve Loop(3) = {2, 6, -5};     Plane Surface(3) = {3};
Curve Loop(4) = {3, 4, -6};     Plane Surface(4) = {4};

Surface Loop(1) = {1, 2, 3, 4}; Volume(1) = {1};

Physical Surface("boundary") = {1, 2, 3, 4};
Physical Volume("domain") = {1};

Mesh.CharacteristicLengthMax = 0.5;
