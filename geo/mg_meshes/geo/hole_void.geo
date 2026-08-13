SetFactory("OpenCASCADE");

// 1. Compact Domain (5x3x3)
// Shrinking X from 7 to 5 saves ~28% of the volume, reducing the tet count.
Box(1) = {0, 0, 0, 5, 3, 3};

// 2. The Single Void (1x1x1 internal cavity)
Box(2) = {1, 1, 1, 1, 1, 1};

// 3. The Single Hole (1x1 tunnel passing completely through Y)
Box(3) = {3, -0.5, 1, 1, 4, 1};

// 4. Cut the topology
BooleanDifference(4) = { Volume{1}; Delete; }{ Volume{2, 3}; Delete; };

Physical Volume("domain", 1) = {Volume{:}};
Physical Surface("boundary", 2) = {Surface{:}};

Mesh.MeshSizeMin = 4;
Mesh.MeshSizeMax = 4;

Mesh.Algorithm = 1;
