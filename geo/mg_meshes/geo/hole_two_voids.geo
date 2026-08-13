SetFactory("OpenCASCADE");

// 1. Domain (7x3x3)
// This is the smallest integer bounding box that allows 1-unit thick walls.
Box(1) = {0, 0, 0, 7, 3, 3};

// 2. The Voids (1x1x1)
Box(2) = {1, 1, 1, 1, 1, 1};
Box(3) = {5, 1, 1, 1, 1, 1};

// 3. The Hole (1x1 tunnel passing completely through Y)
Box(4) = {3, -0.5, 1, 1, 4, 1};

// 4. Cut the topology
BooleanDifference(5) = { Volume{1}; Delete; }{ Volume{2, 3, 4}; Delete; };

Physical Volume("domain", 1) = {Volume{:}};
Physical Surface("boundary", 2) = {Surface{:}};

Mesh.MeshSizeMin = 4;
Mesh.MeshSizeMax = 4;

Mesh.Algorithm = 1; 
