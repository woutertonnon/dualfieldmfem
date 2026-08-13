SetFactory("OpenCASCADE");

// 1. Create the 7 cubic blocks (1x1x1 each)
Box(1) = {-1, -1, -1, 1, 1, 1};
Box(2) = { 0, -1, -1, 1, 1, 1};
Box(3) = {-1,  0, -1, 1, 1, 1};
Box(4) = { 0,  0, -1, 1, 1, 1};
Box(5) = {-1, -1,  0, 1, 1, 1};
Box(6) = { 0, -1,  0, 1, 1, 1};
Box(7) = { 0,  0,  0, 1, 1, 1};

// 2. Merge them together so they share internal faces and nodes
BooleanFragments{ Volume{1:7}; Delete; }{}

// 3. Force a structured, transfinite grid on ALL entities
// The ":" operator safely selects all entities of that type
Transfinite Curve {:} = 3;
Transfinite Surface {:};
Transfinite Volume {:};

// 4. Assign Physical Groups
// Add all 7 volumes to the domain
Physical Volume("The_Domain", 1) = {Volume{:}};

// Use CombinedBoundary to extract ONLY the external faces (ignores internal shared faces)
bnd() = CombinedBoundary{ Volume{:}; };
Physical Surface("The_Boundary", 2) = {bnd[]};
