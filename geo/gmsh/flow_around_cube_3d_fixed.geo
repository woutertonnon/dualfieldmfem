SetFactory("OpenCASCADE");

L      = 1.0;
halfL  = L/2.0;

xmin   = -5.0*L;
xmax   = 22.5*L;
ymin   = -4.0*L;
ymax   =  4.0*L;
zmin   = -2.5*L;
zmax   =  2.5*L;

eps    = 1e-3*L;

lc_far  = 1.50*L;
lc_wake = 1.25*L;
lc_near = 1.10*L;
lc_cube = 1.00*L;

// Square cylinder (cube) obstacle centered at the origin.
cube_h    = 0.70*(zmax-zmin);
cube_zmin = -0.5*cube_h;
cube_zmax =  0.5*cube_h;

Box(1) = {xmin, ymin, zmin, xmax-xmin, ymax-ymin, zmax-zmin};
Box(2) = {-halfL, -halfL, cube_zmin, L, L, cube_h};

fluidVol[] = BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; };

// Boundary identification with correct OpenCASCADE selection syntax
inlet[]  = Surface In BoundingBox{ xmin-eps, ymin-eps, zmin-eps, xmin+eps, ymax+eps, zmax+eps };
outlet[] = Surface In BoundingBox{ xmax-eps, ymin-eps, zmin-eps, xmax+eps, ymax+eps, zmax+eps };

yminS[]  = Surface In BoundingBox{ xmin-eps, ymin-eps, zmin-eps, xmax+eps, ymin+eps, zmax+eps };
ymaxS[]  = Surface In BoundingBox{ xmin-eps, ymax-eps, zmin-eps, xmax+eps, ymax+eps, zmax+eps };
zminS[]  = Surface In BoundingBox{ xmin-eps, ymin-eps, zmin-eps, xmax+eps, ymax+eps, zmin+eps };
zmaxS[]  = Surface In BoundingBox{ xmin-eps, ymin-eps, zmax-eps, xmax+eps, ymax+eps, zmax+eps };

cube[] = Surface In BoundingBox{ -halfL-eps, -halfL-eps, cube_zmin-eps,
                                  halfL+eps,  halfL+eps, cube_zmax+eps };

sideWalls[] = {yminS[], ymaxS[], zminS[], zmaxS[]};

Physical Volume("fluid")       = {fluidVol[]};
Physical Surface("inlet")      = {inlet[]};
Physical Surface("outlet")     = {outlet[]};
Physical Surface("side_walls") = {sideWalls[]};
Physical Surface("cylinder")   = {cube[]};

Field[1] = Box;
Field[1].VIn  = lc_cube;
Field[1].VOut = lc_far;
Field[1].XMin = -1.5*L;
Field[1].XMax =  1.5*L;
Field[1].YMin = -1.5*L;
Field[1].YMax =  1.5*L;
Field[1].ZMin = cube_zmin - 0.5*L;
Field[1].ZMax = cube_zmax + 0.5*L;

Field[2] = Box;
Field[2].VIn  = lc_wake;
Field[2].VOut = lc_far;
Field[2].XMin = -0.5*L;
Field[2].XMax = 12.0*L;
Field[2].YMin = -2.5*L;
Field[2].YMax =  2.5*L;
Field[2].ZMin = cube_zmin - 0.5*L;
Field[2].ZMax = cube_zmax + 0.5*L;

Field[3] = Min;
Field[3].FieldsList = {1, 2};
Background Field = 3;

Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.Algorithm3D = 1; // Delaunay; does not require Netgen
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 0;

Mesh 3;
