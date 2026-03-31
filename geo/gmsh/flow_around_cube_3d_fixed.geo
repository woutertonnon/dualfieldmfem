SetFactory("OpenCASCADE");

L      = 1.0;
halfL  = L/2.0;

xmin   = -5.0*L;
xmax   = 22.5*L;
ymin   = -5.0*L;
ymax   =  5.0*L;
zmin   = -5.0*L;
zmax   =  5.0*L;

eps    = 1e-3*L;

lc_far  = 1.10*L;
lc_wake = 0.30*L;
lc_near = 0.16*L;
lc_cyl  = 0.10*L;

// Finite cylinder obstacle centered at the origin.
// Domain height is (zmax-zmin)=10L, so 80% span gives 8L.
cyl_r    = halfL;
cyl_h    = 0.70*(zmax-zmin);
cyl_zmin = -0.5*cyl_h;
cyl_zmax =  0.5*cyl_h;

Box(1) = {xmin, ymin, zmin, xmax-xmin, ymax-ymin, zmax-zmin};
Cylinder(2) = {0, 0, cyl_zmin, 0, 0, cyl_h, cyl_r};

fluidVol[] = BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; };

// Boundary identification with correct OpenCASCADE selection syntax
inlet[]  = Surface In BoundingBox{ xmin-eps, ymin-eps, zmin-eps, xmin+eps, ymax+eps, zmax+eps };
outlet[] = Surface In BoundingBox{ xmax-eps, ymin-eps, zmin-eps, xmax+eps, ymax+eps, zmax+eps };

yminS[]  = Surface In BoundingBox{ xmin-eps, ymin-eps, zmin-eps, xmax+eps, ymin+eps, zmax+eps };
ymaxS[]  = Surface In BoundingBox{ xmin-eps, ymax-eps, zmin-eps, xmax+eps, ymax+eps, zmax+eps };
zminS[]  = Surface In BoundingBox{ xmin-eps, ymin-eps, zmin-eps, xmax+eps, ymax+eps, zmin+eps };
zmaxS[]  = Surface In BoundingBox{ xmin-eps, ymin-eps, zmax-eps, xmax+eps, ymax+eps, zmax+eps };

cylinder[] = Surface In BoundingBox{ -cyl_r-eps,  -cyl_r-eps,  cyl_zmin-eps,
                                      cyl_r+eps,   cyl_r+eps,  cyl_zmax+eps };

sideWalls[] = {yminS[], ymaxS[], zminS[], zmaxS[]};

Physical Volume("fluid")       = {fluidVol[]};
Physical Surface("inlet")      = {inlet[]};
Physical Surface("outlet")     = {outlet[]};
Physical Surface("side_walls") = {sideWalls[]};
Physical Surface("cylinder")   = {cylinder[]};

Field[1] = Box;
Field[1].VIn  = lc_cyl;
Field[1].VOut = lc_far;
Field[1].XMin = -1.2*L;
Field[1].XMax =  1.2*L;
Field[1].YMin = -1.2*L;
Field[1].YMax =  1.2*L;
Field[1].ZMin = cyl_zmin - 0.4*L;
Field[1].ZMax = cyl_zmax + 0.4*L;

Field[2] = Box;
Field[2].VIn  = lc_wake;
Field[2].VOut = lc_far;
Field[2].XMin = -0.2*L;
Field[2].XMax = 10.0*L;
Field[2].YMin = -2.0*L;
Field[2].YMax =  2.0*L;
Field[2].ZMin = cyl_zmin - 0.5*L;
Field[2].ZMax = cyl_zmax + 0.5*L;

Field[3] = Distance;
Field[3].SurfacesList = {cylinder[]};
Field[3].Sampling = 100;

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = lc_near;
Field[4].SizeMax = lc_far;
Field[4].DistMin = 0.35*L;
Field[4].DistMax = 2.50*L;

Field[5] = Min;
Field[5].FieldsList = {1, 2, 4};
Background Field = 5;

Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.Algorithm3D = 1; // Delaunay; does not require Netgen
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 0;

Mesh 3;
