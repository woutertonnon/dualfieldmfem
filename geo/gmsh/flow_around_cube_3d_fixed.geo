SetFactory("OpenCASCADE");

// Schaefer-Turek-like channel with an offset square cylinder.
// Domain: [0, 2.2] x [0, 0.41] x [0, 0.41]
// Square cylinder center offset to (cx, cy) = (0.2, 0.2) in x-y plane;
// cylinder spans the full z-extent of the domain (true square cylinder).

L      = 0.08;       // cube side length (matches typical cylinder diameter)
halfL  = L/2.0;

xmin   = 0.0;
xmax   = 2.2;
ymin   = 0.0;
ymax   = 0.41;
zmin   = 0.0;
zmax   = 0.41;

// Offset cylinder center (as in the Schaefer-Turek 2D-2 benchmark):
// 0.2 from left wall, 0.2 from bottom wall (=> 0.21 from top wall).
cx     = 0.2;
cy     = 0.2;
cz     = 0.5*(zmin + zmax);

eps    = 1e-4*L;

// Uniform characteristic mesh size throughout the domain.
lc     = 0.1;

// Square cylinder obstacle — spans the full z-extent of the domain.
// Extend slightly past z=zmin and z=zmax so the boolean cut produces clean
// through-holes in the top/bottom faces (avoids co-planar face artefacts).
cube_overshoot = 0.1*L;
cube_zmin = zmin - cube_overshoot;
cube_zmax = zmax + cube_overshoot;
cube_h    = cube_zmax - cube_zmin;

Box(1) = {xmin, ymin, zmin, xmax-xmin, ymax-ymin, zmax-zmin};
Box(2) = {cx - halfL, cy - halfL, cube_zmin, L, L, cube_h};

fluidVol[] = BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; };

// Boundary identification with correct OpenCASCADE selection syntax
inlet[]  = Surface In BoundingBox{ xmin-eps, ymin-eps, zmin-eps, xmin+eps, ymax+eps, zmax+eps };
outlet[] = Surface In BoundingBox{ xmax-eps, ymin-eps, zmin-eps, xmax+eps, ymax+eps, zmax+eps };

yminS[]  = Surface In BoundingBox{ xmin-eps, ymin-eps, zmin-eps, xmax+eps, ymin+eps, zmax+eps };
ymaxS[]  = Surface In BoundingBox{ xmin-eps, ymax-eps, zmin-eps, xmax+eps, ymax+eps, zmax+eps };
zminS[]  = Surface In BoundingBox{ xmin-eps, ymin-eps, zmin-eps, xmax+eps, ymax+eps, zmin+eps };
zmaxS[]  = Surface In BoundingBox{ xmin-eps, ymin-eps, zmax-eps, xmax+eps, ymax+eps, zmax+eps };

cube[] = Surface In BoundingBox{ cx-halfL-eps, cy-halfL-eps, cube_zmin-eps,
                                  cx+halfL+eps, cy+halfL+eps, cube_zmax+eps };

sideWalls[] = {yminS[], ymaxS[], zminS[], zmaxS[]};

Physical Volume("fluid")       = {fluidVol[]};
Physical Surface("inlet")      = {inlet[]};
Physical Surface("outlet")     = {outlet[]};
Physical Surface("side_walls") = {sideWalls[]};
Physical Surface("cylinder")   = {cube[]};

// Uniform mesh size: no size fields, constant characteristic length.
Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;
Mesh.Algorithm3D = 1; // Delaunay; does not require Netgen
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 0;

Mesh 3;
