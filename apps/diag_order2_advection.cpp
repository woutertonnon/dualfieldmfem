// Standalone diagnostic for the order-2 (Rapetti small-edge) semi-Lagrangian
// advection. Isolates the ND2 reconstruction from the time integration and the
// Stokes solve:
//
//   (1) ND2 must represent a constant 1-form exactly (ProjectCoefficient).
//   (2) Zero-velocity Apply must reproduce omega_old verbatim (transported
//       small edges == original small edges => pure I_{h,2} round-trip).
//   (3) Same checks for an affine field (exercises the genuine ND2 DOFs).
//
// A non-identity round-trip localizes the bug to the Rapetti pipeline.

#include <cmath>
#include <cstdio>

#include "SemiLagrangianAdvectionOrder2.h"
#include "mfem.hpp"

using namespace mfem;

namespace
{
double L2Err(GridFunction &gf, VectorFunctionCoefficient &c)
{
    return gf.ComputeL2Error(c);
}

double MaxDofDiff(const GridFunction &a, const GridFunction &b)
{
    double m = 0.0;
    for (int i = 0; i < a.Size(); ++i)
    {
        m = std::max(m, std::abs(a(i) - b(i)));
    }
    return m;
}

void RunCase(const char *label, int dim,
             std::function<void(const Vector &, Vector &)> field)
{
    Mesh mesh = (dim == 2)
                    ? Mesh::MakeCartesian2D(4, 4, Element::TRIANGLE, true, 1.0, 1.0, false)
                    : Mesh::MakeCartesian3D(2, 2, 2, Element::TETRAHEDRON, 1.0, 1.0, 1.0, false);
    ND_FECollection fec(2, dim);
    FiniteElementSpace nd(&mesh, &fec);

    VectorFunctionCoefficient coeff(
        dim, [&field](const Vector &x, double, Vector &v) { v.SetSize(x.Size()); field(x, v); });

    GridFunction omega_old(&nd), omega_new(&nd);
    omega_old.ProjectCoefficient(coeff);
    double proj_err = L2Err(omega_old, coeff);

    SemiLagrangianAdvection1FormOrder2<2> adv(nd);
    SemiLagrangianAdvection1FormOrder2<2>::VelocityFunc zero =
        [dim](const Vector &, double, int, Vector &v) { v.SetSize(dim); v = 0.0; };
    SemiLagrangianAdvection1FormOrder2<2>::BoundaryFunc bnd =
        [dim](const Vector &, double, int, Vector &v) { v.SetSize(dim); v = 0.0; };

    omega_new = 0.0;
    adv.Apply(zero, bnd, 0.4, 0.2, omega_old, omega_new, 2);

    double rt_dof = MaxDofDiff(omega_new, omega_old);
    double rt_l2 = L2Err(omega_new, coeff);

    std::printf("[%s, %dD]  ndofs=%d\n", label, dim, nd.GetNDofs());
    std::printf("    projection L2 err          : %.3e\n", proj_err);
    std::printf("    zero-vel round-trip dofdiff: %.3e\n", rt_dof);
    std::printf("    zero-vel round-trip L2 err : %.3e\n", rt_l2);
}
}  // namespace

int main(int argc, char **argv)
{
    Mpi::Init(argc, argv);
    Hypre::Init();

    RunCase("constant", 2, [](const Vector &, Vector &v) { v[0] = 1.0; v[1] = 0.0; });
    RunCase("affine", 2, [](const Vector &x, Vector &v)
            { v[0] = 1.0 + 0.35 * x[0] - 0.15 * x[1]; v[1] = -0.2 + 0.10 * x[0] + 0.45 * x[1]; });
    RunCase("constant", 3, [](const Vector &, Vector &v) { v[0] = 1.0; v[1] = 0.0; v[2] = 0.0; });

    return 0;
}
