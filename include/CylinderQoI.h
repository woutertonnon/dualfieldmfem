#ifndef CYLINDER_QOI_H
#define CYLINDER_QOI_H

// Quantities of interest for the Schaefer-Turek flow-around-cylinder benchmark,
// specialised to the H(curl) velocity / H1 pressure (vorticity-based) scheme.
//
// Drag/lift use the natural surface formula in terms of the formulation's
// first-class quantities -- the vorticity omega = curl u (well defined for
// u in H(curl)) and the pressure p:
//
//   F_D = oint_S ( eps * omega * n_y  -  p * n_x ) ds
//   F_L = oint_S ( -eps * omega * n_x -  p * n_y ) ds
//
// where n is the unit normal of the FLUID domain on the cylinder boundary S.
// IMPORTANT: this scheme solves the viscosity-divided momentum, so the stored
// pressure is p_solved = p_phys / eps; the PHYSICAL traction therefore carries
// a single factor eps on BOTH terms:
//   F_D = eps * oint_S ( omega * n_y - p_solved * n_x ) ds, etc.
// CalcOrtho's normal points OUT of the fluid (i.e. INTO the cylinder), the
// opposite of the body-outward convention, so the force on the body flips sign
// -> pass sign = -1.  The dimensionless coefficients are
//   c_{D,L} = 2 F_{D,L} / (Ubar^2 D).
//
// Delta p = p(x_front) - p(x_back) uses a BFS element search for point eval.

#include "mfem.hpp"
#include "FindElementBFS.h"

struct CylinderForces
{
    double cD = 0.0, cL = 0.0, FD = 0.0, FL = 0.0;
};

inline CylinderForces ComputeCylinderForces(
    mfem::GridFunction& u_gf,   // ND velocity
    mfem::GridFunction& p_gf,   // H1 pressure
    int                 cyl_attr,
    double              eps,
    double              Ubar,
    double              D,
    double              sign = 1.0)
{
    mfem::Mesh* mesh = u_gf.FESpace()->GetMesh();
    const int   dim  = mesh->Dimension();

    double FD = 0.0, FL = 0.0;
    mfem::Vector normal(dim), curl;

    for (int be = 0; be < mesh->GetNBE(); ++be)
    {
        if (mesh->GetBdrAttribute(be) != cyl_attr)
            continue;

        mfem::FaceElementTransformations* Tr =
            mesh->GetBdrFaceTransformations(be);
        if (!Tr)
            continue;

        const mfem::FiniteElement& fe = *u_gf.FESpace()->GetFE(Tr->Elem1No);
        const int                  intorder = 2 * fe.GetOrder() + 2;
        const mfem::IntegrationRule& ir =
            mfem::IntRules.Get(Tr->GetGeometryType(), intorder);

        for (int q = 0; q < ir.GetNPoints(); ++q)
        {
            const mfem::IntegrationPoint& ip = ir.IntPoint(q);
            Tr->SetAllIntPoints(&ip);

            Tr->Face->SetIntPoint(&ip);
            mfem::CalcOrtho(Tr->Face->Jacobian(), normal);
            const double ds = normal.Norml2();
            const double nx = normal(0) / ds;
            const double ny = normal(1) / ds;

            u_gf.GetCurl(*Tr->Elem1, curl);  // size 1 in 2D
            const double omega = curl(0);
            const double pval =
                p_gf.GetValue(*Tr->Elem1, Tr->GetElement1IntPoint());

            const double w = ip.weight * ds;  // arc-length measure
            // Force on the body F = oint(-p n + nu*omega (zhat x n)) with the
            // body-outward normal; with CalcOrtho's fluid-outward normal n and
            // p_phys = eps*pval this is eps*(omega*n_y + p*n_x) for drag,
            // eps*(p*n_y - omega*n_x) for lift (sign=-1 below absorbs the
            // body/fluid normal flip).
            FD += w * eps * (-omega * ny - pval * nx);
            FL += w * eps * (omega * nx - pval * ny);
        }
    }

    CylinderForces f;
    f.FD = sign * FD;
    f.FL = sign * FL;
    const double scale = 2.0 / (Ubar * Ubar * D);
    f.cD = scale * f.FD;
    f.cL = scale * f.FL;
    return f;
}

// Evaluate the (scalar) pressure GridFunction at a physical point via BFS
// element search.  Returns 0 if the point is not found.
inline double PressureAtPoint(mfem::GridFunction& p_gf, const mfem::Vector& pt)
{
    mfem::Mesh*           mesh = p_gf.FESpace()->GetMesh();
    mfem::IntegrationPoint ip;
    int elem = FindElementBFS(*mesh, 0, pt, ip);
    if (elem < 0)
        return 0.0;
    mfem::IsoparametricTransformation T;
    mesh->GetElementTransformation(elem, &T);
    T.SetIntPoint(&ip);
    return p_gf.GetValue(T, ip);
}

// Delta p = p(front) - p(back), front/back default to the Schaefer-Turek
// cylinder probe points (0.15,0.2) and (0.25,0.2).
inline double CylinderPressureDrop(
    mfem::GridFunction& p_gf,
    double front_x = 0.15, double front_y = 0.2,
    double back_x  = 0.25, double back_y  = 0.2)
{
    const int    dim = p_gf.FESpace()->GetMesh()->Dimension();
    mfem::Vector pf(dim), pb(dim);
    pf = 0.0;
    pb = 0.0;
    pf(0) = front_x; pf(1) = front_y;
    pb(0) = back_x;  pb(1) = back_y;
    return PressureAtPoint(p_gf, pf) - PressureAtPoint(p_gf, pb);
}

#endif  // CYLINDER_QOI_H
