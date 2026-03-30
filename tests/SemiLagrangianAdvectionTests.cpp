#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "SemiLagrangianAdvection.h"
#include "mfem.hpp"

using namespace mfem;

namespace
{
constexpr int kAttrLeft = 1;
constexpr int kAttrRight = 2;
constexpr int kAttrBottom = 3;
constexpr int kAttrTop = 4;
constexpr int kAttrZMin = 5;
constexpr int kAttrZMax = 6;

Mesh MakeAttributedUnitSquareMesh(int nx, int ny)
{
    Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::TRIANGLE,
                                      true, 1.0, 1.0, false);

    const double tol = 1e-12;
    for (int be = 0; be < mesh.GetNBE(); ++be)
    {
        Array<int> verts;
        mesh.GetBdrElementVertices(be, verts);

        double cx = 0.0;
        double cy = 0.0;
        for (int i = 0; i < verts.Size(); ++i)
        {
            const double *v = mesh.GetVertex(verts[i]);
            cx += v[0];
            cy += v[1];
        }
        cx /= static_cast<double>(verts.Size());
        cy /= static_cast<double>(verts.Size());

        int attr = 0;
        if (std::abs(cx) < tol) { attr = kAttrLeft; }
        else if (std::abs(cx - 1.0) < tol) { attr = kAttrRight; }
        else if (std::abs(cy) < tol) { attr = kAttrBottom; }
        else if (std::abs(cy - 1.0) < tol) { attr = kAttrTop; }

        MFEM_VERIFY(attr != 0, "Could not classify boundary element.");
        mesh.SetBdrAttribute(be, attr);
    }

    mesh.SetAttributes(false, true);
    return mesh;
}

Mesh MakeAttributedUnitCubeMesh(int nx, int ny, int nz)
{
    Mesh mesh = Mesh::MakeCartesian3D(nx, ny, nz, Element::TETRAHEDRON,
                                      1.0, 1.0, 1.0, false);

    const double tol = 1e-12;
    for (int be = 0; be < mesh.GetNBE(); ++be)
    {
        Array<int> verts;
        mesh.GetBdrElementVertices(be, verts);

        double cx = 0.0;
        double cy = 0.0;
        double cz = 0.0;
        for (int i = 0; i < verts.Size(); ++i)
        {
            const double *v = mesh.GetVertex(verts[i]);
            cx += v[0];
            cy += v[1];
            cz += v[2];
        }
        cx /= static_cast<double>(verts.Size());
        cy /= static_cast<double>(verts.Size());
        cz /= static_cast<double>(verts.Size());

        int attr = 0;
        if (std::abs(cx) < tol) { attr = kAttrLeft; }
        else if (std::abs(cx - 1.0) < tol) { attr = kAttrRight; }
        else if (std::abs(cy) < tol) { attr = kAttrBottom; }
        else if (std::abs(cy - 1.0) < tol) { attr = kAttrTop; }
        else if (std::abs(cz) < tol) { attr = kAttrZMin; }
        else if (std::abs(cz - 1.0) < tol) { attr = kAttrZMax; }

        MFEM_VERIFY(attr != 0, "Could not classify 3D boundary element.");
        mesh.SetBdrAttribute(be, attr);
    }

    mesh.SetAttributes(false, true);
    return mesh;
}

SemiLagrangianAdvection1Form<1>::VelocityFunc
ConstantVelocity(double vx, double vy)
{
    return [vx, vy](const Vector &, double, int, Vector &v)
    {
        v.SetSize(2);
        v[0] = vx;
        v[1] = vy;
    };
}

void ProjectAffineField(GridFunction &omega)
{
    VectorFunctionCoefficient coeff(
        2,
        [](const Vector &x, double, Vector &v)
        {
            v.SetSize(2);
            v[0] = 1.0 + 0.35 * x[0] - 0.15 * x[1];
            v[1] = -0.2 + 0.10 * x[0] + 0.45 * x[1];
        });
    coeff.SetTime(0.0);
    omega.ProjectCoefficient(coeff);
}
}

TEST(SemiLagrangianAdvection, ZeroVelocityIsIdentity)
{
    Mesh mesh = MakeAttributedUnitSquareMesh(4, 4);
    ND_FECollection fec(1, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction omega_old(&nd);
    GridFunction omega_new(&nd);
    ProjectAffineField(omega_old);
    omega_new = 0.0;

    SemiLagrangianAdvection1Form<1> advection(nd);

    int boundary_calls = 0;
    SemiLagrangianAdvection1Form<1>::BoundaryFunc boundary =
        [&boundary_calls](const Vector &, double, int, Vector &v)
    {
        boundary_calls++;
        v.SetSize(2);
        v = 0.0;
    };

    advection.Apply(ConstantVelocity(0.0, 0.0), boundary,
                    0.4, 0.2, omega_old, omega_new, 1);

    double max_abs_diff = 0.0;
    for (int i = 0; i < nd.GetNDofs(); ++i)
    {
        max_abs_diff = std::max(max_abs_diff, std::abs(omega_new(i) - omega_old(i)));
    }

    EXPECT_EQ(boundary_calls, 0);
    EXPECT_LT(max_abs_diff, 1e-12);
}

TEST(SemiLagrangianAdvection, SettlsMatchesEulerForSteadyVelocity)
{
    Mesh mesh = MakeAttributedUnitSquareMesh(4, 4);
    ND_FECollection fec(1, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction omega_old(&nd);
    GridFunction omega_euler(&nd);
    GridFunction omega_settls(&nd);
    ProjectAffineField(omega_old);
    omega_euler = 0.0;
    omega_settls = 0.0;

    SemiLagrangianAdvection1Form<1> advection(nd);

    SemiLagrangianAdvection1Form<1>::BoundaryFunc boundary =
        [](const Vector &, double, int, Vector &v)
    {
        v.SetSize(2);
        v = 0.0;
    };

    auto velocity_n = ConstantVelocity(0.25, -0.15);
    SemiLagrangianAdvection1Form<1>::VelocityFunc velocity_nm1 =
        ConstantVelocity(0.25, -0.15);

    advection.Apply(velocity_n, boundary,
                    0.3, 0.1, omega_old, omega_euler, 1);

    advection.Apply(velocity_n, boundary,
                    0.3, 0.1, omega_old, omega_settls,
                    3, nullptr, &velocity_nm1, 2);

    double max_abs_diff = 0.0;
    for (int i = 0; i < nd.GetNDofs(); ++i)
    {
        max_abs_diff = std::max(max_abs_diff,
                                std::abs(omega_euler(i) - omega_settls(i)));
    }

    EXPECT_LT(max_abs_diff, 1e-12);
}

TEST(SemiLagrangianAdvection, FullOutsideTransportUsesCorrectBoundaryAttributePerSide)
{
    struct Case
    {
        const char *name;
        double vx;
        double vy;
        int expected_attr;
    };

    const Case cases[] = {
        {"top", 0.0, -1.0, kAttrTop},
        {"bottom", 0.0, 1.0, kAttrBottom},
        {"left", 1.0, 0.0, kAttrLeft},
        {"right", -1.0, 0.0, kAttrRight},
    };

    for (const Case &c : cases)
    {
        Mesh mesh = MakeAttributedUnitSquareMesh(4, 4);
        ND_FECollection fec(1, mesh.Dimension());
        FiniteElementSpace nd(&mesh, &fec);

        GridFunction omega_old(&nd);
        GridFunction omega_new(&nd);
        omega_old = 0.0;
        omega_new = 0.0;

        SemiLagrangianAdvection1Form<1> advection(nd);

        std::vector<int> attrs;
        SemiLagrangianAdvection1Form<1>::BoundaryFunc boundary =
            [&attrs](const Vector &, double, int bdr_attr, Vector &v)
        {
            attrs.push_back(bdr_attr);
            v.SetSize(2);
            v = 0.0;
            v[0] = static_cast<double>(bdr_attr);
        };

        advection.Apply(ConstantVelocity(c.vx, c.vy), boundary,
                        1.25, 1.25, omega_old, omega_new, 1);

        ASSERT_FALSE(attrs.empty()) << "No boundary quadrature calls for " << c.name;
        for (int attr : attrs)
        {
            EXPECT_EQ(attr, c.expected_attr)
                << "Unexpected boundary attribute on " << c.name;
        }
    }
}

TEST(SemiLagrangianAdvection, PartialOutsideTransportAvoidsUnknownBoundaryAttribute)
{
    Mesh mesh = MakeAttributedUnitSquareMesh(4, 4);
    ND_FECollection fec(1, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction omega_old(&nd);
    GridFunction omega_new(&nd);
    omega_old = 0.0;
    omega_new = 0.0;

    SemiLagrangianAdvection1Form<1> advection(nd);

    std::vector<int> attrs;
    SemiLagrangianAdvection1Form<1>::BoundaryFunc boundary =
        [&attrs](const Vector &, double, int bdr_attr, Vector &v)
    {
        attrs.push_back(bdr_attr);
        v.SetSize(2);
        v = 0.0;
        v[0] = static_cast<double>(bdr_attr);
    };

    advection.Apply(ConstantVelocity(0.0, -1.0), boundary,
                    0.35, 0.35, omega_old, omega_new, 1);

    ASSERT_FALSE(attrs.empty());
    for (int attr : attrs)
    {
        EXPECT_NE(attr, 0) << "Found unknown boundary attribute in partial-outside case";
        EXPECT_EQ(attr, kAttrTop)
            << "Expected top boundary attribute in upward transport case";
    }
}

TEST(SemiLagrangianAdvection, FullOutsideTopEdgeDOFMatchesAnalyticBoundaryIntegral)
{
    Mesh mesh = MakeAttributedUnitSquareMesh(4, 4);
    ND_FECollection fec(1, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction omega_old(&nd);
    GridFunction omega_new(&nd);
    omega_old = 0.0;
    omega_new = 0.0;

    SemiLagrangianAdvection1Form<1> advection(nd);

    SemiLagrangianAdvection1Form<1>::BoundaryFunc boundary =
        [](const Vector &, double, int bdr_attr, Vector &v)
    {
        v.SetSize(2);
        v = 0.0;
        v[0] = static_cast<double>(bdr_attr);
    };

    const double vx = 0.0;
    const double vy = -1.0;
    const double dt = 1.25;
    advection.Apply(ConstantVelocity(vx, vy), boundary,
                    dt, dt, omega_old, omega_new, 1);

    int chosen_edge = -1;
    for (int be = 0; be < mesh.GetNBE(); ++be)
    {
        if (mesh.GetBdrAttribute(be) != kAttrTop) { continue; }

        const int edge = mesh.GetBdrElementFaceIndex(be);
        Array<int> verts;
        mesh.GetEdgeVertices(edge, verts);

        const double *v0 = mesh.GetVertex(verts[0]);
        const double *v1 = mesh.GetVertex(verts[1]);
        const double xmid = 0.5 * (v0[0] + v1[0]);
        if (xmid > 0.2 && xmid < 0.8)
        {
            chosen_edge = edge;
            break;
        }
    }

    ASSERT_GE(chosen_edge, 0);

    Array<int> edge_verts;
    mesh.GetEdgeVertices(chosen_edge, edge_verts);
    const double *v0 = mesh.GetVertex(edge_verts[0]);
    const double *v1 = mesh.GetVertex(edge_verts[1]);

    const double d0x = v0[0] - dt * vx;
    const double d1x = v1[0] - dt * vx;
    const double expected = static_cast<double>(kAttrTop) * (d1x - d0x);

    Array<int> dofs;
    nd.GetEdgeDofs(chosen_edge, dofs);
    ASSERT_EQ(dofs.Size(), 1);

    const double actual = omega_new(dofs[0]);
    EXPECT_NEAR(actual, expected, 1e-12);
}

TEST(SemiLagrangianAdvection, FullOutsideTransportIn3DUsesTopBoundaryAttribute)
{
    Mesh mesh = MakeAttributedUnitCubeMesh(2, 2, 2);
    ND_FECollection fec(1, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction omega_old(&nd);
    GridFunction omega_new(&nd);
    omega_old = 0.0;
    omega_new = 0.0;

    SemiLagrangianAdvection1Form<1> advection(nd);

    std::vector<int> attrs;
    SemiLagrangianAdvection1Form<1>::BoundaryFunc boundary =
        [&attrs](const Vector &, double, int bdr_attr, Vector &v)
    {
        attrs.push_back(bdr_attr);
        v.SetSize(3);
        v = 0.0;
        v[0] = static_cast<double>(bdr_attr);
    };

    SemiLagrangianAdvection1Form<1>::VelocityFunc velocity =
        [](const Vector &, double, int, Vector &v)
    {
        v.SetSize(3);
        v[0] = 0.0;
        v[1] = 0.0;
        v[2] = -1.0;
    };

    advection.Apply(velocity, boundary, 1.2, 1.2, omega_old, omega_new, 1);

    ASSERT_FALSE(attrs.empty());
    for (int attr : attrs)
    {
        EXPECT_EQ(attr, kAttrZMax);
    }
}

// --------------------------------------------------------------------------
// SplitLineIntoSegments degenerate-case tests
// --------------------------------------------------------------------------

namespace
{
/// Helper: compute total coverage of segments over [0,1].
double SegmentCoverage(const std::vector<LineSegment> &segments)
{
    double cov = 0.0;
    for (const auto &seg : segments) { cov += seg.s_end - seg.s_start; }
    return cov;
}

/// Helper: build a simple Cartesian triangle mesh and initialize its tables.
Mesh MakeSimpleTriMesh(int nx, int ny)
{
    Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::TRIANGLE,
                                      true, 1.0, 1.0, false);
    mesh.ElementToElementTable();
    mesh.GetEdgeVertexTable();
    return mesh;
}
} // anonymous namespace

TEST(SplitLineIntoSegments, LineThroughVertex)
{
    // Line through interior vertex (0.25,0.25) on a 4x4 triangle mesh.
    // Starts/ends inside the domain to avoid boundary perturbation issues.
    Mesh mesh = MakeSimpleTriMesh(4, 4);

    Vector pos1(2), pos2(2);
    pos1[0] = 0.01; pos1[1] = 0.01;
    pos2[0] = 0.49; pos2[1] = 0.49;

    std::vector<LineSegment> segments;
    Array<int> verts;
    bool ok = SplitLineIntoSegmentsRobust(
        mesh, 0, pos1, pos2, segments, nullptr, verts);

    EXPECT_TRUE(ok);
    ASSERT_FALSE(segments.empty());
    EXPECT_NEAR(SegmentCoverage(segments), 1.0, 1e-6);
}

TEST(SplitLineIntoSegments, LineAlongEdge)
{
    // Line along x=0.25, slightly inside the domain to avoid boundary issues.
    // Still lies exactly on a vertical mesh edge (degenerate to a face).
    Mesh mesh = MakeSimpleTriMesh(4, 4);

    Vector pos1(2), pos2(2);
    pos1[0] = 0.25; pos1[1] = 0.01;
    pos2[0] = 0.25; pos2[1] = 0.99;

    std::vector<LineSegment> segments;
    Array<int> verts;
    bool ok = SplitLineIntoSegmentsRobust(
        mesh, 0, pos1, pos2, segments, nullptr, verts);

    EXPECT_TRUE(ok);
    ASSERT_FALSE(segments.empty());
    EXPECT_NEAR(SegmentCoverage(segments), 1.0, 1e-6);
}

TEST(SplitLineIntoSegments, LineThroughMultipleVertices)
{
    // Near-diagonal line slightly inside the domain, still passes through
    // interior vertices at (0.25,0.25), (0.5,0.5), (0.75,0.75).
    Mesh mesh = MakeSimpleTriMesh(4, 4);

    Vector pos1(2), pos2(2);
    pos1[0] = 0.01; pos1[1] = 0.01;
    pos2[0] = 0.99; pos2[1] = 0.99;

    std::vector<LineSegment> segments;
    Array<int> verts;
    bool ok = SplitLineIntoSegmentsRobust(
        mesh, 0, pos1, pos2, segments, nullptr, verts);

    EXPECT_TRUE(ok);
    ASSERT_FALSE(segments.empty());
    EXPECT_NEAR(SegmentCoverage(segments), 1.0, 1e-6);
}

TEST(SplitLineIntoSegments, GaussKronrodMatchesCellSplit)
{
    // For a non-degenerate line, verify Gauss-Kronrod produces the same
    // result as cell-split integration (within tolerance).
    Mesh mesh = MakeSimpleTriMesh(4, 4);
    ND_FECollection fec(1, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction omega(&nd);
    ProjectAffineField(omega);

    // Non-degenerate line: avoids vertices
    Vector pos1(2), pos2(2);
    pos1[0] = 0.1; pos1[1] = 0.15;
    pos2[0] = 0.7; pos2[1] = 0.85;

    auto segments = SplitLineIntoSegments(mesh, 0, pos1, pos2);
    ASSERT_FALSE(segments.empty());

    GaussLegendreRule<1> rule;
    CellSplitWorkspace ws;
    double cs_result = IntegrateLineTangentialCellSplit(
        mesh, omega, pos1, pos2, segments, rule, ws);

    double gk_error = 0.0;
    double gk_result = IntegrateLineTangentialGaussKronrod(
        mesh, omega, 0, pos1, pos2, 1e-8, 15, &gk_error);

    EXPECT_NEAR(cs_result, gk_result, 1e-5)
        << "Gauss-Kronrod and cell-split results diverge";
}

TEST(SplitLineIntoSegments, IntegralAccuracyAfterPerturbation)
{
    // Line through interior vertex (0.25,0.25): integral of affine field
    // computed via the robust (perturbed) cell-split path should match
    // the Gauss-Kronrod reference to high accuracy.
    Mesh mesh = MakeSimpleTriMesh(4, 4);
    ND_FECollection fec(1, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction omega(&nd);
    ProjectAffineField(omega);

    // Interior line through vertex (0.25,0.25)
    Vector pos1(2), pos2(2);
    pos1[0] = 0.01; pos1[1] = 0.01;
    pos2[0] = 0.49; pos2[1] = 0.49;

    // Gauss-Kronrod reference (high accuracy, no cell-splitting)
    double gk_error = 0.0;
    double reference = IntegrateLineTangentialGaussKronrod(
        mesh, omega, 0, pos1, pos2, 1e-10, 15, &gk_error);

    // Use robust path (will trigger perturbation since line goes through vertex)
    std::vector<LineSegment> segments;
    Array<int> verts;
    bool ok = SplitLineIntoSegmentsRobust(
        mesh, 0, pos1, pos2, segments, nullptr, verts);
    ASSERT_TRUE(ok);

    GaussLegendreRule<1> rule;
    CellSplitWorkspace ws;
    double result = IntegrateLineTangentialCellSplit(
        mesh, omega, pos1, pos2, segments, rule, ws);

    // Perturbation is O(1e-9), so error should be negligible
    EXPECT_NEAR(result, reference, 1e-6)
        << "Integral after perturbation deviates from Gauss-Kronrod reference";
}
