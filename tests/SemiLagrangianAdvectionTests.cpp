#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "SemiLagrangianAdvection.h"
#include "SemiLagrangianAdvectionOrder2.h"
#include "SmallEdgeEnumeration.h"
#include "RapettiProjection.h"
#include "VelocitySmoothing.h"
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

// ==========================================================================
//  Second-order (Rapetti / small-edge) tests
// ==========================================================================

namespace
{
SemiLagrangianAdvection1FormOrder2<2>::VelocityFunc
ConstantVelocityND2(double vx, double vy)
{
    return [vx, vy](const Vector &, double, int, Vector &v)
    {
        v.SetSize(2);
        v[0] = vx;
        v[1] = vy;
    };
}

SemiLagrangianAdvection1FormOrder2<2>::VelocityFunc
ConstantVelocityND2_3D(double vx, double vy, double vz)
{
    return [vx, vy, vz](const Vector &, double, int, Vector &v)
    {
        v.SetSize(3);
        v[0] = vx;
        v[1] = vy;
        v[2] = vz;
    };
}

SemiLagrangianAdvection1FormOrder2<2>::BoundaryFunc
ZeroBoundaryND2(int dim)
{
    return [dim](const Vector &, double, int, Vector &v)
    {
        v.SetSize(dim);
        v = 0.0;
    };
}
} // anonymous namespace

// --------------------------------------------------------------------------
// SmallEdgeTopology tests
// --------------------------------------------------------------------------

TEST(SmallEdgeTopology, CountsCorrect2D)
{
    Mesh mesh = MakeAttributedUnitSquareMesh(2, 2);
    SmallEdgeTopology topo(mesh);

    EXPECT_EQ(topo.NumSmallEdgesPerElement(), 9);
    EXPECT_EQ(topo.NumBigEdgesPerElement(), 3);
    EXPECT_EQ(topo.NumVerticesPerElement(), 3);
}

TEST(SmallEdgeTopology, CountsCorrect3D)
{
    Mesh mesh = MakeAttributedUnitCubeMesh(2, 2, 2);
    SmallEdgeTopology topo(mesh);

    EXPECT_EQ(topo.NumSmallEdgesPerElement(), 24);
    EXPECT_EQ(topo.NumBigEdgesPerElement(), 6);
    EXPECT_EQ(topo.NumVerticesPerElement(), 4);
}

TEST(SmallEdgeTopology, EndpointsAreMidpoints2D)
{
    // Verify that small edge endpoints are midpoints of vertex + edge endpoint
    Mesh mesh = MakeAttributedUnitSquareMesh(2, 2);
    SmallEdgeTopology topo(mesh);

    for (int el = 0; el < mesh.GetNE(); ++el)
    {
        Array<int> verts;
        mesh.GetElementVertices(el, verts);

        for (int k = 0; k < 9; ++k)
        {
            const LocalSmallEdge &se = topo.GetLocalSmallEdge(k);
            Vector a(2), b(2);
            topo.GetSmallEdgeEndpoints(el, k, a, b);

            const double *vi = mesh.GetVertex(verts[se.local_vertex]);
            const double *va = mesh.GetVertex(verts[se.edge_v0]);
            const double *vb = mesh.GetVertex(verts[se.edge_v1]);

            for (int d = 0; d < 2; ++d)
            {
                EXPECT_NEAR(a[d], 0.5 * (vi[d] + va[d]), 1e-14)
                    << "elem=" << el << " k=" << k << " d=" << d;
                EXPECT_NEAR(b[d], 0.5 * (vi[d] + vb[d]), 1e-14)
                    << "elem=" << el << " k=" << k << " d=" << d;
            }
        }
    }
}

TEST(SmallEdgeTopology, EdgeLocalSmallEdgesCount)
{
    // Each big edge has exactly 2 edge-local small edges
    Mesh mesh = MakeAttributedUnitSquareMesh(2, 2);
    SmallEdgeTopology topo(mesh);

    for (int e = 0; e < topo.NumBigEdgesPerElement(); ++e)
    {
        int idx[2];
        topo.GetEdgeLocalSmallEdges(e, idx);
        EXPECT_GE(idx[0], 0);
        EXPECT_GE(idx[1], 0);
        EXPECT_LT(idx[0], 9);
        EXPECT_LT(idx[1], 9);
        EXPECT_NE(idx[0], idx[1]);

        // Verify: the vertex of each edge-local small edge IS an endpoint
        // of the big edge
        const LocalSmallEdge &se0 = topo.GetLocalSmallEdge(idx[0]);
        const LocalSmallEdge &se1 = topo.GetLocalSmallEdge(idx[1]);
        EXPECT_EQ(se0.local_big_edge, e);
        EXPECT_EQ(se1.local_big_edge, e);
        EXPECT_TRUE(se0.local_vertex == se0.edge_v0 ||
                    se0.local_vertex == se0.edge_v1);
        EXPECT_TRUE(se1.local_vertex == se1.edge_v0 ||
                    se1.local_vertex == se1.edge_v1);
    }
}

TEST(SmallEdgeTopology, FaceInteriorCount2D)
{
    // In 2D, one face (the element itself) with 3 face-interior small edges
    Mesh mesh = MakeAttributedUnitSquareMesh(2, 2);
    SmallEdgeTopology topo(mesh);

    EXPECT_EQ(topo.NumFaces(), 1);
    EXPECT_EQ(topo.NumFaceInteriorSmallEdges(0), 3);

    // Verify: face-interior small edges have vertex NOT on the big edge
    for (int ki : topo.GetFaceInteriorSmallEdges(0))
    {
        const LocalSmallEdge &se = topo.GetLocalSmallEdge(ki);
        EXPECT_NE(se.local_vertex, se.edge_v0);
        EXPECT_NE(se.local_vertex, se.edge_v1);
    }
}

TEST(SmallEdgeTopology, FaceInteriorCount3D)
{
    // In 3D, 4 faces, each with 3 face-interior small edges
    Mesh mesh = MakeAttributedUnitCubeMesh(2, 2, 2);
    SmallEdgeTopology topo(mesh);

    EXPECT_EQ(topo.NumFaces(), 4);
    for (int f = 0; f < 4; ++f)
    {
        EXPECT_EQ(topo.NumFaceInteriorSmallEdges(f), 3);
    }
}

TEST(SmallEdgeTopology, SmallEdgeDirectionIsHalfBigEdge)
{
    // Direction of small edge {v_i, e_j} should be 0.5*(v_b - v_a),
    // independent of v_i.
    Mesh mesh = MakeAttributedUnitSquareMesh(2, 2);
    SmallEdgeTopology topo(mesh);

    for (int el = 0; el < std::min(mesh.GetNE(), 4); ++el)
    {
        Array<int> verts;
        mesh.GetElementVertices(el, verts);

        for (int k = 0; k < 9; ++k)
        {
            const LocalSmallEdge &se = topo.GetLocalSmallEdge(k);
            Vector a(2), b(2);
            topo.GetSmallEdgeEndpoints(el, k, a, b);

            const double *va = mesh.GetVertex(verts[se.edge_v0]);
            const double *vb = mesh.GetVertex(verts[se.edge_v1]);

            for (int d = 0; d < 2; ++d)
            {
                double expected_dir = 0.5 * (vb[d] - va[d]);
                double actual_dir = b[d] - a[d];
                EXPECT_NEAR(actual_dir, expected_dir, 1e-14)
                    << "el=" << el << " k=" << k << " d=" << d;
            }
        }
    }
}

// --------------------------------------------------------------------------
// RapettiProjector tests
// --------------------------------------------------------------------------

TEST(RapettiProjector, MMatrixDimensions2D)
{
    // M_{ij} = ∫_{s_i} w^{s_j} is 9×9 in 2D (NOT symmetric: different
    // integration domains vs. integrands).
    RapettiProjector proj(2);
    const auto &M = proj.GetMMatrix();
    EXPECT_EQ(M.Height(), 9);
    EXPECT_EQ(M.Width(), 9);

    // Diagonal entries should be non-zero (shape function s_k integrated
    // over its own small edge should give a positive value for matching
    // edge-local entries).
    bool has_nonzero_diag = false;
    for (int i = 0; i < 9; ++i)
    {
        if (std::abs(M(i, i)) > 1e-14)
        {
            has_nonzero_diag = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero_diag) << "M should have some nonzero diagonal entries";
}

TEST(RapettiProjector, MMatrixDimensions3D)
{
    RapettiProjector proj(3);
    const auto &M = proj.GetMMatrix();
    EXPECT_EQ(M.Height(), 24);
    EXPECT_EQ(M.Width(), 24);

    bool has_nonzero_diag = false;
    for (int i = 0; i < 24; ++i)
    {
        if (std::abs(M(i, i)) > 1e-14)
        {
            has_nonzero_diag = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero_diag) << "M should have some nonzero diagonal entries";
}

TEST(RapettiProjector, ProjectConstant1Form2D)
{
    // A constant 1-form ω = (1, 0) on the reference triangle.
    // Small edge integral of ω over small edge s_k is:
    //   ∫_{s_k} ω · dl = direction_k · (1, 0) = (b_k - a_k) · (1, 0)
    // After I_{h,2} projection and RapettiToND2 conversion, the result
    // should reproduce the constant field.
    RapettiProjector proj(2);
    RapettiShapeFunctions shapes(2);
    const int n_small = 9;

    // Reference vertices: v0=(0,0), v1=(1,0), v2=(0,1)
    // MFEM edges: e0=(0,1), e1=(1,2), e2=(2,0)
    double rv[3][2] = {{0, 0}, {1, 0}, {0, 1}};
    static const int ev[3][2] = {{0, 1}, {1, 2}, {2, 0}};

    // Compute raw integrals of constant 1-form (1, 0) over small edges
    std::vector<double> raw(n_small);
    for (int k = 0; k < n_small; ++k)
    {
        int vi = k / 3;
        int ej = k % 3;
        int ea = ev[ej][0], eb = ev[ej][1];

        double ax = 0.5 * (rv[vi][0] + rv[ea][0]);
        double bx = 0.5 * (rv[vi][0] + rv[eb][0]);
        double ay = 0.5 * (rv[vi][1] + rv[ea][1]);
        double by = 0.5 * (rv[vi][1] + rv[eb][1]);

        // ∫ (1,0) · (b-a) = bx - ax
        raw[k] = bx - ax;
    }

    std::vector<double> rapetti_dofs(n_small);
    proj.ProjectElement(raw.data(), rapetti_dofs.data());

    // Verify: reconstruct the 1-form at test points using Rapetti DOFs
    // and check it equals (1, 0).
    Vector pt(2), val(2);
    double test_pts[][2] = {{0.25, 0.25}, {0.1, 0.1}, {0.5, 0.1}, {0.1, 0.5}};
    for (auto &tp : test_pts)
    {
        pt[0] = tp[0]; pt[1] = tp[1];
        // Reconstruct: ω_h(x) = Σ_k rapetti_dofs[k] * w^{s_k}(x)
        Vector reconstructed(2);
        reconstructed = 0.0;
        for (int k = 0; k < n_small; ++k)
        {
            shapes.Eval(k, pt, val);
            for (int d = 0; d < 2; ++d)
            {
                reconstructed[d] += rapetti_dofs[k] * val[d];
            }
        }
        EXPECT_NEAR(reconstructed[0], 1.0, 1e-10)
            << "Constant (1,0) not reproduced at ("
            << tp[0] << "," << tp[1] << ")";
        EXPECT_NEAR(reconstructed[1], 0.0, 1e-10)
            << "Constant (1,0) not reproduced at ("
            << tp[0] << "," << tp[1] << ")";
    }
}

TEST(RapettiProjector, ProjectConstant1Form3D)
{
    // Constant 1-form (1, 0, 0) on reference tetrahedron
    RapettiProjector proj(3);
    RapettiShapeFunctions shapes(3);
    const int n_small = 24;

    double rv[4][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    static const int ev[6][2] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
    };

    std::vector<double> raw(n_small);
    for (int k = 0; k < n_small; ++k)
    {
        int vi = k / 6;
        int ej = k % 6;
        int ea = ev[ej][0], eb = ev[ej][1];

        double bx = 0.5 * (rv[vi][0] + rv[eb][0]);
        double ax = 0.5 * (rv[vi][0] + rv[ea][0]);
        raw[k] = bx - ax;  // ∫ (1,0,0) · dl = dx component of direction
    }

    std::vector<double> rapetti_dofs(n_small);
    proj.ProjectElement(raw.data(), rapetti_dofs.data());

    // Check at interior test points
    Vector pt(3), val(3);
    double test_pts[][3] = {{0.1, 0.1, 0.1}, {0.25, 0.25, 0.1}, {0.1, 0.2, 0.3}};
    for (auto &tp : test_pts)
    {
        pt[0] = tp[0]; pt[1] = tp[1]; pt[2] = tp[2];
        Vector reconstructed(3);
        reconstructed = 0.0;
        for (int k = 0; k < n_small; ++k)
        {
            shapes.Eval(k, pt, val);
            for (int d = 0; d < 3; ++d)
            {
                reconstructed[d] += rapetti_dofs[k] * val[d];
            }
        }
        EXPECT_NEAR(reconstructed[0], 1.0, 1e-10)
            << "3D constant not reproduced at ("
            << tp[0] << "," << tp[1] << "," << tp[2] << ")";
        EXPECT_NEAR(reconstructed[1], 0.0, 1e-10);
        EXPECT_NEAR(reconstructed[2], 0.0, 1e-10);
    }
}

// --------------------------------------------------------------------------
// RapettiToND2Converter tests
// --------------------------------------------------------------------------

TEST(RapettiToND2Converter, TMatrixDimensions)
{
    RapettiToND2Converter conv2d(2);
    EXPECT_EQ(conv2d.NumND2(), 8);
    EXPECT_EQ(conv2d.NumSmall(), 9);

    RapettiToND2Converter conv3d(3);
    EXPECT_EQ(conv3d.NumND2(), 20);
    EXPECT_EQ(conv3d.NumSmall(), 24);
}

TEST(RapettiToND2Converter, RoundtripConstant2D)
{
    // Convert constant field Rapetti → ND₂ → Rapetti, verify reconstruction
    RapettiProjector proj(2);
    RapettiToND2Converter conv(2);
    RapettiShapeFunctions shapes(2);
    const int n_small = 9;

    // Build raw integrals for constant (0, 1)
    double rv[3][2] = {{0, 0}, {1, 0}, {0, 1}};
    static const int ev[3][2] = {{0, 1}, {1, 2}, {2, 0}};

    std::vector<double> raw(n_small);
    for (int k = 0; k < n_small; ++k)
    {
        int vi = k / 3, ej = k % 3;
        int ea = ev[ej][0], eb = ev[ej][1];
        double by = 0.5 * (rv[vi][1] + rv[eb][1]);
        double ay = 0.5 * (rv[vi][1] + rv[ea][1]);
        raw[k] = by - ay;
    }

    std::vector<double> rapetti(n_small);
    proj.ProjectElement(raw.data(), rapetti.data());

    // Convert to ND₂
    std::vector<double> nd2(conv.NumND2());
    conv.Convert(rapetti.data(), nd2.data());

    // Convert back
    std::vector<double> rapetti_back(n_small);
    conv.ConvertInverse(nd2.data(), rapetti_back.data());

    // Reconstruct from rapetti_back and check at test point
    Vector pt(2), val(2);
    pt[0] = 0.25; pt[1] = 0.25;
    Vector reconstructed(2);
    reconstructed = 0.0;
    for (int k = 0; k < n_small; ++k)
    {
        shapes.Eval(k, pt, val);
        for (int d = 0; d < 2; ++d)
        {
            reconstructed[d] += rapetti_back[k] * val[d];
        }
    }
    EXPECT_NEAR(reconstructed[0], 0.0, 1e-8);
    EXPECT_NEAR(reconstructed[1], 1.0, 1e-8);
}

// --------------------------------------------------------------------------
// VelocitySmoothing tests
// --------------------------------------------------------------------------

TEST(VelocitySmoothing, ConstantFieldUnchanged2D)
{
    // Box average of a constant velocity field should be exactly that constant
    Mesh mesh = MakeAttributedUnitSquareMesh(4, 4);
    ND_FECollection fec(2, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction u(&nd);
    VectorFunctionCoefficient coef(2, [](const Vector &, Vector &v)
    {
        v.SetSize(2);
        v[0] = 3.0;
        v[1] = -2.0;
    });
    u.ProjectCoefficient(coef);

    SmoothedVelocityField smooth(mesh);
    smooth.Update(u);

    Vector pt(2), vel(2);
    // Test at several interior points
    double pts[][2] = {{0.3, 0.4}, {0.5, 0.5}, {0.7, 0.2}};
    for (auto &p : pts)
    {
        pt[0] = p[0]; pt[1] = p[1];
        smooth.Evaluate(pt, 0.0, 0, vel);
        EXPECT_NEAR(vel[0], 3.0, 1e-6)
            << "Constant vx not preserved at (" << p[0] << "," << p[1] << ")";
        EXPECT_NEAR(vel[1], -2.0, 1e-6)
            << "Constant vy not preserved at (" << p[0] << "," << p[1] << ")";
    }
}

TEST(VelocitySmoothing, ConstantFieldUnchanged3D)
{
    Mesh mesh = MakeAttributedUnitCubeMesh(2, 2, 2);
    ND_FECollection fec(2, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction u(&nd);
    VectorFunctionCoefficient coef(3, [](const Vector &, Vector &v)
    {
        v.SetSize(3);
        v[0] = 1.0;
        v[1] = -0.5;
        v[2] = 2.0;
    });
    u.ProjectCoefficient(coef);

    SmoothedVelocityField smooth(mesh);
    smooth.Update(u);

    Vector pt(3), vel(3);
    pt[0] = 0.4; pt[1] = 0.3; pt[2] = 0.5;
    smooth.Evaluate(pt, 0.0, 0, vel);
    EXPECT_NEAR(vel[0], 1.0, 1e-5);
    EXPECT_NEAR(vel[1], -0.5, 1e-5);
    EXPECT_NEAR(vel[2], 2.0, 1e-5);
}

// --------------------------------------------------------------------------
// Full order-2 advection tests
// --------------------------------------------------------------------------

TEST(SemiLagrangianAdvectionOrder2, ZeroVelocityIsIdentity2D)
{
    Mesh mesh = MakeAttributedUnitSquareMesh(4, 4);
    ND_FECollection fec(2, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction omega_old(&nd);
    GridFunction omega_new(&nd);

    // Project a smooth field: linear in x and y
    VectorFunctionCoefficient coef(2,
        [](const Vector &x, double, Vector &v)
    {
        v.SetSize(2);
        v[0] = 1.0 + 0.5 * x[0] - 0.3 * x[1];
        v[1] = -0.2 + 0.4 * x[0] + 0.6 * x[1];
    });
    coef.SetTime(0.0);
    omega_old.ProjectCoefficient(coef);
    omega_new = 0.0;

    SemiLagrangianAdvection1FormOrder2<2> advection(nd);

    advection.Apply(ConstantVelocityND2(0.0, 0.0),
                    ZeroBoundaryND2(2),
                    0.4, 0.2, omega_old, omega_new, 2);

    double max_diff = 0.0;
    for (int i = 0; i < nd.GetNDofs(); ++i)
    {
        max_diff = std::max(max_diff, std::abs(omega_new(i) - omega_old(i)));
    }

    EXPECT_LT(max_diff, 1e-8)
        << "Zero velocity should preserve the field exactly";
}

TEST(SemiLagrangianAdvectionOrder2, ZeroVelocityIsIdentity3D)
{
    Mesh mesh = MakeAttributedUnitCubeMesh(2, 2, 2);
    ND_FECollection fec(2, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction omega_old(&nd);
    GridFunction omega_new(&nd);

    VectorFunctionCoefficient coef(3,
        [](const Vector &x, double, Vector &v)
    {
        v.SetSize(3);
        v[0] = 1.0 + 0.3 * x[0];
        v[1] = -0.5 + 0.2 * x[1];
        v[2] = 0.1 * x[2];
    });
    coef.SetTime(0.0);
    omega_old.ProjectCoefficient(coef);
    omega_new = 0.0;

    SemiLagrangianAdvection1FormOrder2<2> advection(nd);

    advection.Apply(ConstantVelocityND2_3D(0.0, 0.0, 0.0),
                    ZeroBoundaryND2(3),
                    0.4, 0.2, omega_old, omega_new, 2);

    double max_diff = 0.0;
    for (int i = 0; i < nd.GetNDofs(); ++i)
    {
        max_diff = std::max(max_diff, std::abs(omega_new(i) - omega_old(i)));
    }

    EXPECT_LT(max_diff, 1e-8)
        << "Zero velocity should preserve the 3D field exactly";
}

TEST(SemiLagrangianAdvectionOrder2, NonZeroVelocity3DNoNaN)
{
    // Test that order-2 advection in 3D with non-zero velocity produces
    // finite (non-NaN) results.  Exercises the full pipeline:
    // small-edge enumeration, Heun tracing, cell-split line integrals,
    // Rapetti projection, and Rapetti->ND2 conversion.
    Mesh mesh = MakeAttributedUnitCubeMesh(3, 3, 3);
    ND_FECollection fec(2, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction omega_old(&nd);
    GridFunction omega_new(&nd);

    VectorFunctionCoefficient coef(3,
        [](const Vector &x, double, Vector &v)
    {
        v.SetSize(3);
        v[0] = std::sin(M_PI * x[0]) * std::cos(M_PI * x[1]);
        v[1] = -std::cos(M_PI * x[0]) * std::sin(M_PI * x[1]);
        v[2] = 0.0;
    });
    coef.SetTime(0.0);
    omega_old.ProjectCoefficient(coef);
    omega_new = 0.0;

    SemiLagrangianAdvection1FormOrder2<2> advection(nd);

    // Small constant velocity, small dt so points stay interior
    advection.Apply(ConstantVelocityND2_3D(0.01, 0.02, 0.0),
                    ZeroBoundaryND2(3),
                    0.1, 0.01, omega_old, omega_new, 2);

    // Check no NaN in output
    bool has_nan = false;
    for (int i = 0; i < omega_new.Size(); ++i)
    {
        if (!std::isfinite(omega_new(i)))
        {
            has_nan = true;
            break;
        }
    }
    EXPECT_FALSE(has_nan) << "3D order-2 advection with non-zero velocity produced NaN";

    // Check the result is non-trivial (not all zeros)
    double max_val = 0.0;
    for (int i = 0; i < omega_new.Size(); ++i)
    {
        max_val = std::max(max_val, std::abs(omega_new(i)));
    }
    EXPECT_GT(max_val, 1e-6) << "Result should be non-trivial";
}

TEST(SemiLagrangianAdvectionOrder2, ConstantFieldPreserved2D)
{
    // A constant 1-form field should be exactly preserved under advection
    // with any velocity, because constant fields are in the FE space and
    // the pullback of a constant 1-form is constant.
    Mesh mesh = MakeAttributedUnitSquareMesh(4, 4);
    ND_FECollection fec(2, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction omega_old(&nd);
    GridFunction omega_new(&nd);

    VectorFunctionCoefficient coef(2, [](const Vector &, Vector &v)
    {
        v.SetSize(2);
        v[0] = 1.0;
        v[1] = 0.0;
    });
    omega_old.ProjectCoefficient(coef);
    omega_new = 0.0;

    SemiLagrangianAdvection1FormOrder2<2> advection(nd);

    // Small constant velocity so almost everything stays interior.
    // Use a boundary function that returns the correct constant (1,0)
    // so that points leaving the domain still get the correct pullback.
    SemiLagrangianAdvection1FormOrder2<2>::BoundaryFunc const_bdr =
        [](const Vector &, double, int, Vector &v)
    {
        v.SetSize(2);
        v[0] = 1.0;
        v[1] = 0.0;
    };
    advection.Apply(ConstantVelocityND2(0.05, 0.03),
                    const_bdr,
                    0.1, 0.05, omega_old, omega_new, 2);

    // Measure L2 error
    BilinearForm mass(&nd);
    mass.AddDomainIntegrator(new VectorFEMassIntegrator());
    mass.Assemble();
    mass.Finalize();

    GridFunction diff(&nd);
    for (int i = 0; i < nd.GetNDofs(); ++i)
    {
        diff(i) = omega_new(i) - omega_old(i);
    }

    Vector Mdiff(nd.GetVSize());
    mass.SpMat().Mult(diff, Mdiff);
    double l2_err = std::sqrt(std::abs(diff * Mdiff));

    EXPECT_LT(l2_err, 1e-6)
        << "Constant 1-form (1,0) not preserved under small translation";
}

TEST(SemiLagrangianAdvectionOrder2, BDF2HistoryShift)
{
    // Verify BDF2 produces two distinct output fields (tilde_1, tilde_2)
    // that are different when given different history
    Mesh mesh = MakeAttributedUnitSquareMesh(4, 4);
    ND_FECollection fec(2, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    // Create two different history fields
    GridFunction omega_n1(&nd), omega_n2(&nd);
    VectorFunctionCoefficient coef1(2,
        [](const Vector &x, double, Vector &v)
    {
        v.SetSize(2);
        v[0] = std::sin(M_PI * x[0]) * std::cos(M_PI * x[1]);
        v[1] = -std::cos(M_PI * x[0]) * std::sin(M_PI * x[1]);
    });
    VectorFunctionCoefficient coef2(2,
        [](const Vector &x, double, Vector &v)
    {
        v.SetSize(2);
        v[0] = 0.9 * std::sin(M_PI * x[0]) * std::cos(M_PI * x[1]);
        v[1] = -0.9 * std::cos(M_PI * x[0]) * std::sin(M_PI * x[1]);
    });
    coef1.SetTime(0.0);
    coef2.SetTime(0.0);
    omega_n1.ProjectCoefficient(coef1);
    omega_n2.ProjectCoefficient(coef2);

    GridFunction tilde1(&nd), tilde2(&nd);

    SemiLagrangianAdvection1FormOrder2<2> advection(nd);

    advection.ApplyBDF2(ConstantVelocityND2(0.0, 0.0),
                        ZeroBoundaryND2(2),
                        0.2, 0.1,
                        omega_n1, omega_n2,
                        tilde1, tilde2, 2);

    // With zero velocity: tilde1 should equal omega_n1,
    // tilde2 should equal omega_n2
    double max_diff_1 = 0.0, max_diff_2 = 0.0;
    for (int i = 0; i < nd.GetNDofs(); ++i)
    {
        max_diff_1 = std::max(max_diff_1, std::abs(tilde1(i) - omega_n1(i)));
        max_diff_2 = std::max(max_diff_2, std::abs(tilde2(i) - omega_n2(i)));
    }

    EXPECT_LT(max_diff_1, 1e-8)
        << "BDF2 tilde_1 should match omega_n1 with zero velocity";
    EXPECT_LT(max_diff_2, 1e-8)
        << "BDF2 tilde_2 should match omega_n2 with zero velocity";
}

// --------------------------------------------------------------------------
// FindElementBFS edge case: point on mesh face
// --------------------------------------------------------------------------

TEST(FindElementBFS, PointOnEdgeMidpointIsFound2D)
{
    // Verify that a point at an interior edge midpoint can be located
    // (important for small edge endpoints)
    Mesh mesh = MakeAttributedUnitSquareMesh(4, 4);
    mesh.ElementToElementTable();
    mesh.GetEdgeVertexTable();

    // Pick an interior edge midpoint
    int found_count = 0;
    Array<int> verts;
    for (int e = 0; e < mesh.GetNEdges(); ++e)
    {
        mesh.GetEdgeVertices(e, verts);
        const double *v0 = mesh.GetVertex(verts[0]);
        const double *v1 = mesh.GetVertex(verts[1]);
        double mx = 0.5 * (v0[0] + v1[0]);
        double my = 0.5 * (v0[1] + v1[1]);

        // Skip boundary edges
        if (mx < 1e-10 || mx > 1.0 - 1e-10 ||
            my < 1e-10 || my > 1.0 - 1e-10) { continue; }

        Vector pt(2);
        pt[0] = mx;
        pt[1] = my;
        IntegrationPoint ip;
        int elem = FindElementBFS(mesh, 0, pt, ip);

        if (elem >= 0) { ++found_count; }
    }

    // Most interior edge midpoints should be found (they lie on shared
    // faces and one of the adjacent elements should accept them).
    EXPECT_GT(found_count, 0)
        << "FindElementBFS should locate at least some edge midpoints";
}

TEST(FindElementBFS, PointSlightlyOffFaceIsFound)
{
    // Points slightly perturbed from a face should always be found
    Mesh mesh = MakeAttributedUnitSquareMesh(4, 4);
    mesh.ElementToElementTable();
    mesh.GetEdgeVertexTable();

    Vector pt(2);
    IntegrationPoint ip;

    // Point in the interior, slightly off an edge
    pt[0] = 0.25 + 1e-10;
    pt[1] = 0.5;
    int elem = FindElementBFS(mesh, 0, pt, ip);
    EXPECT_GE(elem, 0) << "Perturbed-off-face point should be found";

    pt[0] = 0.25 - 1e-10;
    pt[1] = 0.5;
    elem = FindElementBFS(mesh, 0, pt, ip);
    EXPECT_GE(elem, 0) << "Other-side perturbed point should be found";
}

// --------------------------------------------------------------------------
// Convergence test: advect a linear field with constant velocity
// --------------------------------------------------------------------------

TEST(SemiLagrangianAdvectionOrder2, LinearFieldExactTranslation2D)
{
    // Advect a linear 1-form with constant velocity.
    // Exact pullback of a linear 1-form under translation is still linear,
    // which is in the ND₂ space. So the result should be exact.
    //
    // Use a finer mesh to avoid boundary effects.
    Mesh mesh = Mesh::MakeCartesian2D(8, 8, Element::TRIANGLE,
                                       true, 1.0, 1.0, false);
    // Set boundary attributes
    for (int be = 0; be < mesh.GetNBE(); ++be)
    {
        Array<int> v;
        mesh.GetBdrElementVertices(be, v);
        double cx = 0, cy = 0;
        for (int i = 0; i < v.Size(); ++i)
        {
            const double *p = mesh.GetVertex(v[i]);
            cx += p[0]; cy += p[1];
        }
        cx /= v.Size(); cy /= v.Size();
        int attr = 1;
        if (std::abs(cx) < 1e-12) attr = 1;
        else if (std::abs(cx - 1.0) < 1e-12) attr = 2;
        else if (std::abs(cy) < 1e-12) attr = 3;
        else if (std::abs(cy - 1.0) < 1e-12) attr = 4;
        mesh.SetBdrAttribute(be, attr);
    }
    mesh.SetAttributes(false, true);

    ND_FECollection fec(2, 2);
    FiniteElementSpace nd(&mesh, &fec);

    // Linear field: ω = (1 + 0.5x, -0.3y)
    auto linear_field = [](const Vector &x, double, Vector &v)
    {
        v.SetSize(2);
        v[0] = 1.0 + 0.5 * x[0];
        v[1] = -0.3 * x[1];
    };

    GridFunction omega_old(&nd), omega_new(&nd), omega_exact(&nd);

    VectorFunctionCoefficient coef_old(2, linear_field);
    coef_old.SetTime(0.0);
    omega_old.ProjectCoefficient(coef_old);

    // Advect with small constant velocity (so nothing exits the domain)
    const double vx = 0.02, vy = 0.01;
    const double dt = 0.05;

    SemiLagrangianAdvection1FormOrder2<2> advection(nd);
    // Use boundary function that matches the linear field (to avoid boundary
    // errors when departure points land slightly outside the domain).
    SemiLagrangianAdvection1FormOrder2<2>::BoundaryFunc linear_bdr =
        [&linear_field](const Vector &pt, double t, int, Vector &v)
    {
        linear_field(pt, t, v);
    };
    advection.Apply(ConstantVelocityND2(vx, vy),
                    linear_bdr,
                    dt, dt, omega_old, omega_new, 2);

    // Exact: pullback under translation (x - dt*v) of a 1-form is the same
    // 1-form evaluated at the departure point. For a closed 1-form ω = df,
    // the pullback is exact. For a general linear 1-form, the pullback under
    // translation is the original 1-form evaluated at (x - dt*v):
    //   (X*ω)(x) = ω(x - dt*v)
    // So at point x: ω_exact(x) = (1 + 0.5*(x[0] - dt*vx), -0.3*(x[1] - dt*vy))
    auto exact_pullback = [vx, vy, dt](const Vector &x, double, Vector &v)
    {
        v.SetSize(2);
        v[0] = 1.0 + 0.5 * (x[0] - dt * vx);
        v[1] = -0.3 * (x[1] - dt * vy);
    };
    VectorFunctionCoefficient coef_exact(2, exact_pullback);
    coef_exact.SetTime(0.0);
    omega_exact.ProjectCoefficient(coef_exact);

    // Compute L2 error
    BilinearForm mass(&nd);
    mass.AddDomainIntegrator(new VectorFEMassIntegrator());
    mass.Assemble();
    mass.Finalize();

    GridFunction diff(&nd);
    for (int i = 0; i < nd.GetVSize(); ++i)
    {
        diff(i) = omega_new(i) - omega_exact(i);
    }
    Vector Mdiff(nd.GetVSize());
    mass.SpMat().Mult(diff, Mdiff);
    double l2_err = std::sqrt(std::abs(diff * Mdiff));

    // For a linear field with constant velocity, the Heun tracer is exact,
    // and the I_{h,2} projection should be exact (linear is in the space).
    // Allow some tolerance for numerical errors.
    EXPECT_LT(l2_err, 1e-4)
        << "Linear field should be near-exactly transported by constant velocity";
}

// --------------------------------------------------------------------------
//  Second-order convergence: exact pullback reference
// --------------------------------------------------------------------------

TEST(SemiLagrangianAdvectionOrder2, TaylorGreenConvergence2D)
{
    // Verify O(h²) spatial convergence of the I_{h,2} operator.
    //
    // Uses rigid rotation velocity u = (-(y-½), x-½) so that Heun tracing
    // is exact to O(τ³) (linear velocity). The initial 1-form is Taylor-Green:
    //   ω₀ = (cos πx sin πy, -sin πx cos πy).
    //
    // The exact pullback has a closed form:
    //   (X*_τ ω₀)(x) = R(τ) · ω₀(dep(x))
    // where dep(x) = c + R(-τ)(x-c) and c = (½,½).
    //
    // The L² error ‖ω̃ - X*_τ ω₀‖ is measured against the exact continuous
    // pullback via ComputeL2Error. Rates should be ≈ 2.

    const double cx = 0.5, cy = 0.5;

    // Rigid rotation around (cx, cy)
    SemiLagrangianAdvection1FormOrder2<2>::VelocityFunc velocity =
        [cx, cy](const mfem::Vector &pt, double, int, mfem::Vector &v) {
        v.SetSize(2);
        v[0] = -(pt[1] - cy);
        v[1] =   pt[0] - cx;
    };

    // Taylor-Green initial 1-form
    auto omega_init = [](const mfem::Vector &x, double, mfem::Vector &v) {
        v.SetSize(2);
        v[0] =  std::cos(M_PI * x[0]) * std::sin(M_PI * x[1]);
        v[1] = -std::sin(M_PI * x[0]) * std::cos(M_PI * x[1]);
    };

    // Boundary: ω₀ for departure points outside domain (the line integral
    // evaluates ω₀, so the boundary callback must also return ω₀).
    SemiLagrangianAdvection1FormOrder2<2>::BoundaryFunc boundary =
        [&omega_init](const mfem::Vector &pt, double t, int,
                      mfem::Vector &v) {
        omega_init(pt, t, v);
    };

    // Use a small τ so that no departure points exit the domain.
    // This isolates the spatial convergence rate from boundary effects.
    // (When departure points exit, the boundary callback provides exact ω₀
    // while the interior uses the FE interpolant — this seam produces an
    // O(h^{3/2}) error that degrades the observed rate.)
    const double test_tau = 1e-4;
    const double test_ct = std::cos(test_tau), test_st = std::sin(test_tau);

    auto test_exact = [cx, cy, test_ct, test_st](
        const mfem::Vector &x, double, mfem::Vector &v) {
        double dx = x[0] - cx, dy = x[1] - cy;
        double dep0 = cx + test_ct * dx + test_st * dy;
        double dep1 = cy - test_st * dx + test_ct * dy;
        double w0 =  std::cos(M_PI * dep0) * std::sin(M_PI * dep1);
        double w1 = -std::sin(M_PI * dep0) * std::cos(M_PI * dep1);
        v.SetSize(2);
        v[0] = test_ct * w0 - test_st * w1;
        v[1] = test_st * w0 + test_ct * w1;
    };

    const std::vector<int> Ns = {4, 8, 16, 32};
    std::vector<double> errors;

    for (int N : Ns)
    {
        mfem::Mesh mesh = MakeAttributedUnitSquareMesh(N, N);
        mfem::ND_FECollection fec(2, mesh.Dimension());
        mfem::FiniteElementSpace nd(&mesh, &fec);

        mfem::GridFunction omega_old(&nd), omega_new(&nd);
        mfem::VectorFunctionCoefficient coef_init(2, omega_init);
        coef_init.SetTime(0.0);
        omega_old.ProjectCoefficient(coef_init);

        SemiLagrangianAdvection1FormOrder2<2> advection(nd);
        advection.Apply(velocity, boundary, test_tau, test_tau,
                        omega_old, omega_new, 2);

        mfem::VectorFunctionCoefficient coef_exact(2, test_exact);
        coef_exact.SetTime(0.0);
        double l2_err = omega_new.ComputeL2Error(coef_exact);
        errors.push_back(l2_err);
    }

    // Print convergence table
    std::cout << "  h-convergence table (tau=" << test_tau << "):"
              << std::endl;
    for (std::size_t i = 0; i < errors.size(); ++i)
    {
        std::cout << "  N=" << std::setw(3) << Ns[i]
                  << "  h=" << 1.0 / Ns[i]
                  << "  L2_err=" << std::scientific << std::setprecision(4)
                  << errors[i];
        if (i > 0)
        {
            double rate = std::log(errors[i - 1] / errors[i])
                          / std::log(2.0);
            std::cout << "  rate=" << std::fixed << std::setprecision(2)
                      << rate;
        }
        std::cout << std::endl;
    }

    // Check convergence rates (expect ≈ 2.0 for O(h²))
    for (std::size_t i = 1; i < errors.size(); ++i)
    {
        double rate = std::log(errors[i - 1] / errors[i]) / std::log(2.0);
        EXPECT_GT(rate, 1.8)
            << "Rate at N=" << Ns[i] << " should be ≈ 2.0 (O(h²))";
        EXPECT_LT(rate, 2.2)
            << "Rate at N=" << Ns[i] << " should not exceed 2.2";
    }
}
