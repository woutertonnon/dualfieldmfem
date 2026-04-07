#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include <mpi.h>
#include "SemiLagrangianAdvection.h"
#include "mfem.hpp"

using namespace mfem;

namespace
{

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
        if (std::abs(cx) < tol) { attr = 1; }
        else if (std::abs(cx - 1.0) < tol) { attr = 2; }
        else if (std::abs(cy) < tol) { attr = 3; }
        else if (std::abs(cy - 1.0) < tol) { attr = 4; }

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

        double cx = 0.0, cy = 0.0, cz = 0.0;
        for (int i = 0; i < verts.Size(); ++i)
        {
            const double *v = mesh.GetVertex(verts[i]);
            cx += v[0]; cy += v[1]; cz += v[2];
        }
        cx /= static_cast<double>(verts.Size());
        cy /= static_cast<double>(verts.Size());
        cz /= static_cast<double>(verts.Size());

        int attr = 0;
        if (std::abs(cx) < tol) { attr = 1; }
        else if (std::abs(cx - 1.0) < tol) { attr = 2; }
        else if (std::abs(cy) < tol) { attr = 3; }
        else if (std::abs(cy - 1.0) < tol) { attr = 4; }
        else if (std::abs(cz) < tol) { attr = 5; }
        else if (std::abs(cz - 1.0) < tol) { attr = 6; }

        MFEM_VERIFY(attr != 0, "Could not classify 3D boundary element.");
        mesh.SetBdrAttribute(be, attr);
    }

    mesh.SetAttributes(false, true);
    return mesh;
}

SemiLagrangianAdvection1Form<1>::VelocityFunc
ConstantVelocity2D(double vx, double vy)
{
    return [vx, vy](const Vector &, double, int, Vector &v)
    {
        v.SetSize(2);
        v[0] = vx;
        v[1] = vy;
    };
}

SemiLagrangianAdvection1Form<1>::VelocityFunc
ConstantVelocity3D(double vx, double vy, double vz)
{
    return [vx, vy, vz](const Vector &, double, int, Vector &v)
    {
        v.SetSize(3);
        v[0] = vx;
        v[1] = vy;
        v[2] = vz;
    };
}

SemiLagrangianAdvection1Form<1>::BoundaryFunc
ZeroBoundary(int dim)
{
    return [dim](const Vector &, double, int, Vector &v)
    {
        v.SetSize(dim);
        v = 0.0;
    };
}

void ProjectAffineField2D(GridFunction &omega)
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

void ProjectAffineField3D(GridFunction &omega)
{
    VectorFunctionCoefficient coeff(
        3,
        [](const Vector &x, double, Vector &v)
        {
            v.SetSize(3);
            v[0] = 1.0 + 0.35 * x[0] - 0.15 * x[1] + 0.1 * x[2];
            v[1] = -0.2 + 0.10 * x[0] + 0.45 * x[1] - 0.2 * x[2];
            v[2] = 0.3 - 0.25 * x[0] + 0.05 * x[1] + 0.4 * x[2];
        });
    coeff.SetTime(0.0);
    omega.ProjectCoefficient(coeff);
}

/// Compute core-weighted edge partition (same logic as the MPI app).
void ComputeEdgePartition(int n_edges, int myid, int nprocs,
                          const std::vector<int> &all_threads,
                          int &edge_start, int &edge_end,
                          std::vector<int> &counts,
                          std::vector<int> &displs)
{
    const int total_threads =
        std::accumulate(all_threads.begin(), all_threads.end(), 0);

    counts.resize(nprocs);
    displs.resize(nprocs);

    int prev_end = 0;
    for (int r = 0; r < nprocs; ++r)
    {
        displs[r] = prev_end;
        int next_end;
        if (r == nprocs - 1)
        {
            next_end = n_edges;
        }
        else
        {
            double cumulative_weight = 0.0;
            for (int q = 0; q <= r; ++q)
                cumulative_weight += all_threads[q];
            next_end = static_cast<int>(
                std::round(static_cast<double>(n_edges) *
                           cumulative_weight / total_threads));
        }
        counts[r] = next_end - prev_end;
        prev_end = next_end;
    }

    edge_start = displs[myid];
    edge_end = displs[myid] + counts[myid];
}

} // anonymous namespace

// =========================================================================
// Test: Edge partitioning covers all edges exactly once
// =========================================================================

TEST(EdgePartition, CoversAllEdges)
{
    int myid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Test with various edge counts
    for (int n_edges : {1, 7, 16, 100, 1023, 10000})
    {
        std::vector<int> all_threads(nprocs, 1); // uniform weights
        int edge_start, edge_end;
        std::vector<int> counts, displs;
        ComputeEdgePartition(n_edges, myid, nprocs, all_threads,
                             edge_start, edge_end, counts, displs);

        // Every rank's range must be non-negative
        ASSERT_GE(edge_end - edge_start, 0)
            << "n_edges=" << n_edges << " rank=" << myid;

        // Total of all counts must equal n_edges
        int total = 0;
        for (int r = 0; r < nprocs; ++r) total += counts[r];
        EXPECT_EQ(total, n_edges)
            << "n_edges=" << n_edges;

        // Partitions must be contiguous and non-overlapping
        for (int r = 1; r < nprocs; ++r)
        {
            EXPECT_EQ(displs[r], displs[r - 1] + counts[r - 1])
                << "n_edges=" << n_edges << " rank=" << r;
        }

        // First starts at 0, last ends at n_edges
        EXPECT_EQ(displs[0], 0);
        EXPECT_EQ(displs[nprocs - 1] + counts[nprocs - 1], n_edges);
    }
}

TEST(EdgePartition, WeightedPartitionProportional)
{
    int myid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (nprocs < 2) { GTEST_SKIP() << "Requires >= 2 ranks"; }

    const int n_edges = 10000;
    // First rank gets 4 threads, rest get 1
    std::vector<int> all_threads(nprocs, 1);
    all_threads[0] = 4;

    int edge_start, edge_end;
    std::vector<int> counts, displs;
    ComputeEdgePartition(n_edges, myid, nprocs, all_threads,
                         edge_start, edge_end, counts, displs);

    // Rank 0 should get roughly 4x the edges of other ranks
    if (nprocs == 2)
    {
        // 4/(4+1) = 80% of edges for rank 0
        double ratio = static_cast<double>(counts[0]) / n_edges;
        EXPECT_NEAR(ratio, 0.8, 0.01);
    }

    // Total must still be exact
    int total = 0;
    for (int r = 0; r < nprocs; ++r) total += counts[r];
    EXPECT_EQ(total, n_edges);
}

// =========================================================================
// Test: Edge-range Apply matches full Apply (2D)
// =========================================================================

TEST(SemiLagrangianMPI, EdgeRangeMatchesFullApply2D)
{
    int myid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    Mesh mesh = MakeAttributedUnitSquareMesh(4, 4);
    ND_FECollection fec(1, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction omega_old(&nd);
    GridFunction omega_serial(&nd);
    GridFunction omega_mpi(&nd);
    ProjectAffineField2D(omega_old);
    omega_serial = 0.0;
    omega_mpi = 0.0;

    SemiLagrangianAdvection1Form<1> advection(nd);
    auto velocity = ConstantVelocity2D(0.25, -0.15);
    auto boundary = ZeroBoundary(2);

    // Serial: full edge range (default)
    advection.Apply(velocity, boundary, 0.3, 0.1,
                    omega_old, omega_serial, 1);

    // MPI: each rank processes its edge range, then Allgatherv
    const int n_edges = mesh.GetNEdges();
    std::vector<int> all_threads(nprocs, 1);
    int edge_start, edge_end;
    std::vector<int> counts, displs;
    ComputeEdgePartition(n_edges, myid, nprocs, all_threads,
                         edge_start, edge_end, counts, displs);

    advection.Apply(velocity, boundary, 0.3, 0.1,
                    omega_old, omega_mpi, 1,
                    nullptr, nullptr, 2,
                    SemiLagrangianAdvection1Form<1>::kSingleElement,
                    edge_start, edge_end);

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   omega_mpi.GetData(), counts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    // Compare: every DOF must match
    double max_diff = 0.0;
    for (int i = 0; i < nd.GetNDofs(); ++i)
    {
        max_diff = std::max(max_diff,
                            std::abs(omega_mpi(i) - omega_serial(i)));
    }
    EXPECT_LT(max_diff, 1e-12)
        << "MPI edge-range Apply differs from serial (2D)";
}

// =========================================================================
// Test: Edge-range Apply matches full Apply (3D)
// =========================================================================

TEST(SemiLagrangianMPI, EdgeRangeMatchesFullApply3D)
{
    int myid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    Mesh mesh = MakeAttributedUnitCubeMesh(2, 2, 2);
    ND_FECollection fec(1, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction omega_old(&nd);
    GridFunction omega_serial(&nd);
    GridFunction omega_mpi(&nd);
    ProjectAffineField3D(omega_old);
    omega_serial = 0.0;
    omega_mpi = 0.0;

    SemiLagrangianAdvection1Form<1> advection(nd);
    auto velocity = ConstantVelocity3D(0.15, -0.10, 0.20);
    auto boundary = ZeroBoundary(3);

    // Serial: full edge range
    advection.Apply(velocity, boundary, 0.3, 0.1,
                    omega_old, omega_serial, 1);

    // MPI: partitioned
    const int n_edges = mesh.GetNEdges();
    std::vector<int> all_threads(nprocs, 1);
    int edge_start, edge_end;
    std::vector<int> counts, displs;
    ComputeEdgePartition(n_edges, myid, nprocs, all_threads,
                         edge_start, edge_end, counts, displs);

    advection.Apply(velocity, boundary, 0.3, 0.1,
                    omega_old, omega_mpi, 1,
                    nullptr, nullptr, 2,
                    SemiLagrangianAdvection1Form<1>::kSingleElement,
                    edge_start, edge_end);

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   omega_mpi.GetData(), counts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    double max_diff = 0.0;
    for (int i = 0; i < nd.GetNDofs(); ++i)
    {
        max_diff = std::max(max_diff,
                            std::abs(omega_mpi(i) - omega_serial(i)));
    }
    EXPECT_LT(max_diff, 1e-12)
        << "MPI edge-range Apply differs from serial (3D)";
}

// =========================================================================
// Test: Heun (trace_order=2) with edge range matches serial
// =========================================================================

TEST(SemiLagrangianMPI, HeunEdgeRangeMatchesSerial)
{
    int myid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    Mesh mesh = MakeAttributedUnitSquareMesh(4, 4);
    ND_FECollection fec(1, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction omega_old(&nd);
    GridFunction omega_serial(&nd);
    GridFunction omega_mpi(&nd);
    ProjectAffineField2D(omega_old);
    omega_serial = 0.0;
    omega_mpi = 0.0;

    SemiLagrangianAdvection1Form<1> advection(nd);
    auto velocity = ConstantVelocity2D(0.30, 0.20);
    auto boundary = ZeroBoundary(2);

    // Serial with Heun (trace_order=2)
    advection.Apply(velocity, boundary, 0.3, 0.1,
                    omega_old, omega_serial, 2);

    // MPI partitioned with Heun
    const int n_edges = mesh.GetNEdges();
    std::vector<int> all_threads(nprocs, 1);
    int edge_start, edge_end;
    std::vector<int> counts, displs;
    ComputeEdgePartition(n_edges, myid, nprocs, all_threads,
                         edge_start, edge_end, counts, displs);

    advection.Apply(velocity, boundary, 0.3, 0.1,
                    omega_old, omega_mpi, 2,
                    nullptr, nullptr, 2,
                    SemiLagrangianAdvection1Form<1>::kSingleElement,
                    edge_start, edge_end);

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   omega_mpi.GetData(), counts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    double max_diff = 0.0;
    for (int i = 0; i < nd.GetNDofs(); ++i)
    {
        max_diff = std::max(max_diff,
                            std::abs(omega_mpi(i) - omega_serial(i)));
    }
    EXPECT_LT(max_diff, 1e-12)
        << "MPI Heun edge-range Apply differs from serial";
}

// =========================================================================
// Test: Zero velocity with edge range is still identity
// =========================================================================

TEST(SemiLagrangianMPI, ZeroVelocityEdgeRangeIsIdentity)
{
    int myid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    Mesh mesh = MakeAttributedUnitSquareMesh(4, 4);
    ND_FECollection fec(1, mesh.Dimension());
    FiniteElementSpace nd(&mesh, &fec);

    GridFunction omega_old(&nd);
    GridFunction omega_mpi(&nd);
    ProjectAffineField2D(omega_old);
    omega_mpi = 0.0;

    SemiLagrangianAdvection1Form<1> advection(nd);
    auto velocity = ConstantVelocity2D(0.0, 0.0);
    auto boundary = ZeroBoundary(2);

    const int n_edges = mesh.GetNEdges();
    std::vector<int> all_threads(nprocs, 1);
    int edge_start, edge_end;
    std::vector<int> counts, displs;
    ComputeEdgePartition(n_edges, myid, nprocs, all_threads,
                         edge_start, edge_end, counts, displs);

    advection.Apply(velocity, boundary, 0.4, 0.2,
                    omega_old, omega_mpi, 1,
                    nullptr, nullptr, 2,
                    SemiLagrangianAdvection1Form<1>::kSingleElement,
                    edge_start, edge_end);

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   omega_mpi.GetData(), counts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    double max_diff = 0.0;
    for (int i = 0; i < nd.GetNDofs(); ++i)
    {
        max_diff = std::max(max_diff,
                            std::abs(omega_mpi(i) - omega_old(i)));
    }
    EXPECT_LT(max_diff, 1e-12)
        << "Zero velocity with MPI edge partitioning should be identity";
}

// =========================================================================
// Custom main: MPI init/finalize around GoogleTest
// =========================================================================

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    // Only rank 0 prints test output
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (myid != 0)
    {
        ::testing::TestEventListeners &listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
