#include <gtest/gtest.h>
#include "mfem.hpp"
#include "incidence.hpp"

void runIncidenceTest(mfem::Mesh& mesh,
                      const unsigned order = 1)
{
    ASSERT_EQ(mesh.Dimension(), 3);
    ASSERT_GT(order, 0);

    const unsigned DIM = 3;

    // 3D De Rham spaces: H1 -> ND -> RT
    mfem::H1_FECollection h1_fec(order    , DIM);
    mfem::ND_FECollection nd_fec(order    , DIM);
    mfem::RT_FECollection rt_fec(order - 1, DIM);

    mfem::FiniteElementSpace h1(&mesh, &h1_fec);
    mfem::FiniteElementSpace nd(&mesh, &nd_fec);
    mfem::FiniteElementSpace rt(&mesh, &rt_fec);

    const mfem::SparseMatrix d0 = assembleDiscreteGradient(&h1, &nd);
    const mfem::SparseMatrix d1 = assembleDiscreteCurl(&nd, &rt);

    std::unique_ptr<mfem::SparseMatrix> d1d0(mfem::Mult(d1, d0));

    ASSERT_NEAR(
        d1d0->MaxNorm() / std::max(d0.MaxNorm(), d1.MaxNorm()),
        0.0, 1e-12
    ) << "Complex property failed at order " << order << std::endl;
    ASSERT_GT(d0.MaxNorm(), 0.0);
    ASSERT_GT(d1.MaxNorm(), 0.0);
}

TEST(IncidenceTest, ComplexPropertyTetsO1)
{
    const unsigned n = 5;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        n, n + 1, n + 2, mfem::Element::TETRAHEDRON
    );

    runIncidenceTest(mesh);
}

TEST(IncidenceTest, ComplexPropertyTetsOp)
{
    const unsigned n = 2,
                   pmax = 3;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        n, n + 1, n + 2, mfem::Element::TETRAHEDRON
    );

    for(unsigned p = 1; p <= pmax; ++p)
        runIncidenceTest(mesh, p);
}

// TEST(IncidenceTest, ComplexPropertyHexO1)
// {
//     const unsigned n = 5;
//     mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
//         n, n + 1, n + 2, mfem::Element::HEXAHEDRON
//     );
//
//     runIncidenceTest(mesh, 1);
// }
//
// TEST(IncidenceTest, ComplexPropertyHexOp)
// {
//     const unsigned n = 3,
//                    pmax = 3;
//     mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
//         n, n + 1, n + 2, mfem::Element::HEXAHEDRON
//     );
//
//     for(unsigned p = 1; p <= pmax; ++p)
//         runIncidenceTest(mesh, p);
// }


// TEST(IncidenceTest, CurlCurl3D)
// {
//     const unsigned int n = 5,
//                        DIM = 3;
//
//     const mfem::Element::Type el_type =
//         mfem::Element::TETRAHEDRON;
//
//     mfem::ConstantCoefficient one(1.0);
//     mfem::ConstantCoefficient four(4.0);
//
//     mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
//         n, n + 1, n + 2, el_type, std::sqrt(2), std::sqrt(3), std::sqrt(5)
//     );
//
//     const int nv = mesh.GetNV(),
//               ne = mesh.GetNEdges(),
//               nf = mesh.GetNFaces();
//
//     const mfem::SparseMatrix
//         d0 = assembleVertexEdge(mesh),
//         d1 = assembleFaceEdge(mesh, DIM);
//
//     ASSERT_EQ(d0.NumCols(), nv);
//     ASSERT_EQ(d0.NumRows(), ne);
//     ASSERT_EQ(d1.NumRows(), nf);
//     ASSERT_EQ(d1.NumCols(), ne);
//
//     auto hcurl_fec = std::make_unique<mfem::ND_FECollection>(
//         1, DIM
//     );
//     auto hcurl = std::make_unique<mfem::FiniteElementSpace>(
//         &mesh, hcurl_fec.get()
//     );
//
//     auto hdiv_fec = std::make_unique<mfem::RT_FECollection>(
//         0, DIM
//     );
//     auto hdiv = std::make_unique<mfem::FiniteElementSpace>(
//         &mesh, hdiv_fec.get()
//     );
//
//     auto mass_ = std::make_unique<mfem::BilinearForm>(
//         hdiv.get()
//     );
//     mass_->AddDomainIntegrator(
//         new mfem::VectorFEMassIntegrator(four)
//     );
//     mass_->Assemble(); mass_->Finalize();
//     const mfem::SparseMatrix& mass = mass_->SpMat();
//
//     ASSERT_EQ(mass.NumRows(), nf);
//
//     auto curlcurl_ = std::make_unique<mfem::BilinearForm>(
//         hcurl.get()
//     );
//     curlcurl_->AddDomainIntegrator(new mfem::CurlCurlIntegrator(one));
//     curlcurl_->Assemble(); curlcurl_->Finalize();
//     const mfem::SparseMatrix& curlcurl = curlcurl_->SpMat();
//
//     ASSERT_EQ(curlcurl.NumRows(), ne);
//
//     std::unique_ptr<mfem::SparseMatrix> curlcurl_incidence_;
//     {
//         auto tmp = std::unique_ptr<mfem::SparseMatrix>(
//             mfem::Mult(mass, d1)
//         );
//         auto d1T = std::unique_ptr<mfem::SparseMatrix>(
//             mfem::Transpose(d1)
//         );
//         curlcurl_incidence_ = std::unique_ptr<mfem::SparseMatrix>(
//             mfem::Mult(*d1T, *tmp)
//         );
//     }
//     const mfem::SparseMatrix& curlcurl_incidence = *curlcurl_incidence_;
//
//     auto diff = std::unique_ptr<mfem::SparseMatrix>(
//         mfem::Add( 1.0, curlcurl,
//                   -1.0, curlcurl_incidence)
//     );
//
//     ASSERT_GT(curlcurl_incidence.MaxNorm(), 1e-12);
//     ASSERT_GT(curlcurl.MaxNorm(), 1e-12);
//     ASSERT_LT(diff->MaxNorm() / curlcurl.MaxNorm(), 1e-10);
// }
