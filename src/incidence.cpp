#include "incidence.h"

#include <stdexcept>

mfem::SparseMatrix assembleDiscreteGradient(
    mfem::FiniteElementSpace* h1,
    mfem::FiniteElementSpace* hcurl)
{
    // 1. Verify the meshes match
    MFEM_VERIFY(
        h1->GetMesh() == hcurl->GetMesh(),
        "H1 and H(curl) spaces must be defined on the same mesh!");

    // 2. Verify De Rham complex components (Lagrangian -> Nedelec)
    auto h1_fec = dynamic_cast<const mfem::H1_FECollection*>(h1->FEColl());
    auto nd_fec = dynamic_cast<const mfem::ND_FECollection*>(hcurl->FEColl());

    MFEM_VERIFY(
        h1_fec != nullptr,
        "Domain space must use an H1_FECollection (Lagrangian).");
    MFEM_VERIFY(
        nd_fec != nullptr,
        "Range space must use an ND_FECollection (Nedelec).");

    // 3. Verify polynomial orders are high enough and compatible
    const int p_h1 = h1_fec->GetOrder();
    const int p_nd = nd_fec->GetOrder();
    MFEM_VERIFY(
        p_h1 == p_nd,
        "Discrete De Rham complex requires H1 and H(curl) to have the exact "
        "same polynomial order!");

    // 4. Assemble the exact discrete gradient operator
    mfem::DiscreteLinearOperator grad(h1, hcurl);
    grad.AddDomainInterpolator(new mfem::GradientInterpolator());
    grad.Assemble();
    grad.Finalize();

    return *grad.LoseMat();
}

mfem::SparseMatrix assembleDiscreteCurl(
    mfem::FiniteElementSpace* hcurl,
    mfem::FiniteElementSpace* hdiv)
{
    // 1. Verify the meshes match
    MFEM_VERIFY(
        hcurl->GetMesh() == hdiv->GetMesh(),
        "H(curl) and H(div) spaces must be defined on the same mesh!");

    // 2. Verify De Rham complex components (Nedelec -> Raviart-Thomas)
    auto nd_fec = dynamic_cast<const mfem::ND_FECollection*>(hcurl->FEColl());
    auto rt_fec = dynamic_cast<const mfem::RT_FECollection*>(hdiv->FEColl());

    MFEM_VERIFY(
        nd_fec != nullptr,
        "Domain space must use an ND_FECollection (Nedelec).");
    MFEM_VERIFY(
        rt_fec != nullptr,
        "Range space must use an RT_FECollection (Raviart-Thomas).");

    // 3. Verify polynomial orders match the sequence (ND is p, RT is p-1)
    const int p_nd = nd_fec->GetOrder();
    const int p_rt = rt_fec->GetOrder();
    MFEM_VERIFY(
        p_nd == p_rt,
        "Discrete De Rham complex requires H(div) order to be exactly one less "
        "than H(curl) order!");

    // 4. Assemble the exact discrete curl operator
    mfem::DiscreteLinearOperator curl(hcurl, hdiv);
    curl.AddDomainInterpolator(new mfem::CurlInterpolator());
    curl.Assemble();
    curl.Finalize();

    return *curl.LoseMat();
}
