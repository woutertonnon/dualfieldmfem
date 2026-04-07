#ifndef INCIDENCE_H
#define INCIDENCE_H

#include "mfem.hpp"

mfem::SparseMatrix assembleDiscreteGradient(
    mfem::FiniteElementSpace* h1,
    mfem::FiniteElementSpace* hcurl);
mfem::SparseMatrix assembleDiscreteCurl(
    mfem::FiniteElementSpace* hcurl,
    mfem::FiniteElementSpace* hdiv);
// mfem::SparseMatrix assembleDiscreteDiv(mfem::FiniteElementSpace* hdiv,
// mfem::FiniteElementSpace* l2);

#endif
