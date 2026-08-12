#ifndef SPECTRA_ERROR_OP_HPP
#define SPECTRA_ERROR_OP_HPP

#include <Eigen/Core>
#include <complex>
#include <mfem.hpp>

class ErrorOperator : public mfem::Operator
{
private:
    const mfem::Operator& matOp;
    const mfem::Operator& precOp;
    mutable mfem::Vector  zVec;

public:
    ErrorOperator(const mfem::Operator& mat, const mfem::Operator& prec);

    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const override;
};

class SpectraAdapter
{
private:
    const mfem::Operator& mfemOp;
    mutable mfem::Vector  xVec;
    mutable mfem::Vector  yVec;

public:
    using Scalar = double;

    SpectraAdapter(const mfem::Operator& op);

    int rows() const;
    int cols() const;

    void perform_op(const double* xIn, double* yOut) const;
};

Eigen::VectorXcd computeErrorOperatorEigenvalues(
    const mfem::Operator& mat,
    const mfem::Operator& prec,
    const int             numEigenvalues = 1,
    const double          tol            = 1e-3,
    const bool            printResults   = false);

#endif  // SPECTRA_ERROR_OP_HPP
