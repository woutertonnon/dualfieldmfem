#include "SpectraErrorOp.h"

#include <Spectra/GenEigsSolver.h>

#include <algorithm>
#include <iostream>

ErrorOperator::ErrorOperator(
    const mfem::Operator& mat,
    const mfem::Operator& prec)
    : mfem::Operator(mat.Height()), matOp(mat), precOp(prec), zVec(mat.Height())
{
    MFEM_VERIFY(mat.Height() == mat.Width(), "Matrix must be square");
    MFEM_VERIFY(prec.Height() == prec.Width(), "Preconditioner must be square");
    MFEM_VERIFY(
        mat.Height() == prec.Height(),
        "Matrix and Preconditioner dimensions must match");
}

void ErrorOperator::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    y = x;
    matOp.Mult(x, zVec);
    precOp.AddMult(zVec, y, -1.0);
}

SpectraAdapter::SpectraAdapter(const mfem::Operator& op)
    : mfemOp(op),
      xVec(const_cast<double*>(static_cast<double*>(nullptr)), 0),
      yVec(static_cast<double*>(nullptr), 0)
{
}

int SpectraAdapter::rows() const { return mfemOp.Height(); }

int SpectraAdapter::cols() const { return mfemOp.Width(); }

void SpectraAdapter::perform_op(const double* xIn, double* yOut) const
{
    xVec.SetDataAndSize(const_cast<double*>(xIn), mfemOp.Width());
    yVec.SetDataAndSize(yOut, mfemOp.Height());

    mfemOp.Mult(xVec, yVec);

    xVec.SetDataAndSize(nullptr, 0);
    yVec.SetDataAndSize(nullptr, 0);
}

Eigen::VectorXcd computeErrorOperatorEigenvalues(
    const mfem::Operator& mat,
    const mfem::Operator& prec,
    const int             numEigenvalues,
    const double          tol,
    const bool            printResults)
{
    ErrorOperator  errorOp(mat, prec);
    SpectraAdapter spectraOp(errorOp);

    const int ncv = std::min(
        mat.Height(),
        std::max(32, std::min(2 * numEigenvalues + 1, mat.Height())));

    Spectra::GenEigsSolver<SpectraAdapter> eigs(spectraOp, numEigenvalues, ncv);

    eigs.init();

    // Pass tolerance here (max iterations default is 1000, can also be
    // parameterized)
    const int nConv = eigs.compute(Spectra::SortRule::LargestMagn, 1000, tol);

    Eigen::VectorXcd results;

    if (eigs.info() == Spectra::CompInfo::Successful)
    {
        results = eigs.eigenvalues();

        if (printResults)
        {
            // Save previous cout state to restore later
            std::ios oldState(nullptr);
            oldState.copyfmt(std::cout);

            std::cout << "Spectra: Computed " << nConv
                      << " converged eigenvalues for Error Operator.\n";
            std::cout << std::string(75, '-') << "\n";

            // Header
            std::cout << std::left << std::setw(6) << "Idx" << std::right
                      << std::setw(15) << "Real Part" << std::setw(20)
                      << "Imag Part" << std::setw(18) << "Magnitude" << "\n";

            std::cout << std::string(75, '-') << "\n";

            // Formatting settings for numbers
            std::cout << std::scientific << std::setprecision(6) << std::right;

            for (int i = 0; i < numEigenvalues; i++)
            {
                std::complex<double> ev   = results(i);
                const char           sign = (ev.imag() >= 0) ? '+' : '-';

                std::cout << std::left << std::setw(6) << i << std::right
                          << std::setw(15) << ev.real() << "  " << sign << "  "
                          << std::setw(13) << std::abs(ev.imag()) << "i"
                          << std::setw(18) << std::abs(ev) << "\n";
            }
            std::cout << std::string(75, '-') << "\n";

            // Restore previous cout state
            std::cout.copyfmt(oldState);
        }
    }
    else
    {
        std::cerr << "Spectra computation failed or did not converge.\n";
        return Eigen::VectorXcd();
    }

    return results;
}
