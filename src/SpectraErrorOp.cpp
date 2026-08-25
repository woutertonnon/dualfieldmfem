#include "SpectraErrorOp.h"
#include "StokesOperator.h"

#include <Spectra/GenEigsSolver.h>

#include <algorithm>
#include <filesystem>
#include <iomanip>
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
    const bool            printResults,
    Eigen::MatrixXcd*     eigenvectors,
    const bool            saveEigenvectorsVTU,
    const StokesNitsche::StokesNitscheOperator* stokesOp,
    const std::string&    vtuPrefix)
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
        Eigen::MatrixXcd eigenvecs = eigs.eigenvectors();

        if (eigenvectors)
        {
            *eigenvectors = eigenvecs;
        }

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

            std::cout << "Eigenvector norms (2-norm):\n";
            std::cout << std::left << std::setw(6) << "Idx" << std::right
                      << std::setw(20) << "||Re(v)||_2" << std::setw(20)
                      << "||Im(v)||_2" << "\n";
            std::cout << std::string(46, '-') << "\n";
            for (int i = 0; i < eigenvecs.cols(); i++)
            {
                const double re_norm = eigenvecs.col(i).real().norm();
                const double im_norm = eigenvecs.col(i).imag().norm();
                std::cout << std::left << std::setw(6) << i << std::right
                          << std::setw(20) << re_norm << std::setw(20)
                          << im_norm << "\n";
            }
            std::cout << std::string(46, '-') << "\n";

            // Restore previous cout state
            std::cout.copyfmt(oldState);
        }

        if (saveEigenvectorsVTU)
        {
            MFEM_VERIFY(
                stokesOp != nullptr,
                "Saving eigenvectors to VTU requires a StokesNitscheOperator.");

            const auto& hcurl = stokesOp->getHCurlSpace();
            const auto& h1    = stokesOp->getH1Space();
            const int   nu    = hcurl.GetNDofs();
            const int   np    = h1.GetNDofs();

            MFEM_VERIFY(
                nu + np == mat.Height(),
                "Eigenvector size does not match Stokes (HCurl,H1) block sizes.");

            std::filesystem::path prefix_path(vtuPrefix);
            if (prefix_path.has_parent_path())
            {
                std::filesystem::create_directories(prefix_path.parent_path());
            }

            for (int i = 0; i < eigenvecs.cols(); i++)
            {
                mfem::GridFunction u_real(
                    const_cast<mfem::FiniteElementSpace*>(&hcurl));
                mfem::GridFunction p_real(
                    const_cast<mfem::FiniteElementSpace*>(&h1));
                mfem::GridFunction u_imag(
                    const_cast<mfem::FiniteElementSpace*>(&hcurl));
                mfem::GridFunction p_imag(
                    const_cast<mfem::FiniteElementSpace*>(&h1));
                mfem::GridFunction u_mag(
                    const_cast<mfem::FiniteElementSpace*>(&hcurl));
                mfem::GridFunction p_mag(
                    const_cast<mfem::FiniteElementSpace*>(&h1));

                for (int j = 0; j < nu; ++j)
                {
                    u_real(j) = eigenvecs(j, i).real();
                    u_imag(j) = eigenvecs(j, i).imag();
                    u_mag(j)  = std::abs(eigenvecs(j, i));
                }
                for (int j = 0; j < np; ++j)
                {
                    p_real(j) = eigenvecs(nu + j, i).real();
                    p_imag(j) = eigenvecs(nu + j, i).imag();
                    p_mag(j)  = std::abs(eigenvecs(nu + j, i));
                }

                const std::string dc_name =
                    vtuPrefix + "_eigvec_" + std::to_string(i);
                mfem::ParaViewDataCollection dc(
                    dc_name,
                    &const_cast<StokesNitsche::StokesNitscheOperator*>(stokesOp)
                         ->getMesh());
                dc.SetLevelsOfDetail(1);
                dc.SetDataFormat(mfem::VTKFormat::BINARY);
                dc.SetHighOrderOutput(true);
                dc.SetCycle(i);
                dc.SetTime(static_cast<double>(i));
                dc.RegisterField("u_real", &u_real);
                dc.RegisterField("p_real", &p_real);
                dc.RegisterField("u_imag", &u_imag);
                dc.RegisterField("p_imag", &p_imag);
                dc.RegisterField("u_mag", &u_mag);
                dc.RegisterField("p_mag", &p_mag);
                dc.Save();
            }
        }
    }
    else
    {
        std::cerr << "Spectra computation failed or did not converge.\n";
        return Eigen::VectorXcd();
    }

    return results;
}
