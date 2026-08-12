#include "StokesMG.h"
#include "BoundaryOperators.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>
#include <vector>
#include <string>
#include <algorithm> // For std::max

// --- Configuration ---
constexpr bool DIRECT_SOLVE = false;

// --- Helper Math Functions for the Trig Manufactured Solution ---
constexpr double PI = 3.14159265358979323846;

inline double f_trig(double t)
{
    return std::sin(PI * t) * std::sin(PI * t);
}
inline double h_trig(double t)
{
    return PI * std::sin(2.0 * PI * t);
}
inline double d2f_trig(double t)
{
    return 2.0 * PI * PI * std::cos(2.0 * PI * t);
}
inline double d2h_trig(double t)
{
    return -4.0 * PI * PI * PI * std::sin(2.0 * PI * t);
}

// --- Exact Solution & Source Functions ---

void u_exact_func(const mfem::Vector& x, mfem::Vector& u)
{
    double px = x(0), py = x(1), pz = x(2);
    u(0) = f_trig(px) * h_trig(py) * h_trig(pz);
    u(1) = h_trig(px) * f_trig(py) * h_trig(pz);
    u(2) = -2.0 * h_trig(px) * h_trig(py) * f_trig(pz);
}

// 1. EXACT VELOCITY CURL FOR H(CURL) NORM
void curl_u_exact_func(const mfem::Vector& x, mfem::Vector& curl)
{
    double px = x(0), py = x(1), pz = x(2);

    double f_x = f_trig(px), h_x = h_trig(px), d2f_x = d2f_trig(px);
    double f_y = f_trig(py), h_y = h_trig(py), d2f_y = d2f_trig(py);
    double f_z = f_trig(pz), h_z = h_trig(pz), d2f_z = d2f_trig(pz);

    curl(0) = -2.0 * h_x * d2f_y * f_z - h_x * f_y * d2f_z;
    curl(1) = f_x * h_y * d2f_z + 2.0 * d2f_x * h_y * f_z;
    curl(2) = d2f_x * f_y * h_z - f_x * d2f_y * h_z;
}

double p_exact_func(const mfem::Vector& x)
{
    return std::cos(PI * x(0)) + std::cos(PI * x(1)) + std::cos(PI * x(2));
}

// 2. EXACT PRESSURE GRADIENT FOR H1 NORM
void grad_p_exact_func(const mfem::Vector& x, mfem::Vector& grad)
{
    grad(0) = -PI * std::sin(PI * x(0));
    grad(1) = -PI * std::sin(PI * x(1));
    grad(2) = -PI * std::sin(PI * x(2));
}

void f_rhs_func(const mfem::Vector& x, mfem::Vector& f)
{
    double px = x(0), py = x(1), pz = x(2);

    double f_x = f_trig(px), h_x = h_trig(px), d2f_x = d2f_trig(px), d2h_x = d2h_trig(px);
    double f_y = f_trig(py), h_y = h_trig(py), d2f_y = d2f_trig(py), d2h_y = d2h_trig(py);
    double f_z = f_trig(pz), h_z = h_trig(pz), d2f_z = d2f_trig(pz), d2h_z = d2h_trig(pz);

    double lap_u_x = d2f_x * h_y * h_z + f_x * d2h_y * h_z + f_x * h_y * d2h_z;
    double lap_u_y = d2h_x * f_y * h_z + h_x * d2f_y * h_z + h_x * f_y * d2h_z;
    double lap_u_z = -2.0 * (d2h_x * h_y * f_z + h_x * d2h_y * f_z + h_x * h_y * d2f_z);

    double grad_p_x = -PI * std::sin(PI * px);
    double grad_p_y = -PI * std::sin(PI * py);
    double grad_p_z = -PI * std::sin(PI * pz);

    f(0) = -lap_u_x + grad_p_x;
    f(1) = -lap_u_y + grad_p_y;
    f(2) = -lap_u_z + grad_p_z;
}

// --- Convergence Study ---
void RunStokesMGStudy(std::shared_ptr<mfem::Mesh> mesh_ptr,
                      const int max_refs,
                      const bool save_solution = false)
{
    constexpr double THETA   = 1.0;
    constexpr double PENALTY = 10.0;
    constexpr double FACTOR  = 1.0;
    constexpr double TAU     = 0.0;

    StokesNitsche::StokesMG mg_solver(mesh_ptr, TAU,
            THETA, PENALTY, FACTOR,
            StokesNitsche::MassLumping::Diagonal,
            StokesNitsche::SmootherType::Chebyshev
    );
    mg_solver.setOperatorMode(StokesNitsche::OperatorMode::Galerkin);
    mg_solver.setIterativeMode(false);
    mg_solver.setCycleType(StokesNitsche::MGCycleType::VCycle);
    mg_solver.setSmoothIterations(1, 1);

    // Standard Coefficients
    mfem::VectorFunctionCoefficient u_coeff(3, u_exact_func);
    mfem::FunctionCoefficient       p_coeff(p_exact_func);
    mfem::VectorFunctionCoefficient f_coeff(3, f_rhs_func);

    // Sobolev Derivative Coefficients
    mfem::VectorFunctionCoefficient curl_u_coeff(3, curl_u_exact_func);
    mfem::VectorFunctionCoefficient grad_p_coeff(3, grad_p_exact_func);

    std::cout << "\n" << (DIRECT_SOLVE ? "Using UMFPACK (Direct)" : "Using Multigrid Preconditioned GMRES") << "\n\n";

    // Updated Header to include mesh width (h)
    std::cout << std::setw(6)  << "Level"
              << std::setw(12) << "h_max"
              << std::setw(12) << "DOFs"
              << std::setw(10) << "Iters"
              << std::setw(14) << "HCurl(u)" << std::setw(8) << "Rate"
              << std::setw(14) << "H1(p)" << std::setw(8) << "Rate\n";
    std::cout << std::string(84, '-') << std::endl;

    double err_u_prev = 0.0, err_p_prev = 0.0;

    // Optional VTK output initialization
    std::unique_ptr<mfem::ParaViewDataCollection> pd;
    if (save_solution)
    {
        pd = std::make_unique<mfem::ParaViewDataCollection>("Stokes_Manufactured_Solution", mesh_ptr.get());
        pd->SetLevelsOfDetail(1);
        pd->SetDataFormat(mfem::VTKFormat::BINARY);
        pd->SetHighOrderOutput(true);
    }

    for (int l = 0; l <= max_refs; ++l)
    {
        // if (l > 0)
        // {
        //     mg_solver.removeRefinement();
        //     mg_solver.addRefinement();
        // }
        // mg_solver.addRefinement(1);
        mg_solver.addRefinement();


        StokesNitsche::StokesNitscheOperator& op =
            *const_cast<StokesNitsche::StokesNitscheOperator*>(&mg_solver.getFinestOperator());

        mfem::Mesh& current_mesh = const_cast<mfem::Mesh&>(op.getMesh());

        // Calculate maximum mesh width (h_max)
        double h_max = 0.0;
        for (int i = 0; i < current_mesh.GetNE(); ++i)
            h_max = std::max(h_max, current_mesh.GetElementSize(i));

        if (save_solution)
            pd->SetMesh(&current_mesh);

        op.setOperatorMode(StokesNitsche::OperatorMode::Galerkin);

        const unsigned int nu = op.getHCurlSpace().GetNDofs();
        const unsigned int np = op.getH1Space().GetNDofs();
        const int extra_dofs = DIRECT_SOLVE ? 1 : 0;

        mfem::Vector rhs(nu + np + extra_dofs);
        rhs = 0.0;

        mfem::FiniteElementSpace& hcurl = op.getHCurlSpace();
        mfem::FiniteElementSpace& h1 = op.getH1Space();

        mfem::LinearForm fu(&hcurl, rhs.GetData());
        fu.AddBdrFaceIntegrator(
            new ND_NitscheLFIntegrator(THETA, PENALTY, u_coeff, FACTOR)
        );
        fu.AddDomainIntegrator(
            new mfem::VectorFEDomainLFIntegrator(f_coeff)
        );
        fu.Assemble();

        mfem::Vector x(nu + np + extra_dofs);
        x = 0.0;
        mfem::FGMRESSolver gmres;

        if (DIRECT_SOLVE)
        {
            mfem::UMFPackSolver solver;
            mfem::SparseMatrix A = *op.getFullGalerkinSystem();
            solver.SetOperator(A);
            solver.Mult(rhs, x);
        }
        else
        {
            gmres.SetAbsTol(1e-12);
            gmres.SetRelTol(1e-6);
            gmres.SetMaxIter(500);
            gmres.SetPrintLevel(0);
            gmres.SetOperator(op);
            gmres.SetPreconditioner(mg_solver);
            gmres.SetKDim(128);

            gmres.Mult(rhs, x);
            op.eliminateConstants(x);
        }

        mfem::GridFunction u_h(&hcurl, x.GetData());
        mfem::GridFunction p_h(&h1, x.GetData() + nu);

        // COMPUTE SOBOLEV NORMS
        const double err_u = u_h.ComputeHCurlError(&u_coeff, &curl_u_coeff);
        const double err_p = p_h.ComputeH1Error(&p_coeff, &grad_p_coeff);

        double rate_u = (l > 0) ? std::log2(err_u_prev / err_u) : 0.0;
        double rate_p = (l > 0) ? std::log2(err_p_prev / err_p) : 0.0;

        int iters = DIRECT_SOLVE ? 1 : gmres.GetNumIterations();

        // Output with h_max included
        std::cout << std::setw(6)  << l
                  << std::scientific << std::setprecision(4) << std::setw(12) << h_max
                  << std::setw(12) << op.Height()
                  << std::setw(10) << iters
                  << std::scientific << std::setprecision(3)
                  << std::setw(14) << err_u << std::fixed << std::setprecision(2) << std::setw(8) << rate_u
                  << std::scientific << std::setprecision(3)
                  << std::setw(14) << err_p << std::fixed << std::setprecision(2) << std::setw(8) << rate_p
                  << std::endl;

        err_u_prev = err_u;
        err_p_prev = err_p;

        if (save_solution)
        {
            mfem::GridFunction u_exact_gf(&hcurl);
            u_exact_gf.ProjectCoefficient(u_coeff);
            mfem::GridFunction p_exact_gf(&h1);
            p_exact_gf.ProjectCoefficient(p_coeff);

            pd->SetCycle(l);
            pd->SetTime((double)l);
            pd->RegisterField("velocity", &u_h);
            pd->RegisterField("pressure", &p_h);
            pd->RegisterField("velocity_exact", &u_exact_gf);
            pd->RegisterField("pressure_exact", &p_exact_gf);
            pd->Save();
        }
    }
    std::cout << std::string(84, '-') << "\n" << std::endl;
}

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_OPENMP
    mfem::Device("omp");
#endif
    const unsigned int n = 1;
    auto mesh_ptr = std::make_shared<mfem::Mesh>(
        mfem::Mesh::MakeCartesian3D(n, n, n, mfem::Element::TETRAHEDRON)
    );

    const bool save_results = false;
    RunStokesMGStudy(mesh_ptr, 6, save_results);

    return 0;
}
