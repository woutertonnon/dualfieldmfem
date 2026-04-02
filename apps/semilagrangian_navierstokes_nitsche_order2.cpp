#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <iomanip>
#include <limits>
#include <sstream>
#include <boost/program_options.hpp>

#include "mfem.hpp"
#include "BoundaryOperators.h"
#include "io.h"
#include "StokesOperators.h"
#include "FindElementBFS.h"
#include "SemiLagrangianAdvectionOrder2.h"

using namespace mfem;
using namespace std;
namespace po = boost::program_options;

namespace
{
std::string FormatDuration(double seconds)
{
    if (!std::isfinite(seconds) || seconds < 0.0)
    {
        return "--:--";
    }
    auto total = static_cast<long long>(std::llround(seconds));
    const long long h = total / 3600;
    const long long m = (total % 3600) / 60;
    const long long s = total % 60;

    std::ostringstream os;
    os << std::setfill('0');
    if (h > 0)
    {
        os << std::setw(2) << h << ":";
    }
    os << std::setw(2) << m << ":" << std::setw(2) << s;
    return os.str();
}
}

int main(int argc, char *argv[])
{
    mfem::Device device("omp");

    // ---- Parse command-line options ----
    std::string config_path;

    try
    {
        po::options_description desc("Allowed options");
        desc.add_options()("help,h", "produce help message")("config,c",
                                                             po::value<std::string>(&config_path)
                                                                 ->default_value("../data/config/StokesTest/StokesTest_conv_order2_ref2.json"),
                                                             "path to JSON configuration file");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            std::cout << desc << "\n";
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error parsing command line: " << e.what() << "\n";
        return 1;
    }

    // ---- Configuration ----
    DualFieldConfig config(config_path);

    double viscosity = config.get_viscosity();
    int refinements = config.get_refinements();
    int visualisation = config.get_visualisation();
    int printlevel = config.get_printlevel();
    double tol = config.get_tol();
    std::string mesh_string = config.get_mesh();
    std::string output_file = config.get_outputfile();
    double theta = 1.;
    double Cw = 100.;
    double dt = config.get_dt();
    double T = config.get_T();
    int trace_order = config.get_value<int>("trace_order", 1);  // default Euler

    if (trace_order < 1 || trace_order > 3)
    {
        std::cerr << "[warn] Unsupported trace_order=" << trace_order
                  << ", using Euler (trace_order=1)." << std::endl;
        trace_order = 1;
    }

    // ---- Mesh and FE spaces (order 2) ----
    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; l++)
    {
        mesh.UniformRefinement();
    }
    int dim = mesh.Dimension();

    const int order = 2;
    mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order, dim);
    mfem::FiniteElementCollection *fec_CG = new mfem::H1_FECollection(order, dim);
    mfem::FiniteElementSpace ND(&mesh, fec_ND);
    mfem::FiniteElementSpace CG(&mesh, fec_CG);

    int num_it = 0;

    std::cout << "Second-order semi-Lagrangian Navier-Stokes (Eq. 34)\n"
              << "  dim=" << dim << ", order=" << order
              << ", ND DOFs=" << ND.GetNDofs()
              << ", CG DOFs=" << CG.GetNDofs()
              << ", elements=" << mesh.GetNE()
              << ", dt=" << dt << ", T=" << T
              << ", viscosity=" << viscosity
              << ", trace_order=" << trace_order << std::endl;

    // ---- Boundary attribute marker (optional) ----
    mfem::Array<int> lid_marker;
    const mfem::Array<int> *lid_marker_ptr = nullptr;
    if (config.has_lid_attributes())
    {
        lid_marker = config.get_lid_marker(mesh.bdr_attributes.Max());
        lid_marker_ptr = &lid_marker;
    }

    // ---- Assemble Stokes system ----
    // BDF1 bootstrap: mass coefficient = 1/dt
    // BDF2 steps: mass coefficient = 3/(2*dt)
    // We start with BDF1 and reassemble for BDF2 after the first step.
    hcurl::StokesSystem sys_bdf1(ND, CG, 1.0 / dt, viscosity, theta, Cw,
                                  /*sigma=*/0.0, /*gamma=*/0.0);
    hcurl::StokesSystem sys_bdf2(ND, CG, 1.5 / dt, viscosity, theta, Cw,
                                  /*sigma=*/0.0, /*gamma=*/0.0);

    hcurl::StokesRHS rhs(ND, CG,
                         config.get_exact_data("force_data"),
                         config.get_exact_data("boundary_data_u"),
                         theta, Cw, viscosity, 0.0, lid_marker_ptr);
    hcurl::StokesSolution x(ND, CG);

    // ---- Initial condition ----
    mfem::VectorFunctionCoefficient u_init(dim, config.get_exact_data("initial_data_u"));
    u_init.SetTime(0.);
    x.get_u().ProjectCoefficient(u_init);

    // ---- Semi-Lagrangian advection operator (order 2) ----
    SemiLagrangianAdvection1FormOrder2<2> advection(ND);

    // Velocity history for BDF2: need ω^{n-1} and ω^{n-2}
    mfem::GridFunction omega_nm1(&ND);  // ω^{n-1}
    mfem::GridFunction omega_nm2(&ND);  // ω^{n-2}
    mfem::GridFunction omega_tilde_1(&ND);
    mfem::GridFunction omega_tilde_2(&ND);

    // Save ω⁰ so BDF2 can use it as ω^{n-2} in the second timestep
    omega_nm1 = x.get_u();

    // Direct grid function evaluation for Heun corrector step.
    // Dihedral averaging at arrival points is handled internally by
    // the advection operator; this callback is only used at predicted
    // departure points (typically interior to an element).
    auto velocity_func = [&mesh, &x](
        const mfem::Vector &pt, double t, int start_elem_hint, mfem::Vector &v) {
        const int dim = mesh.SpaceDimension();
        v.SetSize(dim);
        // Find the element containing pt via BFS from hint
        mfem::IntegrationPoint ip;
        int elem = FindElementBFS(mesh, start_elem_hint, pt, ip);
        if (elem < 0 && start_elem_hint != 0)
        {
            elem = FindElementBFS(mesh, 0, pt, ip);
        }
        if (elem < 0)
        {
            v = 0.0;
            return;
        }
        // Use a local IsoparametricTransformation for thread safety
        // (mesh.GetElementTransformation(elem) returns a shared pointer).
        mfem::IsoparametricTransformation eltrans;
        mesh.GetElementTransformation(elem, &eltrans);
        eltrans.SetIntPoint(&ip);
        x.get_u().GetVectorValue(eltrans, ip, v);
    };

    // Boundary function
    auto bdr_data = config.get_exact_data("boundary_data_u");
    SemiLagrangianAdvection1FormOrder2<2>::BoundaryFunc boundary_func =
        [&bdr_data, lid_marker_ptr, dim](
            const mfem::Vector &pt, double t, int bdr_attr, mfem::Vector &v) {
        v.SetSize(dim);
        if (!lid_marker_ptr)
        {
            bdr_data(pt, t, v);
            return;
        }
        if (bdr_attr > 0 && bdr_attr <= lid_marker_ptr->Size()
            && (*lid_marker_ptr)[bdr_attr - 1] == 1)
        {
            bdr_data(pt, t, v);
        }
        else
        {
            v = 0.0;
        }
    };

    // ---- Solvers ----
    mfem::MINRESSolver solv_bdf1;
    solv_bdf1.iterative_mode = true;
    solv_bdf1.SetOperator(sys_bdf1);
    solv_bdf1.SetRelTol(tol);
    solv_bdf1.SetAbsTol(1e-10);
    solv_bdf1.SetMaxIter(100000);
    solv_bdf1.SetPrintLevel(0);

    mfem::MINRESSolver solv_bdf2;
    solv_bdf2.iterative_mode = true;
    solv_bdf2.SetOperator(sys_bdf2);
    solv_bdf2.SetRelTol(tol);
    solv_bdf2.SetAbsTol(1e-10);
    solv_bdf2.SetMaxIter(100000);
    solv_bdf2.SetPrintLevel(0);

    // ---- ParaView output ----
    mfem::ParaViewDataCollection vtk_dc("./out/paraview/" + output_file, &mesh);
    if (visualisation > 0)
    {
        vtk_dc.RegisterField("u", &x.get_u());
        vtk_dc.RegisterField("p", &x.get_p());
        vtk_dc.SetCycle(0);
        vtk_dc.SetTime(0.0);
        vtk_dc.Save();
    }

    // ---- CSV logger ----
    double t = 0.;
    int cycle = 0;
    double advect_time_s = 0.0;
    double solve_time_s = 0.0;

    SingleFieldCSVLogger csv(config, cycle, t, &ND, x.get_u(), num_it,
                             &advect_time_s, &solve_time_s);

    // ---- Progress bar ----
    int total_cycles = 0;
    for (double tt = dt; tt < T + tol; tt += dt) { total_cycles++; }

    auto run_start = std::chrono::steady_clock::now();
    auto print_progress = [&](int completed_cycles, double t_now)
    {
        if (total_cycles <= 0) { return; }

        const int width = 32;
        const double frac_raw = static_cast<double>(completed_cycles) / static_cast<double>(total_cycles);
        const double frac = std::clamp(frac_raw, 0.0, 1.0);
        const int filled = static_cast<int>(std::round(frac * width));
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_s = std::chrono::duration<double>(now - run_start).count();
        const double eta_s = (completed_cycles > 0)
                                 ? elapsed_s * (static_cast<double>(total_cycles - completed_cycles) /
                                                static_cast<double>(completed_cycles))
                                 : std::numeric_limits<double>::infinity();

        std::ostringstream line;
        line << "\r[";
        for (int i = 0; i < width; i++)
        {
            line << (i < filled ? '=' : ' ');
        }
        line << "] " << std::setw(3) << static_cast<int>(std::round(frac * 100.0)) << "% "
             << completed_cycles << "/" << total_cycles
             << "  t=" << std::fixed << std::setprecision(3) << t_now << "/" << T
             << "  ETA " << FormatDuration(eta_s);

        std::cout << line.str() << std::flush;
        if (completed_cycles >= total_cycles)
        {
            std::cout << std::endl;
        }
    };

    print_progress(0, 0.0);

    // ---- Time loop ----
    for (t = dt, cycle = 1; t < T + tol; t += dt, cycle++)
    {
        auto advect_start = std::chrono::steady_clock::now();

        if (cycle == 1)
        {
            // ---- BDF1 bootstrap step ----
            // Only ω⁰ is available. Apply BDF1:
            //   ω̃ = I_{h,2}(X̄*_{τ} ω⁰)
            //   [M/dt + εK, B; B^T, 0] [ω¹; p¹] = [(1/dt)M·ω̃ + f; 0]
            advection.Apply(velocity_func, boundary_func, t, dt,
                            x.get_u(), omega_tilde_1, trace_order,
                            &x.get_u());

            advect_time_s = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - advect_start)
                                .count();

            rhs.Update(omega_tilde_1, t, 1.0 / dt);

            auto solve_start = std::chrono::steady_clock::now();
            solv_bdf1.Mult(rhs, x);
            solve_time_s = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - solve_start)
                               .count();
            num_it = solv_bdf1.GetNumIterations();

            // Shift history: omega_nm1 was set to ω⁰ before the loop.
            // After solving, x contains ω¹.
            // For cycle 2 (BDF2): omega_nm1 = ω¹, omega_nm2 = ω⁰
            omega_nm2 = omega_nm1;  // ω⁰
            omega_nm1 = x.get_u();  // ω¹
        }
        else
        {
            // ---- BDF2 step ----
            // ω̃₁ = I_{h,2}(X̄*_{τ} ω^{n-1})
            // ω̃₂ = I_{h,2}(X̄*_{2τ} ω^{n-2})
            // [3M/(2dt) + εK, B; B^T, 0] [ωⁿ; pⁿ]
            //   = [(4/(2dt))M·ω̃₁ - (1/(2dt))M·ω̃₂ + f; 0]
            advection.ApplyBDF2(velocity_func, boundary_func, t, dt,
                                omega_nm1, omega_nm2,
                                omega_tilde_1, omega_tilde_2, trace_order,
                                &omega_nm1);

            advect_time_s = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - advect_start)
                                .count();

            rhs.UpdateBDF2(omega_tilde_1, omega_tilde_2, t, dt);

            auto solve_start = std::chrono::steady_clock::now();
            solv_bdf2.Mult(rhs, x);
            solve_time_s = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - solve_start)
                               .count();
            num_it = solv_bdf2.GetNumIterations();

            // Shift history: ω^{n-2} ← ω^{n-1}, ω^{n-1} ← ωⁿ
            omega_nm2 = omega_nm1;
            omega_nm1 = x.get_u();
        }

        // Log + progress
        csv.WriteRow();
        print_progress(cycle, t);

        // Visualize
        if (visualisation > 0 && cycle % visualisation == 0)
        {
            vtk_dc.SetCycle(cycle);
            vtk_dc.SetTime(t);
            vtk_dc.Save();
        }
    }

    delete fec_ND;
    delete fec_CG;

    return 0;
}
