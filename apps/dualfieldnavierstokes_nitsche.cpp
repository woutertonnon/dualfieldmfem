// dualfieldnavierstokes_nitsche.cpp
//
// Mixed dual-field Navier-Stokes: one H(curl) system + one H(div) system.
//
//   H(curl) system:  u1 in ND, p1 in CG   — Nitsche weak BCs
//   H(div)  system:  u2 in RT, w2 in ND, p2 in DG — strong normal BCs
//
// Cross-coupling through vorticities (Gauss-Seidel sweep):
//   1. Solve H(curl) using w2 = vorticity from H(div) solution (GridFunction)
//   2. Solve H(div)  using curl(u1) = vorticity from H(curl) solution
//
// Energy conservation:
//   (w2 x u1, u1) = 0  and  (curl(u1) x u2, u2) = 0
// Both nonlinear terms vanish independently (a x b . b = 0).

#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <boost/program_options.hpp>

#include "mfem.hpp"
#include "BoundaryOperators.h"
#include "io.h"
#include "StokesOperators.h"

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

    // ---- Parse command-line options ----------------------------------------
    std::string config_path;

    try
    {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("config,c",
             po::value<std::string>(&config_path)
                 ->default_value("../data/config/StokesTest/StokesTest_conv_order1_ref2.json"),
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

    std::cout << "Using config file: " << config_path << std::endl;

    DualFieldConfig config(config_path);
    const int printlevel = config.get_printlevel();
    if (config.get_value<bool>("dump_config", false))
    {
        config.PrintTree(config.get_tree());
    }

    // ---- Configuration -----------------------------------------------------
    double viscosity   = config.get_viscosity();
    int    refinements = config.get_refinements();
    int    order       = config.get_order();
    int    visualisation = config.get_visualisation();
    double tol         = config.get_tol();
    double dt          = config.get_dt();
    double T           = config.get_T();
    std::string mesh_string  = config.get_mesh();
    std::string output_file  = config.get_outputfile();
    std::string solver_type  = config.get_solver();
    bool use_hypre_pc = (solver_type == "GMRES_HYPRE" ||
                         solver_type == "GMRES_AMSADS" ||
                         solver_type == "HYPRE");

    // ---- H(curl) stabilisation parameters ----------------------------------
    //
    // Adjoint-consistency parameter for Nitsche BCs:
    //   theta = -1 gives the adjoint-consistent (incomplete) variant.
    double hcurl_theta = config.get_value<double>("hcurl_theta", -1.0);
    // Nitsche boundary penalty: (Cw / h_F) * int_{dOmega} (n x u).(n x v) dS.
    // Must be large enough to enforce BCs; 10000 is safe.
    double hcurl_Cw = config.get_value<double>("hcurl_Cw", 10000.0);
    // Normal-jump penalty: (sigma * nu / h_F) * sum_F int_F [[u]].[[v]] dF.
    // For ND the tangential component is continuous; [[u]] is purely normal.
    // Not needed when curl-jump ghost penalty is active.
    double hcurl_sigma = config.get_value<double>("hcurl_sigma", 0.0);
    // Curl-jump ghost penalty: (gamma * nu * h_F) * sum_F int_F [[curl u]].[[curl v]] dF.
    // Smooths the vorticity curl(u) used as the lagged advection coefficient
    // in the H(div) system.  h_F scaling ensures convergence is not degraded.
    double hcurl_gamma = config.get_value<double>("hcurl_gamma", 100.0);
    // Heumann upwind: upwind_scale * sum_F int_F |w.n_F| [[u]].[[v]] dF.
    // Wind-adaptive normal-jump penalty (Heumann, Hiptmair, Pagliantini 2016).
    double hcurl_upwind = config.get_value<double>("hcurl_upwind", 0.0);
    // PSPG pressure stabilisation: delta ~ h^2.
    double hcurl_delta = config.get_value<double>("hcurl_delta", 0.0);

    // ---- H(div) stabilisation parameters -----------------------------------
    //
    // Interior-face tangential-jump penalty (DG penalty):
    //   (sigma * nu / h_F) * sum_F int_F [[u]].[[v]] dF
    // For RT the normal component is continuous; [[u]] is purely tangential.
    // Controls tangential DOF oscillations that arise when viscosity enters
    // only through the vorticity coupling B^T D^{-1} B (volume-only).
    double hdiv_sigma = config.get_value<double>("hdiv_sigma", 100.0);
    // Interior-face div-jump ghost penalty:
    //   (gamma * nu * h_F) * sum_F int_F [[div u]] [[div v]] dF
    // Not needed: div(u) couples to pressure through the saddle-point
    // structure, and the convection coefficient w lives in conforming ND.
    double hdiv_gamma = config.get_value<double>("hdiv_gamma", 0.0);
    // Boundary tangential penalty (Nitsche-style):
    //   (Cw / h_F) * int_{dOmega} (n x u).(n x v) dS
    // Penalises the tangential component of u on the boundary.  For RT the
    // normal component u.n is strongly enforced via essential BCs; this term
    // weakly controls the tangential trace.  A matching RHS consistency term
    // is added so the exact solution is not penalised.
    double hdiv_Cw = config.get_value<double>("hdiv_Cw", 100.0);

    // ---- Mesh and FE spaces ------------------------------------------------
    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; l++)
        mesh.UniformRefinement();
    int dim = mesh.Dimension();

    // DG ~ L2, ND ~ H(curl), RT ~ H(div), CG ~ H1
    mfem::FiniteElementCollection *fec_DG = new mfem::L2_FECollection(order-1, dim);
    mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order, dim);
    mfem::FiniteElementCollection *fec_RT = new mfem::RT_FECollection(order-1, dim);
    mfem::FiniteElementCollection *fec_CG = new mfem::H1_FECollection(order, dim);
    mfem::FiniteElementSpace DG(&mesh, fec_DG);
    mfem::FiniteElementSpace ND(&mesh, fec_ND);
    mfem::FiniteElementSpace RT(&mesh, fec_RT);
    mfem::FiniteElementSpace CG(&mesh, fec_CG);

    int num_it_A1 = 0, num_it_A2 = 0;

    // ---- Boundary attribute marker (optional) ------------------------------
    mfem::Array<int> lid_marker;
    const mfem::Array<int> *lid_marker_ptr = nullptr;
    if (config.has_lid_attributes())
    {
        lid_marker = config.get_lid_marker(mesh.bdr_attributes.Max());
        lid_marker_ptr = &lid_marker;
        if (printlevel > 0)
        {
            std::cout << "Lid-driven cavity mode: nonzero BCs on attributes";
            for (int i = 0; i < lid_marker.Size(); i++)
                if (lid_marker[i]) std::cout << " " << i + 1;
            std::cout << std::endl;
        }
    }

    // ---- Essential DOFs for H(div) -----------------------------------------
    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1; // all boundaries essential for RT (u.n prescribed everywhere)

    mfem::Array<int> ess_tdof;
    RT.GetEssentialTrueDofs(ess_bdr, ess_tdof);

    // ---- H(curl) system: u1 in ND, p1 in CG --------------------------------
    hcurl::StokesSystem   hcurl_sys(ND, CG, 1./dt, viscosity, hcurl_theta, hcurl_Cw,
                                    hcurl_sigma, hcurl_gamma, hcurl_upwind, hcurl_delta);
    hcurl::StokesRHS      hcurl_rhs(ND, CG,
                                    config.get_exact_data("force_data"),
                                    config.get_exact_data("boundary_data_u"),
                                    hcurl_theta, hcurl_Cw, viscosity, hcurl_delta,
                                    lid_marker_ptr);
    hcurl::StokesSolution hcurl_x(ND, CG);

    // ---- H(div) system: u2 in RT, w2 in ND, p2 in DG -----------------------
    // Main-step system (full step dt): maps half-time n-1/2 -> n+1/2.
    hdiv::StokesSystem   hdiv_sys(RT, ND, DG, ess_tdof, 1./dt, viscosity,
                                  hdiv_sigma, hdiv_Cw, hdiv_gamma);
    // Start-up system (half step dt/2): maps t=0 -> t=dt/2.
    hdiv::StokesSystem   hdiv_sys_half(RT, ND, DG, ess_tdof, 2./dt, viscosity,
                                       hdiv_sigma, hdiv_Cw, hdiv_gamma);
    hdiv::StokesRHS      hdiv_rhs(RT, ND, DG, ess_tdof,
                                  config.get_exact_data("force_data"),
                                  config.get_exact_data("boundary_data_u"),
                                  viscosity, lid_marker_ptr, hdiv_Cw);
    hdiv::StokesSolution hdiv_x(RT, ND, DG);

    // ---- Initial conditions ------------------------------------------------
    mfem::VectorFunctionCoefficient u_init(3, config.get_exact_data("initial_data_u"));
    u_init.SetTime(0.);
    hcurl_x.get_u().ProjectCoefficient(u_init);
    hdiv_x.get_u().ProjectCoefficient(u_init);

    // ---- Cross-vorticity coefficients --------------------------------------
    // H(div) vorticity w2 is an explicit GridFunction in ND.
    mfem::VectorGridFunctionCoefficient w_hdiv(&hdiv_x.get_w());
    // H(curl) vorticity is curl(u1), computed on-the-fly from the ND solution.
    mfem::CurlGridFunctionCoefficient   w_hcurl(&hcurl_x.get_u());

    // ---- Time bookkeeping --------------------------------------------------
    // Staggered timeline:
    //   u1 on full times  t_full = n*dt
    //   u2,w2 on half times t_half = (n+1/2)*dt
    double t_full = 0.0;
    double t_half = 0.0;
    int    cycle  = 0;

    // ---- Visualisation -----------------------------------------------------

    mfem::ParaViewDataCollection vtk_dc(
        "./out/paraview/" + output_file, &mesh);
    if (visualisation > 0)
    {
        vtk_dc.RegisterField("u1", &hcurl_x.get_u());
        vtk_dc.RegisterField("u2", &hdiv_x.get_u());
        vtk_dc.RegisterField("w2", &hdiv_x.get_w());
        vtk_dc.RegisterField("p1", &hcurl_x.get_p());
        vtk_dc.RegisterField("p2", &hdiv_x.get_p());
        vtk_dc.SetCycle(0);
        vtk_dc.SetTime(0.0);
        vtk_dc.Save();
    }

    // ---- Solvers -----------------------------------------------------------
    std::unique_ptr<hcurl::GMRESSolver> hcurl_solv;
    std::unique_ptr<hcurl::GMRESAMSSolver> hcurl_solv_ams;
    std::unique_ptr<hdiv::GMRESSolver> hdiv_solv;
    std::unique_ptr<hdiv::GMRESADSSolver> hdiv_solv_ads;
    std::unique_ptr<hdiv::GMRESSolver> hdiv_solv_half;
    std::unique_ptr<hdiv::GMRESADSSolver> hdiv_solv_half_ads;

    if (use_hypre_pc)
    {
        hcurl_solv_ams = std::make_unique<hcurl::GMRESAMSSolver>(ND, CG, num_it_A1, viscosity, tol);
        hdiv_solv_ads = std::make_unique<hdiv::GMRESADSSolver>(RT, ND, DG, num_it_A2, 1./dt, tol);
        hdiv_solv_half_ads = std::make_unique<hdiv::GMRESADSSolver>(RT, ND, DG, num_it_A2, 2./dt, tol);
    }
    else
    {
        hcurl_solv = std::make_unique<hcurl::GMRESSolver>(ND, CG, num_it_A1, viscosity, tol);
        hdiv_solv = std::make_unique<hdiv::GMRESSolver>(RT, ND, DG, num_it_A2, 1./dt, tol);
        hdiv_solv_half = std::make_unique<hdiv::GMRESSolver>(RT, ND, DG, num_it_A2, 2./dt, tol);
    }

    // ---- CSV logging -------------------------------------------------------
    DualFieldCSVLogger csv(config, cycle, t_full, t_half, &ND, &RT,
                           hcurl_x.get_u(), hdiv_x.get_u(), hdiv_x.get_w(),
                           num_it_A1, num_it_A2);

    int total_cycles = 0;
    for (double t = dt; t < T + tol; t += dt) { total_cycles++; }

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

    // ---- Start-up half step for H(div) ------------------------------------
    // Compute u2^{1/2}, w2^{1/2} before first H(curl) solve so convection in
    // the H(curl) system does not use an uninitialized H(div) vorticity field.
    //
    // Time-centering for RHS forcing:
    //   startup step is [0, dt/2] so the midpoint is t = dt/4.
    t_half = 0.5 * dt;
    hdiv_sys_half.Update(w_hcurl);
    hdiv_rhs.Update(hdiv_x.get_u(), 0.25 * dt, 2./dt);
    if (use_hypre_pc)
    {
        hdiv_solv_half_ads->SetOperator(hdiv_sys_half);
        hdiv_solv_half_ads->Mult(hdiv_rhs, hdiv_x);
    }
    else
    {
        hdiv_solv_half->SetOperator(hdiv_sys_half);
        hdiv_solv_half->Mult(hdiv_rhs, hdiv_x);
    }

    // Log initialized staggered state: (t_full, t_half) = (0, dt/2).
    csv.WriteRow();

    // ---- Time loop ---------------------------------------------------------
    for (t_full = dt, cycle = 1; t_full < T + tol; t_full += dt, cycle++)
    {
        // -- H(curl): convection by w_hdiv (vorticity from H(div) system) ----
        hcurl_sys.Update(w_hdiv);
        // Full-step update [t_full-dt, t_full] uses midpoint forcing time.
        hcurl_rhs.Update(hcurl_x.get_u(), t_full - 0.5 * dt, 1./dt);

        if (use_hypre_pc)
        {
            hcurl_solv_ams->SetOperator(hcurl_sys);
            hcurl_solv_ams->Mult(hcurl_rhs, hcurl_x);
        }
        else
        {
            hcurl_solv->SetOperator(hcurl_sys);
            hcurl_solv->Mult(hcurl_rhs, hcurl_x);
        }
        // After this call hcurl_x (and therefore w_hcurl = curl(u1)) is at t_full.

        // -- H(div): advance to the next half time using fresh curl(u1) -------
        t_half = t_full + 0.5 * dt;
        hdiv_sys.Update(w_hcurl);
        // Half-grid update [t_half-dt, t_half] is centered at t = t_full.
        hdiv_rhs.Update(hdiv_x.get_u(), t_half - 0.5 * dt, 1./dt);
        if (use_hypre_pc)
        {
            hdiv_solv_ads->SetOperator(hdiv_sys);
            hdiv_solv_ads->Mult(hdiv_rhs, hdiv_x);
        }
        else
        {
            hdiv_solv->SetOperator(hdiv_sys);
            hdiv_solv->Mult(hdiv_rhs, hdiv_x);
        }

        // Log post-solve state for this cycle with consistent full/half times.
        csv.WriteRow();
        print_progress(cycle, t_full);

        if (visualisation > 0 && cycle % visualisation == 0)
        {
            vtk_dc.SetCycle(cycle);
            vtk_dc.SetTime(t_full);
            vtk_dc.Save();
        }
    }

    delete fec_RT;
    delete fec_ND;
    delete fec_CG;
    delete fec_DG;

    return 0;
}
