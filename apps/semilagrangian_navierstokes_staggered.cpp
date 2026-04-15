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
#include "SemiLagrangianAdvection.h"
#include "StokesMG.h"

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

// Staggered semi-Lagrangian Navier–Stokes driver.
//
// Maintains two solutions on offset timelines:
//   u_A at integer times     t_n     = n·dt
//   u_B at half-integer times t_{n-½} = (n-½)·dt
//
// At the start of cycle m (m=1,2,...): t_A = t_{m-1}, t_B = t_{m-½}.
// Cycle m produces u_A at t_m and u_B at t_{m+½}.
//
// Update ordering per cycle (order matters — each step's velocity is the
// time-midpoint of its own advection interval):
//   1. advance u_A: t_{m-1} → t_m     using current u_B (at t_{m-½} = midpoint)
//   2. advance u_B: t_{m-½} → t_{m+½} using freshly-updated u_A (at t_m = midpoint)
//
// A single-evaluation Euler trace becomes time-centered ≈ midpoint rule
// ≈ 2nd-order in time, at ~Euler cost.
//
// Note on vertex_velocity_mode: must be kSingleElement here. The weighted
// modes (kVertexSolidAngle, kEdgeDihedral) precompute vertex velocity from
// gf_old directly (SemiLagrangianAdvection.h:1483, 1550), which in this app
// is the field being advected (u_A or u_B itself), not the velocity supplied
// via the callback. Only kSingleElement routes velocity through the callback.
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

    // ---- Configuration ----
    DualFieldConfig config(config_path);

    double viscosity = config.get_viscosity();
    int refinements = config.get_refinements();
    int order = config.get_order();
    int visualisation = config.get_visualisation();
    int printlevel = config.get_printlevel();
    double tol = config.get_tol();
    std::string mesh_string = config.get_mesh();
    std::string output_file = config.get_outputfile();
    double theta = 1.;
    double Cw = 40.;
    double dt = config.get_dt();
    double T = config.get_T();
    int trace_order = config.get_value<int>("trace_order", 1);
    std::string vertex_velocity_mode_name =
        config.get_value<std::string>("vertex_velocity_mode", "");
    for (char &ch : vertex_velocity_mode_name)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    // Staggered scheme requires kSingleElement (see file-level note).
    const int vertex_velocity_mode =
        SemiLagrangianAdvection1Form<1>::kSingleElement;
    if (!vertex_velocity_mode_name.empty() &&
        vertex_velocity_mode_name != "single" &&
        vertex_velocity_mode_name != "single_element" &&
        vertex_velocity_mode_name != "random" &&
        vertex_velocity_mode_name != "legacy")
    {
        std::cerr << "[warn] staggered scheme forces vertex_velocity_mode="
                  << "'single_element' (requested '" << vertex_velocity_mode_name
                  << "' ignored). Weighted modes use gf_old for vertex velocity, "
                     "which is inconsistent with the cross-field tracing used here."
                  << std::endl;
    }

    if (trace_order < 1 || trace_order > 2)
    {
        std::cerr << "[warn] staggered scheme supports trace_order in [1,2]; "
                  << "requested " << trace_order
                  << " clamped to 1 (Euler — the staggering provides the time-centering)."
                  << std::endl;
        trace_order = 1;
    }

    // ---- Mesh and FE spaces ----
    auto mesh_ptr = std::make_shared<mfem::Mesh>(mesh_string.c_str(), 1, 1);
    StokesNitsche::StokesMG mg_solver(mesh_ptr, 1./(dt*viscosity),
        theta, Cw);
    mg_solver.setOperatorMode(StokesNitsche::OperatorMode::Galerkin);
    mg_solver.setIterativeMode(false);
    mg_solver.setCycleType(StokesNitsche::MGCycleType::VCycle);
    mg_solver.setSmoothIterations(1, 1);
    for (int l = 0; l < refinements; l++)
    {
        mg_solver.addRefinement();
    }
    StokesNitsche::StokesNitscheOperator& op =
            *const_cast<StokesNitsche::StokesNitscheOperator*>(&mg_solver.getFinestOperator());
    op.setOperatorMode(StokesNitsche::OperatorMode::Galerkin);
    auto fine_mesh_ptr = op.getMeshPtr();
    int dim = fine_mesh_ptr->Dimension();

    mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order, dim);
    mfem::FiniteElementCollection *fec_CG = new mfem::H1_FECollection(order, dim);
    mfem::FiniteElementSpace ND(&op.getMesh(), fec_ND);
    mfem::FiniteElementSpace CG(&op.getMesh(), fec_CG);

    int num_it = 0;

    // ---- Boundary attribute marker (optional) ----
    mfem::Array<int> lid_marker;
    const mfem::Array<int> *lid_marker_ptr = nullptr;
    if (config.has_lid_attributes())
    {
        lid_marker = config.get_lid_marker(fine_mesh_ptr->bdr_attributes.Max());
        lid_marker_ptr = &lid_marker;
    }

    // ---- Assemble system (constant — no Eulerian convection) ----
    // Same matrix for u_A and u_B updates; dt is shared.
    hcurl::StokesSystem sys(ND, CG, 1./dt, viscosity, theta, Cw,
                            /*sigma=*/0.0, /*gamma=*/0.0);
    hcurl::StokesRHS rhs(ND, CG,
                         config.get_exact_data("force_data"),
                         config.get_exact_data("boundary_data_u"),
                         theta, Cw, viscosity, 0.0, lid_marker_ptr);
    hcurl::StokesSolution x_A(ND, CG);
    hcurl::StokesSolution x_B(ND, CG);

    // ---- Initial condition ----
    // u_A lives at integer times (init at t=0); u_B lives at half-integer
    // times (init at t=dt/2).
    mfem::VectorFunctionCoefficient u_init(dim, config.get_exact_data("initial_data_u"));
    u_init.SetTime(0.);
    x_A.get_u().ProjectCoefficient(u_init);
    u_init.SetTime(0.5 * dt);
    x_B.get_u().ProjectCoefficient(u_init);

    // ---- Semi-Lagrangian advection operator ----
    SemiLagrangianAdvection1Form<1> advection(ND);
    mfem::GridFunction omega_tilde_A(&ND);
    mfem::GridFunction omega_tilde_B(&ND);

    // Velocity callback closing over u_B's velocity GridFunction (used when
    // advancing u_A; u_B is at the midpoint in time of u_A's interval).
    auto velocity_from_B = [fine_mesh_ptr, &u_gf = x_B.get_u(), dim](
        const mfem::Vector &pt, double, int start_elem_hint, mfem::Vector &v) {
        v.SetSize(dim);
        mfem::IntegrationPoint ip;
        int elem = FindElementBFS(*fine_mesh_ptr, start_elem_hint, pt, ip);
        if (elem < 0 && start_elem_hint != 0)
        {
            elem = FindElementBFS(*fine_mesh_ptr, 0, pt, ip);
        }
        if (elem >= 0)
        {
            mfem::IsoparametricTransformation eltrans;
            fine_mesh_ptr->GetElementTransformation(elem, &eltrans);
            eltrans.SetIntPoint(&ip);
            u_gf.GetVectorValue(eltrans, ip, v);
        }
        else
        {
            v = 0.0;
        }
    };

    // Velocity callback closing over u_A (used when advancing u_B, AFTER
    // u_A has been advanced this cycle: u_A is at t_m = midpoint of u_B's
    // [t_{m-½}, t_{m+½}] interval).
    auto velocity_from_A = [fine_mesh_ptr, &u_gf = x_A.get_u(), dim](
        const mfem::Vector &pt, double, int start_elem_hint, mfem::Vector &v) {
        v.SetSize(dim);
        mfem::IntegrationPoint ip;
        int elem = FindElementBFS(*fine_mesh_ptr, start_elem_hint, pt, ip);
        if (elem < 0 && start_elem_hint != 0)
        {
            elem = FindElementBFS(*fine_mesh_ptr, 0, pt, ip);
        }
        if (elem >= 0)
        {
            mfem::IsoparametricTransformation eltrans;
            fine_mesh_ptr->GetElementTransformation(elem, &eltrans);
            eltrans.SetIntPoint(&ip);
            u_gf.GetVectorValue(eltrans, ip, v);
        }
        else
        {
            v = 0.0;
        }
    };

    // Boundary function: dispatch based on boundary attribute.
    auto bdr_data = config.get_exact_data("boundary_data_u");
    SemiLagrangianAdvection1Form<1>::BoundaryFunc boundary_func =
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

    // ---- Solver ----
    mfem::FGMRESSolver gmres;
    gmres.SetAbsTol(1e-12);
    gmres.SetRelTol(1e-6);
    gmres.SetMaxIter(500);
    gmres.SetPrintLevel(printlevel > 1 ? 1 : 0);
    gmres.SetOperator(op);
    gmres.SetPreconditioner(mg_solver);
    gmres.SetKDim(128);

    // ---- ParaView output ----
    // One collection with both solutions. Saved twice per cycle (at t_B and t_A).
    mfem::ParaViewDataCollection vtk_dc("./out/paraview/" + output_file, fine_mesh_ptr.get());
    if (visualisation > 0)
    {
        vtk_dc.RegisterField("u_A", &x_A.get_u());
        vtk_dc.RegisterField("p_A", &x_A.get_p());
        vtk_dc.RegisterField("u_B", &x_B.get_u());
        vtk_dc.RegisterField("p_B", &x_B.get_p());
        vtk_dc.SetCycle(0);
        vtk_dc.SetTime(0.0);
        vtk_dc.Save();
    }

    // ---- CSV logger (on u_A — the integer-time, analytically-referenceable solution) ----
    double t_A = 0.;
    double t_B = 0.5 * dt;
    int cycle = 0;
    double advect_time_s = 0.0;
    double solve_time_s = 0.0;
    double trace_time_s = 0.0;
    double split_time_s = 0.0;
    double interior_integral_time_s = 0.0;
    double boundary_integral_time_s = 0.0;
    long long split_calls = 0;
    long long total_segments = 0;
    double edge_thread_min_s = 0.0;
    double edge_thread_avg_s = 0.0;
    double edge_thread_max_s = 0.0;
    double edge_thread_imbalance = 0.0;
    int edge_threads_active = 0;
    long long edge_thread_edges_min = 0;
    long long edge_thread_edges_max = 0;
    double edge_thread_cpu_min_s = 0.0;
    double edge_thread_cpu_avg_s = 0.0;
    double edge_thread_cpu_max_s = 0.0;
    double edge_thread_cpu_util_min = 0.0;
    double edge_thread_cpu_util_avg = 0.0;
    double edge_thread_cpu_util_max = 0.0;
    const bool profile_advection_breakdown =
        config.get_value<bool>("profile_advection_breakdown", false);
    const bool profile_advection_thread_balance =
        config.get_value<bool>("profile_advection_thread_balance", false);

    SemiLagrangianStepStats stats_A;
    SemiLagrangianStepStats stats_B;
    SingleFieldCSVLogger csv(config, cycle, t_A, &ND, x_A.get_u(), num_it,
                             &advect_time_s, &solve_time_s,
                             profile_advection_breakdown ? &trace_time_s : nullptr,
                             profile_advection_breakdown ? &split_time_s : nullptr,
                             profile_advection_breakdown ? &interior_integral_time_s : nullptr,
                             profile_advection_breakdown ? &boundary_integral_time_s : nullptr,
                             profile_advection_breakdown ? &split_calls : nullptr,
                             profile_advection_breakdown ? &total_segments : nullptr,
                             profile_advection_thread_balance ? &edge_thread_min_s : nullptr,
                             profile_advection_thread_balance ? &edge_thread_avg_s : nullptr,
                             profile_advection_thread_balance ? &edge_thread_max_s : nullptr,
                             profile_advection_thread_balance ? &edge_thread_imbalance : nullptr,
                             profile_advection_thread_balance ? &edge_threads_active : nullptr,
                             profile_advection_thread_balance ? &edge_thread_edges_min : nullptr,
                             profile_advection_thread_balance ? &edge_thread_edges_max : nullptr,
                             profile_advection_thread_balance ? &edge_thread_cpu_min_s : nullptr,
                             profile_advection_thread_balance ? &edge_thread_cpu_avg_s : nullptr,
                             profile_advection_thread_balance ? &edge_thread_cpu_max_s : nullptr,
                             profile_advection_thread_balance ? &edge_thread_cpu_util_min : nullptr,
                             profile_advection_thread_balance ? &edge_thread_cpu_util_avg : nullptr,
                             profile_advection_thread_balance ? &edge_thread_cpu_util_max : nullptr);

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
    for (cycle = 1; t_A + dt < T + tol; ++cycle)
    {
        stats_A.enable_breakdown = profile_advection_breakdown;
        stats_A.enable_thread_balance = profile_advection_thread_balance;
        stats_B.enable_breakdown = profile_advection_breakdown;
        stats_B.enable_thread_balance = profile_advection_thread_balance;

        advect_time_s = 0.0;
        solve_time_s = 0.0;

        // --- Advance u_A: t_{m-1} → t_m using current u_B (at t_{m-½} = midpoint) ---
        double t_A_new = t_A + dt;
        auto advect_A_start = std::chrono::steady_clock::now();
        advection.Apply(velocity_from_B, boundary_func, t_A_new, dt,
                        x_A.get_u(), omega_tilde_A, trace_order,
                        (profile_advection_breakdown || profile_advection_thread_balance)
                            ? &stats_A
                            : nullptr,
                        /*velocity_prev=*/nullptr,
                        /*settls_iterations=*/1,
                        vertex_velocity_mode);
        advect_time_s += std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - advect_A_start)
                             .count();

        // --- Solve for u_A^m. x_A is updated in-place BEFORE u_B's advect
        //     so velocity_from_A samples the time-midpoint of u_B's interval. ---
        auto solve_A_start = std::chrono::steady_clock::now();
        rhs.Update(omega_tilde_A, t_A_new, 1./dt);
        rhs.GetBlock(0) *= 1.0/viscosity;
        gmres.Mult(rhs, x_A);
        op.eliminateConstants(x_A);
        t_A = t_A_new;
        solve_time_s += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - solve_A_start)
                            .count();

        // --- Advance u_B: t_{m-½} → t_{m+½} using freshly-updated u_A (at t_m = midpoint) ---
        double t_B_new = t_B + dt;
        auto advect_B_start = std::chrono::steady_clock::now();
        advection.Apply(velocity_from_A, boundary_func, t_B_new, dt,
                        x_B.get_u(), omega_tilde_B, trace_order,
                        (profile_advection_breakdown || profile_advection_thread_balance)
                            ? &stats_B
                            : nullptr,
                        /*velocity_prev=*/nullptr,
                        /*settls_iterations=*/1,
                        vertex_velocity_mode);
        advect_time_s += std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - advect_B_start)
                             .count();

        // --- Solve for u_B^{m+½} ---
        auto solve_B_start = std::chrono::steady_clock::now();
        rhs.Update(omega_tilde_B, t_B_new, 1./dt);
        rhs.GetBlock(0) *= 1.0/viscosity;
        gmres.Mult(rhs, x_B);
        op.eliminateConstants(x_B);
        t_B = t_B_new;
        solve_time_s += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - solve_B_start)
                            .count();

        if (profile_advection_breakdown)
        {
            trace_time_s = stats_A.trace_departure_s + stats_B.trace_departure_s;
            split_time_s = stats_A.split_line_s + stats_B.split_line_s;
            interior_integral_time_s =
                stats_A.interior_integral_s + stats_B.interior_integral_s;
            boundary_integral_time_s =
                stats_A.boundary_integral_s + stats_B.boundary_integral_s;
            split_calls = stats_A.split_calls + stats_B.split_calls;
            total_segments = stats_A.total_segments + stats_B.total_segments;
        }
        if (profile_advection_thread_balance)
        {
            // Report the more loaded of the two advections.
            edge_thread_min_s = std::min(stats_A.edge_thread_min_s, stats_B.edge_thread_min_s);
            edge_thread_avg_s = 0.5 * (stats_A.edge_thread_avg_s + stats_B.edge_thread_avg_s);
            edge_thread_max_s = std::max(stats_A.edge_thread_max_s, stats_B.edge_thread_max_s);
            edge_thread_imbalance =
                std::max(stats_A.edge_thread_imbalance, stats_B.edge_thread_imbalance);
            edge_threads_active = std::max(stats_A.edge_threads_active, stats_B.edge_threads_active);
            edge_thread_edges_min =
                std::min(stats_A.edge_thread_edges_min, stats_B.edge_thread_edges_min);
            edge_thread_edges_max =
                std::max(stats_A.edge_thread_edges_max, stats_B.edge_thread_edges_max);
            edge_thread_cpu_min_s =
                std::min(stats_A.edge_thread_cpu_min_s, stats_B.edge_thread_cpu_min_s);
            edge_thread_cpu_avg_s =
                0.5 * (stats_A.edge_thread_cpu_avg_s + stats_B.edge_thread_cpu_avg_s);
            edge_thread_cpu_max_s =
                std::max(stats_A.edge_thread_cpu_max_s, stats_B.edge_thread_cpu_max_s);
            edge_thread_cpu_util_min =
                std::min(stats_A.edge_thread_cpu_util_min, stats_B.edge_thread_cpu_util_min);
            edge_thread_cpu_util_avg =
                0.5 * (stats_A.edge_thread_cpu_util_avg + stats_B.edge_thread_cpu_util_avg);
            edge_thread_cpu_util_max =
                std::max(stats_A.edge_thread_cpu_util_max, stats_B.edge_thread_cpu_util_max);
        }

        num_it = gmres.GetNumIterations();

        csv.WriteRow();
        print_progress(cycle, t_A);

        if (visualisation > 0 && cycle % visualisation == 0)
        {
            vtk_dc.SetCycle(2*cycle - 1);
            vtk_dc.SetTime(t_B);
            vtk_dc.Save();
            vtk_dc.SetCycle(2*cycle);
            vtk_dc.SetTime(t_A);
            vtk_dc.Save();
        }
    }

    delete fec_ND;
    delete fec_CG;

    return 0;
}
