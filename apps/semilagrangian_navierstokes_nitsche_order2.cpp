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
#include "StokesMG.h"
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

    // Velocity evaluation mode for characteristic tracing:
    //   "none"           — raw ND₂ GridFunction evaluation (no averaging)
    //   "dihedral"       — dihedral-angle-weighted averaging at arrival points
    //   "cg_projection"  — L²-project ND₂ → CG² for a globally continuous velocity
    std::string velocity_mode =
        config.get_value<std::string>("velocity_mode", "dihedral");
    for (char &ch : velocity_mode)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    const bool use_dihedral = (velocity_mode == "dihedral");
    const bool use_cg_proj  = (velocity_mode == "cg_projection" ||
                               velocity_mode == "cg_proj" ||
                               velocity_mode == "cg");

    // ---- Mesh, geometric+p multigrid hierarchy, FE spaces (order 2) ----
    // Mirror the order-1 app: the actual solved operator is the MG finest
    // StokesNitscheOperator, preconditioned by the MG cycle.  The hierarchy
    // is: base mesh (order 1) -> h-refined `refinements` times (order 1)
    // -> one p-refinement to order 2 (the fine level matching ND2).
    auto base_mesh = std::make_shared<mfem::Mesh>(mesh_string.c_str(), 1, 1);
    StokesNitsche::StokesMG mg_solver(base_mesh, 1.0 / (dt * viscosity),
                                      theta, Cw);
    mg_solver.setOperatorMode(StokesNitsche::OperatorMode::Galerkin);
    mg_solver.setIterativeMode(false);
    mg_solver.setCycleType(StokesNitsche::MGCycleType::VCycle);
    mg_solver.setSmoothIterations(1, 1);
    for (int l = 0; l < refinements; l++) { mg_solver.addRefinement(); }
    mg_solver.addRefinement(1);  // p-refine: order 1 -> order 2 (fine level)
    StokesNitsche::StokesNitscheOperator& op =
        *const_cast<StokesNitsche::StokesNitscheOperator*>(
            &mg_solver.getFinestOperator());
    op.setOperatorMode(StokesNitsche::OperatorMode::Galerkin);
    mfem::Mesh &mesh = op.getMesh();
    int dim = mesh.Dimension();

    const int order = 2;
    mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order, dim);
    mfem::FiniteElementCollection *fec_CG = new mfem::H1_FECollection(order, dim);
    mfem::FiniteElementSpace ND(&mesh, fec_ND);
    mfem::FiniteElementSpace CG(&mesh, fec_CG);

    int num_it = 0;

    // ---- CG projection velocity field (optional) ----
    // Vector-valued H¹ space for globally continuous velocity.
    mfem::FiniteElementCollection *fec_VCG = nullptr;
    mfem::FiniteElementSpace *VCG = nullptr;
    mfem::GridFunction *u_cg = nullptr;
    mfem::BilinearForm *mass_cg = nullptr;
    mfem::CGSolver *cg_solver = nullptr;
    if (use_cg_proj)
    {
        fec_VCG = new mfem::H1_FECollection(order, dim);
        VCG = new mfem::FiniteElementSpace(&mesh, fec_VCG, dim);
        u_cg = new mfem::GridFunction(VCG);
        *u_cg = 0.0;

        mass_cg = new mfem::BilinearForm(VCG);
        mass_cg->AddDomainIntegrator(new mfem::VectorMassIntegrator);
        mass_cg->Assemble();
        mass_cg->Finalize();

        cg_solver = new mfem::CGSolver;
        cg_solver->SetOperator(mass_cg->SpMat());
        cg_solver->SetRelTol(1e-12);
        cg_solver->SetAbsTol(1e-15);
        cg_solver->SetMaxIter(200);
        cg_solver->SetPrintLevel(0);
    }

    // Helper: L²-project an ND₂ GridFunction onto the vector CG space.
    auto project_to_cg = [&](const mfem::GridFunction &u_nd) {
        if (!use_cg_proj) { return; }
        // RHS = (u_nd, φ_CG)_{L²}
        mfem::LinearForm rhs_cg(VCG);
        mfem::VectorGridFunctionCoefficient u_coeff(
            const_cast<mfem::GridFunction *>(&u_nd));
        rhs_cg.AddDomainIntegrator(
            new mfem::VectorDomainLFIntegrator(u_coeff));
        rhs_cg.Assemble();
        cg_solver->Mult(rhs_cg, *u_cg);
    };

    std::cout << "Second-order semi-Lagrangian Navier-Stokes (Eq. 34)\n"
              << "  dim=" << dim << ", order=" << order
              << ", ND DOFs=" << ND.GetNDofs()
              << ", CG DOFs=" << CG.GetNDofs()
              << ", elements=" << mesh.GetNE()
              << ", dt=" << dt << ", T=" << T
              << ", viscosity=" << viscosity
              << ", trace_order=" << trace_order
              << ", velocity_mode=" << velocity_mode << std::endl;

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
    // The solved operator is the MG finest StokesNitscheOperator `op`.
    // BDF1 vs BDF2 differ only in the mass coefficient (1/dt vs 1.5/dt),
    // applied to `op` via mg_solver.setTau() in the time loop.
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

    // Velocity callback for characteristic tracing (Heun corrector).
    // In CG projection mode, evaluates the smooth CG field instead of ND₂.
    auto velocity_func = [&mesh, &x, &u_cg, use_cg_proj](
        const mfem::Vector &pt, double t, int start_elem_hint, mfem::Vector &v) {
        const int dim = mesh.SpaceDimension();
        v.SetSize(dim);
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
        mfem::IsoparametricTransformation eltrans;
        mesh.GetElementTransformation(elem, &eltrans);
        eltrans.SetIntPoint(&ip);
        if (use_cg_proj)
        {
            u_cg->GetVectorValue(eltrans, ip, v);
        }
        else
        {
            x.get_u().GetVectorValue(eltrans, ip, v);
        }
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

    // ---- Solver: FGMRES preconditioned by the geometric+p multigrid ----
    // (mirrors the order-1 app; replaces unpreconditioned MINRES which
    //  needed thousands of iterations at order 2).
    mfem::FGMRESSolver gmres;
    gmres.SetAbsTol(1e-12);
    gmres.SetRelTol(1e-6);
    gmres.SetMaxIter(500);
    gmres.SetPrintLevel(0);
    gmres.SetOperator(op);
    gmres.SetPreconditioner(mg_solver);
    gmres.SetKDim(128);
    // Track the MG mass coefficient (tau) so we only re-tau on the BDF1->BDF2
    // transition (setTau rebuilds smoothers + coarse factorization).
    const double tau_bdf1 = 1.0 / (dt * viscosity);
    const double tau_bdf2 = 1.5 / (dt * viscosity);
    double mg_tau_current = -1.0;

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
            if (use_cg_proj) { project_to_cg(x.get_u()); }

            // velocity_gf controls dihedral averaging at arrival points:
            //   dihedral mode  → &x.get_u() (ND₂ field for dihedral weights)
            //   cg_projection  → nullptr (CG callback is already continuous)
            //   none           → nullptr
            mfem::GridFunction *vel_gf =
                use_dihedral ? &x.get_u() : nullptr;

            advection.Apply(velocity_func, boundary_func, t, dt,
                            x.get_u(), omega_tilde_1, trace_order,
                            vel_gf);

            advect_time_s = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - advect_start)
                                .count();

            rhs.Update(omega_tilde_1, t, 1.0 / dt);
            // DEC scaling: StokesNitscheOperator solves the viscosity-scaled
            // system (same convention as the order-1 app).
            rhs.GetBlock(0) *= 1.0 / viscosity;

            if (mg_tau_current != tau_bdf1)
            {
                mg_solver.setTau(tau_bdf1);
                mg_tau_current = tau_bdf1;
            }

            auto solve_start = std::chrono::steady_clock::now();
            gmres.Mult(rhs, x);
            op.eliminateConstants(x);
            solve_time_s = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - solve_start)
                               .count();
            num_it = gmres.GetNumIterations();

            // Shift history: omega_nm1 was set to ω⁰ before the loop.
            // After solving, x contains ω¹.
            // For cycle 2 (BDF2): omega_nm1 = ω¹, omega_nm2 = ω⁰
            omega_nm2 = omega_nm1;  // ω⁰
            omega_nm1 = x.get_u();  // ω¹
        }
        else
        {
            // ---- BDF2 step ----
            if (use_cg_proj) { project_to_cg(omega_nm1); }

            mfem::GridFunction *vel_gf =
                use_dihedral ? &omega_nm1 : nullptr;

            advection.ApplyBDF2(velocity_func, boundary_func, t, dt,
                                omega_nm1, omega_nm2,
                                omega_tilde_1, omega_tilde_2, trace_order,
                                vel_gf);

            advect_time_s = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - advect_start)
                                .count();

            rhs.UpdateBDF2(omega_tilde_1, omega_tilde_2, t, dt);
            rhs.GetBlock(0) *= 1.0 / viscosity;

            if (mg_tau_current != tau_bdf2)
            {
                mg_solver.setTau(tau_bdf2);
                mg_tau_current = tau_bdf2;
            }

            auto solve_start = std::chrono::steady_clock::now();
            gmres.Mult(rhs, x);
            op.eliminateConstants(x);
            solve_time_s = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - solve_start)
                               .count();
            num_it = gmres.GetNumIterations();

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
    delete cg_solver;
    delete mass_cg;
    delete u_cg;
    delete VCG;
    delete fec_VCG;

    return 0;
}
