#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>
#include <memory>
#include <boost/program_options.hpp>

#include <mpi.h>
#include "mfem.hpp"
#include "BoundaryOperators.h"
#include "io.h"
#include "StokesOperators.h"
#include "StokesMG.h"
#include "FindElementBFS.h"
#include "CylinderQoI.h"
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
    MPI_Init(&argc, &argv);
    int myid = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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

    // Consistent-Nitsche "do-nothing" pressure outflow (e.g. channel outlet).
    // When outflow_attributes is set, the marked boundary gets the outflow
    // terms (free u.n, pressure datum via the gamma penalty) instead of the
    // zero-mean pressure constraint.
    mfem::Array<int>        outflow_marker;
    const mfem::Array<int>* outflow_ptr     = nullptr;
    double                  outflow_penalty = 0.0;
    if (config.has_outflow_attributes())
    {
        outflow_marker =
            config.get_outflow_marker(base_mesh->bdr_attributes.Max());
        outflow_ptr     = &outflow_marker;
        outflow_penalty = config.get_outflow_penalty();
        std::cout << "  consistent-Nitsche outflow: gamma=" << outflow_penalty
                  << std::endl;
    }

    StokesNitsche::StokesMG mg_solver(base_mesh, 1.0 / (dt * viscosity),
                                      theta, Cw, 1.0,
                                      StokesNitsche::MassLumping::Diagonal,
                                      StokesNitsche::SmootherType::GaussSeidelForw,
                                      outflow_ptr, outflow_penalty);
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
    // Match the RHS Nitsche penalty/theta to the values the operator actually
    // uses: StokesMG rescales the penalty across the p-refinement (order 1->2),
    // so a hardcoded Cw makes the boundary imposition inconsistent and destroys
    // the order-2 convergence (see the non-MPI order-2 app for the detailed
    // rationale).
    hcurl::StokesRHS rhs(ND, CG,
                         config.get_exact_data("force_data"),
                         config.get_exact_data("boundary_data_u"),
                         op.getTheta(), op.getPenalty(), viscosity, 0.0,
                         lid_marker_ptr);
    hcurl::StokesSolution x(ND, CG);

    // ---- Initial condition ----
    mfem::VectorFunctionCoefficient u_init(dim, config.get_exact_data("initial_data_u"));
    u_init.SetTime(0.);
    x.get_u().ProjectCoefficient(u_init);

    // ---- Semi-Lagrangian advection operator (order 2) ----
    SemiLagrangianAdvection1FormOrder2<2> advection(ND);

    // ---- MPI element partition + per-ND2-DOF ownership ----
    // The advection is distributed over ELEMENTS.  Shared ND2 DOFs on element
    // boundaries are written only by their owning element (min incident
    // element id), so per-rank partial vectors are disjoint and
    // MPI_Allreduce(SUM) reproduces the canonical owner-gated result
    // bit-identically at any rank count.  dof_owner depends only on the
    // (static) mesh, so it is built once and never rebuilt on rebalance.
    const std::vector<int> dof_owner = advection.BuildDofOwnerElemMap();
    const int n_elem = mesh.GetNE();
    std::vector<int> elem_owner(n_elem);
    for (int e = 0; e < n_elem; ++e) { elem_owner[e] = e % nprocs; }
    mfem::Array<int> my_elems;
    my_elems.Reserve((n_elem + nprocs - 1) / nprocs);
    for (int e = 0; e < n_elem; ++e)
    {
        if (elem_owner[e] == myid) { my_elems.Append(e); }
    }

    // Dynamic load-balance state (deterministic on every rank, no broadcast).
    const double rebalance_threshold = 0.05;
    const double ewma_alpha = 0.3;
    const double correction_beta = 0.3;
    std::vector<double> smooth_cost(nprocs, 0.0);
    bool smooth_initialized = false;
    double advect_time_min_s = 0.0, advect_time_max_s = 0.0;

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
    if (visualisation > 0 && myid == 0)
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

    // CSV + progress are rank-0 only (non-root ranks must not touch the
    // output file or stdout progress bar).
    std::unique_ptr<SingleFieldCSVLogger> csv;
    if (myid == 0)
    {
        csv = std::make_unique<SingleFieldCSVLogger>(
            config, cycle, t, &ND, x.get_u(), num_it,
            &advect_time_s, &solve_time_s);
    }

    // ---- Flow-around-cylinder QoI (drag/lift/pressure-drop) ----
    // Enabled when the config sets qoi_cylinder_attribute > 0.  Written to a
    // dedicated CSV alongside the main logger; Strouhal is post-processed from
    // the c_L(t) time series.
    const int    qoi_cyl_attr = config.get_qoi_cylinder_attribute();
    const double qoi_Ubar     = config.get_qoi_Ubar();
    const double qoi_D        = config.get_qoi_diameter();
    const bool   do_qoi       = qoi_cyl_attr > 0;
    std::unique_ptr<std::ofstream> qoi_csv;
    if (do_qoi && myid == 0)
    {
        qoi_csv = std::make_unique<std::ofstream>(
            "./out/data/" + output_file + "_qoi.csv");
        (*qoi_csv) << "cycle,t,cD,cL,dp,FD,FL\n";
        qoi_csv->precision(10);
    }

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

    if (myid == 0) { print_progress(0, 0.0); }

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
                            vel_gf, 0, n_elem, 1, &my_elems, &dof_owner);

            advect_time_s = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - advect_start)
                                .count();

            // Disjoint owner-gated partials → SUM reconstructs the canonical
            // owner-gated field exactly (partition-invariant).
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, omega_tilde_1.GetData(),
                          ND.GetNDofs(), MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);

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
                                vel_gf, 0, n_elem, 1, &my_elems, &dof_owner);

            advect_time_s = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - advect_start)
                                .count();

            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, omega_tilde_1.GetData(),
                          ND.GetNDofs(), MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, omega_tilde_2.GetData(),
                          ND.GetNDofs(), MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);

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

        // ---- Dynamic element load balancing (deterministic, no broadcast) ----
        // Mirrors the order-1 MPI app's edge rebalance, over ELEMENTS. The
        // dof_owner map is mesh-only and never rebuilt; only which elements a
        // rank loops over changes, so correctness/partition-invariance holds.
        const double advect_local = advect_time_s;
        MPI_Allreduce(&advect_local, &advect_time_min_s, 1, MPI_DOUBLE,
                      MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&advect_local, &advect_time_max_s, 1, MPI_DOUBLE,
                      MPI_MAX, MPI_COMM_WORLD);
        advect_time_s = advect_time_max_s;

        const double advect_imbalance =
            (advect_time_max_s > 0.0)
                ? (advect_time_max_s - advect_time_min_s) / advect_time_max_s
                : 0.0;
        if (advect_imbalance > rebalance_threshold && nprocs > 1)
        {
            std::vector<double> times(nprocs);
            MPI_Allgather(&advect_local, 1, MPI_DOUBLE, times.data(), 1,
                          MPI_DOUBLE, MPI_COMM_WORLD);

            std::vector<int> counts(nprocs, 0);
            for (int e = 0; e < n_elem; ++e) { ++counts[elem_owner[e]]; }

            for (int r = 0; r < nprocs; ++r)
            {
                const double t_r = std::max(times[r], 1e-12);
                const double c_r = std::max(counts[r], 1);
                const double cost_inst = t_r / static_cast<double>(c_r);
                smooth_cost[r] = smooth_initialized
                                     ? (1.0 - ewma_alpha) * smooth_cost[r] +
                                           ewma_alpha * cost_inst
                                     : cost_inst;
            }
            smooth_initialized = true;

            double weight_sum = 0.0;
            std::vector<double> weights(nprocs, 0.0);
            for (int r = 0; r < nprocs; ++r)
            {
                weights[r] = 1.0 / std::max(smooth_cost[r], 1e-12);
                weight_sum += weights[r];
            }

            std::vector<double> share(nprocs, 0.0);
            double share_sum = 0.0;
            for (int r = 0; r < nprocs; ++r)
            {
                const double ideal = n_elem * weights[r] / weight_sum;
                share[r] = (1.0 - correction_beta) *
                               static_cast<double>(counts[r]) +
                           correction_beta * ideal;
                share_sum += share[r];
            }
            if (share_sum > 0.0)
            {
                const double norm = n_elem / share_sum;
                for (int r = 0; r < nprocs; ++r) { share[r] *= norm; }
            }

            std::vector<int> target(nprocs, 0);
            std::vector<double> remainder(nprocs, 0.0);
            int assigned = 0;
            for (int r = 0; r < nprocs; ++r)
            {
                target[r] = static_cast<int>(std::floor(share[r]));
                remainder[r] = share[r] - target[r];
                assigned += target[r];
            }
            while (assigned < n_elem)
            {
                int r_best = 0;
                for (int r = 1; r < nprocs; ++r)
                {
                    if (remainder[r] > remainder[r_best]) { r_best = r; }
                }
                ++target[r_best];
                remainder[r_best] = -1.0;
                ++assigned;
            }

            std::vector<int> assigned_count(nprocs, 0);
            for (int e = 0; e < n_elem; ++e)
            {
                int r_best = -1;
                double best_key = std::numeric_limits<double>::infinity();
                for (int r = 0; r < nprocs; ++r)
                {
                    if (target[r] <= 0) { continue; }
                    if (assigned_count[r] >= target[r]) { continue; }
                    const double key = (assigned_count[r] + 0.5) /
                                       static_cast<double>(target[r]);
                    if (key < best_key) { best_key = key; r_best = r; }
                }
                if (r_best < 0) { r_best = e % nprocs; }
                elem_owner[e] = r_best;
                ++assigned_count[r_best];
            }

            my_elems.SetSize(0);
            my_elems.Reserve(target[myid]);
            for (int e = 0; e < n_elem; ++e)
            {
                if (elem_owner[e] == myid) { my_elems.Append(e); }
            }
        }

        // Log + progress (rank 0 only)
        if (myid == 0)
        {
            csv->WriteRow();

            if (do_qoi)
            {
                CylinderForces f = ComputeCylinderForces(
                    x.get_u(), x.get_p(), qoi_cyl_attr, viscosity,
                    qoi_Ubar, qoi_D);
                double dp = CylinderPressureDrop(x.get_p());
                (*qoi_csv) << cycle << ',' << t << ',' << f.cD << ',' << f.cL
                           << ',' << dp << ',' << f.FD << ',' << f.FL << '\n';
                qoi_csv->flush();
            }

            print_progress(cycle, t);
        }

        // Visualize (rank 0 only)
        if (visualisation > 0 && myid == 0 && cycle % visualisation == 0)
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

    MPI_Finalize();
    return 0;
}
