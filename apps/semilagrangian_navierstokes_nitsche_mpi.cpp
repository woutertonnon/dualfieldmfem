#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <boost/program_options.hpp>

#include <mpi.h>
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

} // anonymous namespace

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int myid, nprocs;
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
                                                                 ->default_value("../data/config/StokesTest/StokesTest_conv_order1_ref2.json"),
                                                             "path to JSON configuration file");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            if (myid == 0) std::cout << desc << "\n";
            MPI_Finalize();
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        if (myid == 0) std::cerr << "Error parsing command line: " << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }

    // ---- Configuration ----
    DualFieldConfig config(config_path);
    MPI_Barrier(MPI_COMM_WORLD); // ensure all ranks compiled config library

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
    int settls_iterations = config.get_value<int>("settls_iterations", 2);
    bool weighted_vertex_velocity_legacy =
        config.get_value<bool>("weighted_vertex_velocity", false);
    std::string vertex_velocity_mode_name =
        config.get_value<std::string>("vertex_velocity_mode", "");
    for (char &ch : vertex_velocity_mode_name)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    int vertex_velocity_mode = SemiLagrangianAdvection1Form<1>::kSingleElement;
    if (vertex_velocity_mode_name.empty())
    {
        vertex_velocity_mode = weighted_vertex_velocity_legacy
                                   ? SemiLagrangianAdvection1Form<1>::kVertexSolidAngle
                                   : SemiLagrangianAdvection1Form<1>::kSingleElement;
    }
    else if (vertex_velocity_mode_name == "single" ||
             vertex_velocity_mode_name == "single_element" ||
             vertex_velocity_mode_name == "random" ||
             vertex_velocity_mode_name == "legacy")
    {
        vertex_velocity_mode = SemiLagrangianAdvection1Form<1>::kSingleElement;
    }
    else if (vertex_velocity_mode_name == "vertex" ||
             vertex_velocity_mode_name == "weighted" ||
             vertex_velocity_mode_name == "vertex_solid_angle" ||
             vertex_velocity_mode_name == "solid_angle")
    {
        vertex_velocity_mode = SemiLagrangianAdvection1Form<1>::kVertexSolidAngle;
    }
    else if (vertex_velocity_mode_name == "edge_dihedral" ||
             vertex_velocity_mode_name == "dihedral" ||
             vertex_velocity_mode_name == "edge")
    {
        vertex_velocity_mode = SemiLagrangianAdvection1Form<1>::kEdgeDihedral;
    }
    else
    {
        if (myid == 0)
            std::cerr << "[warn] Unsupported vertex_velocity_mode='"
                      << vertex_velocity_mode_name
                      << "', using legacy single-element mode." << std::endl;
        vertex_velocity_mode = SemiLagrangianAdvection1Form<1>::kSingleElement;
    }

    if (trace_order < 1 || trace_order > 3)
    {
        if (myid == 0)
            std::cerr << "[warn] Unsupported trace_order=" << trace_order
                      << ", using Euler (trace_order=1)." << std::endl;
        trace_order = 1;
    }
    if (settls_iterations < 1)
    {
        if (myid == 0)
            std::cerr << "[warn] settls_iterations must be >= 1, using 1."
                      << std::endl;
        settls_iterations = 1;
    }

    // ---- Mesh and FE spaces (replicated on every rank) ----
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

    // ---- MPI edge partitioning ----
    const int n_edges = fine_mesh_ptr->GetNEdges();
    int my_threads = 1;
#ifdef _OPENMP
    my_threads = omp_get_max_threads();
#endif
    std::vector<int> all_threads(nprocs);
    MPI_Allgather(&my_threads, 1, MPI_INT,
                  all_threads.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Cyclic (round-robin) edge distribution: rank r processes edges
    // r, r+nprocs, r+2*nprocs, ... This interleaves spatially clustered
    // edges across ranks, balancing work between cheap far-field and
    // expensive boundary-layer edges.
    const int edge_start  = myid;      // first edge for this rank
    const int edge_end    = n_edges;   // upper bound (stride skips others)
    const int edge_stride = nprocs;    // step between consecutive edges
    if (myid == 0)
    {
        std::cout << "[MPI] nprocs=" << nprocs
                  << ", n_edges=" << n_edges
                  << " (cyclic distribution)" << std::endl;
        for (int r = 0; r < nprocs; ++r)
        {
            int cnt = (n_edges - r + nprocs - 1) / nprocs;
            std::cout << "  rank " << r << ": threads=" << all_threads[r]
                      << ", " << cnt << " edges (stride=" << nprocs << ")"
                      << std::endl;
        }
    }

    // ---- Boundary attribute marker (optional) ----
    mfem::Array<int> lid_marker;
    const mfem::Array<int> *lid_marker_ptr = nullptr;
    if (config.has_lid_attributes())
    {
        lid_marker = config.get_lid_marker(fine_mesh_ptr->bdr_attributes.Max());
        lid_marker_ptr = &lid_marker;
    }

    // ---- Assemble RHS and solution (system operator comes from MG) ----
    hcurl::StokesRHS rhs(ND, CG,
                         config.get_exact_data("force_data"),
                         config.get_exact_data("boundary_data_u"),
                         theta, Cw, viscosity, 0.0, lid_marker_ptr);
    hcurl::StokesSolution x(ND, CG);

    // ---- Initial condition ----
    mfem::VectorFunctionCoefficient u_init(dim, config.get_exact_data("initial_data_u"));
    u_init.SetTime(0.);
    x.get_u().ProjectCoefficient(u_init);

    // ---- Semi-Lagrangian advection operator ----
    SemiLagrangianAdvection1Form<1> advection(ND);
    mfem::GridFunction omega_tilde(&ND);
    mfem::GridFunction u_prev(&ND);
    mfem::GridFunction u_n_snapshot(&ND);
    bool has_prev_velocity = false;

    // Velocity function: evaluate current velocity GridFunction at arbitrary
    // physical points via BFS element search.
    auto velocity_func = [fine_mesh_ptr, &u_gf = x.get_u(), dim](
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

    SemiLagrangianAdvection1Form<1>::VelocityFunc velocity_prev_func =
        [fine_mesh_ptr, &u_gf = u_prev, dim](
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

    // ---- Solver (set up once — A is constant) ----
    mfem::FGMRESSolver gmres;
    gmres.SetAbsTol(1e-12);
    gmres.SetRelTol(1e-6);
    gmres.SetMaxIter(500);
    gmres.SetPrintLevel(1);
    gmres.SetOperator(op);
    gmres.SetPreconditioner(mg_solver);
    gmres.SetKDim(128);

    // ---- ParaView output (rank 0 only) ----
    mfem::ParaViewDataCollection vtk_dc("./out/paraview/" + output_file, fine_mesh_ptr.get());
    if (visualisation > 0 && myid == 0)
    {
        vtk_dc.RegisterField("u1", &x.get_u());
        vtk_dc.RegisterField("p1", &x.get_p());
        vtk_dc.SetCycle(0);
        vtk_dc.SetTime(0.0);
        vtk_dc.Save();
    }

    // ---- CSV logger (rank 0 only) ----
    double t = 0.;
    int cycle = 0;
    double advect_time_s = 0.0;
    double solve_time_s = 0.0;
    double advect_time_min_s = 0.0;
    double advect_time_max_s = 0.0;
    double solve_time_min_s = 0.0;
    double solve_time_max_s = 0.0;
    double comm_time_s = 0.0;
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

    SemiLagrangianStepStats advection_stats;

    // Only rank 0 creates the CSV logger (it opens a file for writing)
    std::unique_ptr<SingleFieldCSVLogger> csv;
    if (myid == 0)
    {
        csv = std::make_unique<SingleFieldCSVLogger>(
            config, cycle, t, &ND, x.get_u(), num_it,
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
            profile_advection_thread_balance ? &edge_thread_cpu_util_max : nullptr,
            &advect_time_min_s,
            &advect_time_max_s,
            &solve_time_min_s,
            &solve_time_max_s,
            &comm_time_s);
    }

    // ---- Progress bar (rank 0 only) ----
    int total_cycles = 0;
    for (double tt = dt; tt < T + tol; tt += dt) { total_cycles++; }

    auto run_start = std::chrono::steady_clock::now();
    auto print_progress = [&](int completed_cycles, double t_now)
    {
        if (myid != 0 || total_cycles <= 0) { return; }

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
        u_n_snapshot = x.get_u();

        // 1. Semi-Lagrangian advection: cyclic edge distribution across ranks
        omega_tilde = 0.0;
        auto advect_start = std::chrono::steady_clock::now();
        advection_stats.enable_breakdown = profile_advection_breakdown;
        advection_stats.enable_thread_balance = profile_advection_thread_balance;
        const SemiLagrangianAdvection1Form<1>::VelocityFunc *velocity_prev_ptr =
            (trace_order == 3 && has_prev_velocity) ? &velocity_prev_func : nullptr;
        advection.Apply(velocity_func, boundary_func, t, dt,
                        x.get_u(), omega_tilde, trace_order,
                        (profile_advection_breakdown || profile_advection_thread_balance)
                            ? &advection_stats
                            : nullptr,
                        velocity_prev_ptr,
                        settls_iterations,
                        vertex_velocity_mode,
                        edge_start, edge_end, edge_stride);
        advect_time_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - advect_start)
                            .count();

        // 2. Combine DOFs from all ranks (non-overlapping writes, sum = correct).
        //    Barrier first so the Allreduce timing is pure collective cost;
        //    wait-for-slowest is then charged to the advect min/max spread below.
        MPI_Barrier(MPI_COMM_WORLD);
        auto comm_start = std::chrono::steady_clock::now();
        MPI_Allreduce(MPI_IN_PLACE, omega_tilde.GetData(), n_edges,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        comm_time_s = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - comm_start)
                          .count();

        // Report min/max advection time across ranks.
        double advect_local = advect_time_s;
        MPI_Allreduce(&advect_local, &advect_time_min_s, 1,
                      MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&advect_local, &advect_time_max_s, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        advect_time_s = advect_time_max_s;

        if (profile_advection_breakdown)
        {
            trace_time_s = advection_stats.trace_departure_s;
            split_time_s = advection_stats.split_line_s;
            interior_integral_time_s = advection_stats.interior_integral_s;
            boundary_integral_time_s = advection_stats.boundary_integral_s;
            split_calls = advection_stats.split_calls;
            total_segments = advection_stats.total_segments;
        }
        if (profile_advection_thread_balance)
        {
            edge_thread_min_s = advection_stats.edge_thread_min_s;
            edge_thread_avg_s = advection_stats.edge_thread_avg_s;
            edge_thread_max_s = advection_stats.edge_thread_max_s;
            edge_thread_imbalance = advection_stats.edge_thread_imbalance;
            edge_threads_active = advection_stats.edge_threads_active;
            edge_thread_edges_min = advection_stats.edge_thread_edges_min;
            edge_thread_edges_max = advection_stats.edge_thread_edges_max;
            edge_thread_cpu_min_s = advection_stats.edge_thread_cpu_min_s;
            edge_thread_cpu_avg_s = advection_stats.edge_thread_cpu_avg_s;
            edge_thread_cpu_max_s = advection_stats.edge_thread_cpu_max_s;
            edge_thread_cpu_util_min = advection_stats.edge_thread_cpu_util_min;
            edge_thread_cpu_util_avg = advection_stats.edge_thread_cpu_util_avg;
            edge_thread_cpu_util_max = advection_stats.edge_thread_cpu_util_max;
        }

        // 3. Assemble RHS with omega_tilde (advected field, not u^n)
        rhs.Update(omega_tilde, t, 1./dt);
        rhs.GetBlock(0) *= 1.0/viscosity;

        // 4. Solve linear system (replicated — identical on all ranks)
        auto solve_start = std::chrono::steady_clock::now();
        gmres.Mult(rhs, x);
        op.eliminateConstants(x);
        solve_time_s = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - solve_start)
                           .count();
        num_it = gmres.GetNumIterations();

        // Report min/max solve time across ranks (should be ~equal since the
        // solve is replicated — a non-trivial spread would flag a bug).
        double solve_local = solve_time_s;
        MPI_Allreduce(&solve_local, &solve_time_min_s, 1,
                      MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&solve_local, &solve_time_max_s, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // Update previous-step velocity history for SETTLS.
        u_prev = u_n_snapshot;
        has_prev_velocity = true;

        // 5. Log + progress (rank 0 only)
        if (myid == 0)
        {
            csv->WriteRow();
            print_progress(cycle, t);
        }

        // 6. Visualize (rank 0 only)
        if (visualisation > 0 && cycle % visualisation == 0 && myid == 0)
        {
            vtk_dc.SetCycle(cycle);
            vtk_dc.SetTime(t);
            vtk_dc.Save();
        }
    }

    delete fec_ND;
    delete fec_CG;

    MPI_Finalize();
    return 0;
}
