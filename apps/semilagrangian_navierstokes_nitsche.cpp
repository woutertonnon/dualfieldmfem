#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <boost/program_options.hpp>

#include "mfem.hpp"
#include "BoundaryOperators.h"
#include "io.h"
#include "StokesOperators.h"
#include "SemiLagrangianAdvection.h"

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
    double Cw = 100.;
    double dt = config.get_dt();
    double T = config.get_T();

    // ---- Mesh and FE spaces ----
    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; l++)
    {
        mesh.UniformRefinement();
    }
    int dim = mesh.Dimension();

    mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order, dim);
    mfem::FiniteElementCollection *fec_CG = new mfem::H1_FECollection(order, dim);
    mfem::FiniteElementSpace ND(&mesh, fec_ND);
    mfem::FiniteElementSpace CG(&mesh, fec_CG);

    int num_it = 0;

    // ---- Boundary attribute marker (optional) ----
    mfem::Array<int> lid_marker;
    const mfem::Array<int> *lid_marker_ptr = nullptr;
    if (config.has_lid_attributes())
    {
        lid_marker = config.get_lid_marker(mesh.bdr_attributes.Max());
        lid_marker_ptr = &lid_marker;
    }

    // ---- Assemble system (constant — no Eulerian convection) ----
    // No interior-face DG penalties (sigma=0, gamma=0): only Nitsche BCs.
    hcurl::StokesSystem sys(ND, CG, 1./dt, viscosity, theta, Cw,
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

    // ---- Semi-Lagrangian advection operator ----
    SemiLagrangianAdvection1Form<1> advection(ND);
    mfem::GridFunction omega_tilde(&ND);

    // Velocity function: evaluate current velocity GridFunction at arbitrary
    // physical points via BFS element search.
    auto velocity_func = [&mesh, &u_gf = x.get_u(), dim](
        const mfem::Vector &pt, double, mfem::Vector &v) {
        v.SetSize(dim);
        mfem::IntegrationPoint ip;
        int elem = FindElementBFS(mesh, 0, pt, ip);
        if (elem >= 0)
        {
            mfem::IsoparametricTransformation eltrans;
            mesh.GetElementTransformation(elem, &eltrans);
            eltrans.SetIntPoint(&ip);
            u_gf.GetVectorValue(eltrans, ip, v);
        }
        else
        {
            v = 0.0;
        }
    };

    // Boundary function: dispatch based on boundary attribute.
    // On lid-marked faces, apply boundary_data_u; on unmarked faces, apply
    // homogeneous Dirichlet (mirrors hcurl::StokesRHS lid_marker logic).
    auto bdr_data = config.get_exact_data("boundary_data_u");
    SemiLagrangianAdvection1Form<1>::BoundaryFunc boundary_func =
        [&bdr_data, lid_marker_ptr, dim](
            const mfem::Vector &pt, double t, int bdr_attr, mfem::Vector &v) {
        v.SetSize(dim);
        if (!lid_marker_ptr)
        {
            // No lid attributes → apply boundary_data_u everywhere
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
    // Unpreconditioned MINRES on the full symmetric saddle-point system.
    mfem::MINRESSolver solv;
    solv.iterative_mode = true;
    solv.SetOperator(sys);
    solv.SetRelTol(tol);
    solv.SetAbsTol(1e-10);
    solv.SetMaxIter(100000);
    solv.SetPrintLevel(0);

    // ---- ParaView output ----
    mfem::ParaViewDataCollection vtk_dc("./out/paraview/" + output_file, &mesh);
    if (visualisation > 0)
    {
        vtk_dc.RegisterField("u1", &x.get_u());
        vtk_dc.RegisterField("p1", &x.get_p());
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
        // 1. Semi-Lagrangian advection: compute omega_tilde
        auto advect_start = std::chrono::steady_clock::now();
        advection.Apply(velocity_func, boundary_func, t, dt,
                        x.get_u(), omega_tilde, /*trace_order=*/1);
        advect_time_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - advect_start)
                            .count();

        // 2. Assemble RHS with omega_tilde (advected field, not u^n)
        rhs.Update(omega_tilde, t, 1./dt);

        // 3. Solve (reuses UMFPack factorization since A is constant)
        auto solve_start = std::chrono::steady_clock::now();
        solv.Mult(rhs, x);
        solve_time_s = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - solve_start)
                           .count();
        num_it = solv.GetNumIterations();

        // 4. Log + progress
        csv.WriteRow();
        print_progress(cycle, t);

        // 5. Visualize
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
