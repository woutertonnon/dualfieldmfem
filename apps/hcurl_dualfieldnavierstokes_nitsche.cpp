// hcurl_dualfieldnavierstokes_nitsche.cpp
//
// Dual-field Navier-Stokes with BOTH velocity fields in H(curl) (Nedelec).
//
// Both systems are standard hcurl Stokes with Nitsche boundary conditions,
// cross-coupled through their vorticities on a shared ND/CG FE space:
//
//   System 1:  u1 in ND, p1 in CG,  convective vorticity  w2 = curl(u2)
//   System 2:  u2 in ND, p2 in CG,  convective vorticity  w1 = curl(u1)
//
// Time coupling uses a Gauss-Seidel sweep:
//   1. Solve System 1 using w2 = curl(u2^{n})   (previous-step vorticity)
//   2. Solve System 2 using w1 = curl(u1^{n+1}) (freshly updated vorticity)
//
// Energy conservation:
//   Testing System 1 with v = u1:  (curl(u2) x u1, u1) = 0  (a x b . b = 0)
//   Testing System 2 with v = u2:  (curl(u1) x u2, u2) = 0
// Both nonlinear terms vanish independently, giving exact discrete energy
// balance without any DG face penalty or artificial dissipation.
//
// Note: if both systems are initialised from the same u0, the Gauss-Seidel
// coupling is the only source of asymmetry.  The two solutions will in general
// differ from the second time step onward and serve as a cross-check of
// consistency.  For a non-trivial dual-field pair one could use different
// polynomial orders or meshes for the two systems.

#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <boost/program_options.hpp>

#include "mfem.hpp"
#include "BoundaryOperators.h"
#include "io.h"
#include "StokesOperators.h"

using namespace mfem;
using namespace std;
namespace po = boost::program_options;

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
    config.PrintTree(config.get_tree());

    // ---- Configuration -----------------------------------------------------
    double viscosity   = config.get_viscosity();
    int    refinements = config.get_refinements();
    int    order       = config.get_order();
    int    visualisation = config.get_visualisation();
    double tol         = config.get_tol();
    double dt          = config.get_dt();
    double T           = config.get_T();
    double theta       = -1.;    // adjoint-consistent (incomplete) Nitsche
    double Cw          = 10000.;     // no explicit boundary consistency penalty
    // Normal-jump penalty: (sigma * nu / h_F) * sum_F int_F [[u]]·[[v]] dF.
    // Penalises normal-component oscillations in u across interior faces.
    // Scaled by viscosity; sigma=10 is safe for order 1-2.
    double sigma       = 0.0;
    // Curl-jump ghost penalty: (gamma * nu * h_F) * sum_F int_F [[curl u]]·[[curl v]] dF.
    // Smooths the vorticity curl(u) used as the lagged advection coefficient.
    // h_F scaling (not 1/h_F) ensures the term vanishes with mesh refinement.
    double gamma       = 100.0;
    // Heumann upwind: upwind_scale * sum_F int_F |w.n_F| [[u]]·[[v]] dF.
    // Wind-adaptive normal-jump penalty following Heumann, Hiptmair, Pagliantini (2016).
    // Adds O(1) dissipation at faces; coefficient scales naturally with flow speed.
    double upwind_scale = 0.0;
    // Least-squares (PSPG) pressure stabilization:
    //   -delta * int_Omega grad(p) . grad(q) dx   in (1,1) block
    //   +delta * int_Omega (f + (1/dt)*u^n) . grad(q) dx   in RHS(1)
    // Regularises the saddle-point system; delta ~ h^2 is a typical choice.
    double delta = 0.0;
    std::string mesh_string = config.get_mesh();
    std::string output_file = config.get_outputfile();

    // ---- Mesh and FE spaces ------------------------------------------------
    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; l++)
        mesh.UniformRefinement();
    int dim = mesh.Dimension();

    // Both systems share a single pair of FE spaces.
    mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order, dim);
    mfem::FiniteElementCollection *fec_CG = new mfem::H1_FECollection(order, dim);
    mfem::FiniteElementSpace ND(&mesh, fec_ND);
    mfem::FiniteElementSpace CG(&mesh, fec_CG);

    int num_it1 = 0, num_it2 = 0;

    // ---- Boundary attribute marker (optional) ------------------------------
    // When lid_attributes is specified in the config, the nonzero boundary
    // data is applied only on those faces; all others get homogeneous u=0.
    mfem::Array<int> lid_marker;
    const mfem::Array<int> *lid_marker_ptr = nullptr;
    if (config.has_lid_attributes())
    {
        lid_marker = config.get_lid_marker(mesh.bdr_attributes.Max());
        lid_marker_ptr = &lid_marker;
        std::cout << "Lid-driven cavity mode: nonzero BCs on attributes";
        for (int i = 0; i < lid_marker.Size(); i++)
            if (lid_marker[i]) std::cout << " " << i+1;
        std::cout << std::endl;
    }

    // ---- Two H(curl) Stokes systems ----------------------------------------
    //
    // System 1: ND/CG with vorticity from System 2  (w2 = curl(u2))
    // System 2: ND/CG with vorticity from System 1  (w1 = curl(u1))
    //
    hcurl::StokesSystem   sys1(ND, CG, 1./dt, viscosity, theta, Cw, sigma, gamma, upwind_scale, delta);
    hcurl::StokesRHS      rhs1(ND, CG,
                               config.get_exact_data("force_data"),
                               config.get_exact_data("boundary_data_u"),
                               theta, Cw, viscosity, delta, lid_marker_ptr);
    hcurl::StokesSolution x1(ND, CG);

    hcurl::StokesSystem   sys2(ND, CG, 1./dt, viscosity, theta, Cw, sigma, gamma, upwind_scale, delta);
    hcurl::StokesRHS      rhs2(ND, CG,
                               config.get_exact_data("force_data"),
                               config.get_exact_data("boundary_data_u"),
                               theta, Cw, viscosity, delta, lid_marker_ptr);
    hcurl::StokesSolution x2(ND, CG);

    // ---- Initial conditions ------------------------------------------------
    mfem::VectorFunctionCoefficient u_init(3, config.get_exact_data("initial_data_u"));
    u_init.SetTime(0.);
    x1.get_u().ProjectCoefficient(u_init);
    x2.get_u().ProjectCoefficient(u_init);

    // ---- Cross-vorticity coefficients (convection term) -------------------
    // CurlGridFunctionCoefficient always evaluates curl at the *current* state
    // of the GridFunction, so these automatically reflect updated solutions.
    mfem::CurlGridFunctionCoefficient w1(&x1.get_u()); // curl(u1)
    mfem::CurlGridFunctionCoefficient w2(&x2.get_u()); // curl(u2)

    // ---- Upwind transport velocities (Heumann face stabilization) ---------
    // From the characteristic analysis: curl(u2) x u1 = -(u2.grad)u1 + ...,
    // so u1 information propagates along u2 (and vice versa).
    // These wrap the same GridFunctions as w1/w2 and update automatically.
    mfem::VectorGridFunctionCoefficient u1_coef(&x1.get_u()); // transport dir for sys2
    mfem::VectorGridFunctionCoefficient u2_coef(&x2.get_u()); // transport dir for sys1

    // ---- Visualisation -----------------------------------------------------
    double t     = 0.;
    int    cycle = 0;

    mfem::ParaViewDataCollection vtk_dc(
        "./data/visualisation/paraview/" + output_file, &mesh);
    if (visualisation > 0)
    {
        vtk_dc.RegisterField("u1", &x1.get_u());
        vtk_dc.RegisterField("p1", &x1.get_p());
        vtk_dc.RegisterField("u2", &x2.get_u());
        vtk_dc.RegisterField("p2", &x2.get_p());
        vtk_dc.SetCycle(0);
        vtk_dc.SetTime(0.0);
        vtk_dc.Save();
    }

    // ---- Solvers -----------------------------------------------------------
    hcurl::GMRESSolver solv1(ND, CG, num_it1, viscosity);
    hcurl::GMRESSolver solv2(ND, CG, num_it2, viscosity);

    // ---- CSV logging -------------------------------------------------------
    HCurlDualFieldCSVLogger csv(config, cycle, t, &ND, x1.get_u(), x2.get_u(),
                                num_it1, num_it2);

    // ---- Time loop ---------------------------------------------------------
    for (t = dt, cycle = 1; t < T + tol; t += dt, cycle++)
    {
        csv.WriteRow();

        // -- System 1: convection by w2=curl(u2^n), upwind along u2^n --------
        sys1.Update(w2, &u2_coef);
        rhs1.Update(x1.get_u(), t, 1./dt);
        solv1.SetOperator(sys1);
        solv1.Mult(rhs1, x1);
        // After this call x1 (and therefore w1=curl(u1), u1_coef) is ^{n+1}

        // -- System 2: convection by w1=curl(u1^{n+1}), upwind along u1^{n+1}
        sys2.Update(w1, &u1_coef);
        rhs2.Update(x2.get_u(), t, 1./dt);
        solv2.SetOperator(sys2);
        solv2.Mult(rhs2, x2);

        if (cycle % visualisation ==0 )
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
