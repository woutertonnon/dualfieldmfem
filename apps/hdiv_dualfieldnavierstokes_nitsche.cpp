// hdiv_dualfieldnavierstokes_nitsche.cpp
//
// Dual-field Navier-Stokes with BOTH velocity fields in H(div) (Raviart-Thomas).
//
// Both systems are standard hdiv Stokes (velocity-vorticity-pressure),
// cross-coupled through their vorticities on a shared RT/ND/DG FE space:
//
//   System 1:  u1 in RT, w1 in ND, p1 in DG,  convective vorticity  w2
//   System 2:  u2 in RT, w2 in ND, p2 in DG,  convective vorticity  w1
//
// Time coupling uses a Gauss-Seidel sweep:
//   1. Solve System 1 using w2 from system 2 at the previous time step
//   2. Solve System 2 using w1 from the freshly updated system 1
//
// Energy conservation:
//   Testing System 1 with v = u1:  (w2 x u1, u1) = 0  (a x b . b = 0)
//   Testing System 2 with v = u2:  (w1 x u2, u2) = 0
// Both nonlinear terms vanish independently.

#include <iostream>
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
    std::string mesh_string  = config.get_mesh();
    std::string output_file  = config.get_outputfile();

    // ---- Stabilisation parameters ------------------------------------------
    //
    // Interior-face tangential-jump penalty (DG penalty):
    //   (sigma * nu / h_F) * sum_F int_F [[u]] . [[v]] dF
    // For RT elements the normal component is single-valued, so [[u]] is
    // purely tangential.  This controls tangential DOF oscillations that
    // arise when viscosity enters only through the vorticity coupling.
    double sigma = 100.0;

    // Interior-face div-jump ghost penalty:
    //   (gamma * nu * h_F) * sum_F int_F [[div u]] [[div v]] dF
    // For RT elements div(u) is in L2 (discontinuous across faces).
    // Penalises divergence jumps to smooth the pressure coupling.
    // h_F scaling ensures the term vanishes with mesh refinement.
    double gamma = 0.0;

    // Boundary tangential penalty (Nitsche-style):
    //   (Cw / h_F) * int_{dOmega} (n x u) . (n x v) dS
    // Penalises the tangential component of u on the boundary.
    // For RT the normal component u.n is strongly enforced via essential
    // BCs; this term weakly controls the tangential trace.
    // A matching RHS consistency term (Cw/h)(n x u_D).(n x v) is added
    // so that the exact solution is not penalised.
    double Cw = 100.0;

    // ---- Mesh and FE spaces ------------------------------------------------
    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; l++)
        mesh.UniformRefinement();
    int dim = mesh.Dimension();

    // Both systems share the same FE spaces
    mfem::FiniteElementCollection *fec_DG = new mfem::L2_FECollection(order - 1, dim);
    mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order, dim);
    mfem::FiniteElementCollection *fec_RT = new mfem::RT_FECollection(order - 1, dim);
    mfem::FiniteElementSpace DG(&mesh, fec_DG);
    mfem::FiniteElementSpace ND(&mesh, fec_ND);
    mfem::FiniteElementSpace RT(&mesh, fec_RT);

    int num_it1 = 0, num_it2 = 0;

    // ---- Boundary attribute marker (optional) ------------------------------
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

    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1; // all boundaries essential for RT (u·n prescribed everywhere)

    mfem::Array<int> ess_tdof;
    RT.GetEssentialTrueDofs(ess_bdr, ess_tdof);

    // ---- Two H(div) Stokes systems ----------------------------------------
    //
    // System 1: RT/ND/DG with vorticity from System 2  (w2)
    // System 2: RT/ND/DG with vorticity from System 1  (w1)
    //
    hdiv::StokesSystem   sys1(RT, ND, DG, ess_tdof, 1./dt, viscosity, sigma, Cw, gamma);
    hdiv::StokesRHS      rhs1(RT, ND, DG, ess_tdof,
                              config.get_exact_data("force_data"),
                              config.get_exact_data("boundary_data_u"),
                              viscosity, lid_marker_ptr, Cw);
    hdiv::StokesSolution x1(RT, ND, DG);

    hdiv::StokesSystem   sys2(RT, ND, DG, ess_tdof, 1./dt, viscosity, sigma, Cw, gamma);
    hdiv::StokesRHS      rhs2(RT, ND, DG, ess_tdof,
                              config.get_exact_data("force_data"),
                              config.get_exact_data("boundary_data_u"),
                              viscosity, lid_marker_ptr, Cw);
    hdiv::StokesSolution x2(RT, ND, DG);

    // ---- Initial conditions ------------------------------------------------
    mfem::VectorFunctionCoefficient u_init(3, config.get_exact_data("initial_data_u"));
    u_init.SetTime(0.);
    x1.get_u().ProjectCoefficient(u_init);
    x2.get_u().ProjectCoefficient(u_init);

    // ---- Cross-vorticity coefficients (convection term) -------------------
    // In hdiv, vorticity w is explicitly solved for as part of the solution.
    // These wrap the w GridFunctions and update automatically.
    mfem::VectorGridFunctionCoefficient w1(&x1.get_w()); // vorticity from system 1
    mfem::VectorGridFunctionCoefficient w2(&x2.get_w()); // vorticity from system 2

    // ---- Visualisation -----------------------------------------------------
    double t     = 0.;
    int    cycle = 0;

    mfem::ParaViewDataCollection vtk_dc(
        "./out/paraview/" + output_file, &mesh);
    if (visualisation > 0)
    {
        vtk_dc.RegisterField("u1", &x1.get_u());
        vtk_dc.RegisterField("w1", &x1.get_w());
        vtk_dc.RegisterField("p1", &x1.get_p());
        vtk_dc.RegisterField("u2", &x2.get_u());
        vtk_dc.RegisterField("w2", &x2.get_w());
        vtk_dc.RegisterField("p2", &x2.get_p());
        vtk_dc.SetCycle(0);
        vtk_dc.SetTime(0.0);
        vtk_dc.Save();
    }

    // ---- Solvers -----------------------------------------------------------
    hdiv::GMRESSolver solv1(RT, ND, DG, num_it1, 1./dt, tol);
    hdiv::GMRESSolver solv2(RT, ND, DG, num_it2, 1./dt, tol);

    // ---- CSV logging -------------------------------------------------------
    HDivDualFieldCSVLogger csv(config, cycle, t, &RT,
                               x1.get_u(), x2.get_u(),
                               num_it1, num_it2);

    // ---- Time loop ---------------------------------------------------------
    for (t = dt, cycle = 1; t < T + tol; t += dt, cycle++)
    {
        csv.WriteRow();

        // -- System 1: convection by w2 (vorticity from system 2) -----------
        sys1.Update(w2);
        rhs1.Update(x1.get_u(), t, 1./dt);
        solv1.SetOperator(sys1);
        solv1.Mult(rhs1, x1);
        // After this call x1 (and therefore w1) is at time n+1

        // -- System 2: convection by w1 (freshly updated from system 1) -----
        sys2.Update(w1);
        rhs2.Update(x2.get_u(), t, 1./dt);
        solv2.SetOperator(sys2);
        solv2.Mult(rhs2, x2);

        if (visualisation > 0 && cycle % visualisation == 0)
        {
            vtk_dc.SetCycle(cycle);
            vtk_dc.SetTime(t);
            vtk_dc.Save();
        }
    }

    delete fec_RT;
    delete fec_ND;
    delete fec_DG;

    return 0;
}
