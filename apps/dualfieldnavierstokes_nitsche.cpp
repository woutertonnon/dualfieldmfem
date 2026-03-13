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

    // ---- H(curl) stabilisation parameters ----------------------------------
    //
    // Adjoint-consistency parameter for Nitsche BCs:
    //   theta = -1 gives the adjoint-consistent (incomplete) variant.
    double hcurl_theta = -1.;
    // Nitsche boundary penalty: (Cw / h_F) * int_{dOmega} (n x u).(n x v) dS.
    // Must be large enough to enforce BCs; 10000 is safe.
    double hcurl_Cw = 10000.;
    // Normal-jump penalty: (sigma * nu / h_F) * sum_F int_F [[u]].[[v]] dF.
    // For ND the tangential component is continuous; [[u]] is purely normal.
    // Not needed when curl-jump ghost penalty is active.
    double hcurl_sigma = 0.0;
    // Curl-jump ghost penalty: (gamma * nu * h_F) * sum_F int_F [[curl u]].[[curl v]] dF.
    // Smooths the vorticity curl(u) used as the lagged advection coefficient
    // in the H(div) system.  h_F scaling ensures convergence is not degraded.
    double hcurl_gamma = 100.0;
    // Heumann upwind: upwind_scale * sum_F int_F |w.n_F| [[u]].[[v]] dF.
    // Wind-adaptive normal-jump penalty (Heumann, Hiptmair, Pagliantini 2016).
    double hcurl_upwind = 0.0;
    // PSPG pressure stabilisation: delta ~ h^2.
    double hcurl_delta = 0.0;

    // ---- H(div) stabilisation parameters -----------------------------------
    //
    // Interior-face tangential-jump penalty (DG penalty):
    //   (sigma * nu / h_F) * sum_F int_F [[u]].[[v]] dF
    // For RT the normal component is continuous; [[u]] is purely tangential.
    // Controls tangential DOF oscillations that arise when viscosity enters
    // only through the vorticity coupling B^T D^{-1} B (volume-only).
    double hdiv_sigma = 100.0;
    // Interior-face div-jump ghost penalty:
    //   (gamma * nu * h_F) * sum_F int_F [[div u]] [[div v]] dF
    // Not needed: div(u) couples to pressure through the saddle-point
    // structure, and the convection coefficient w lives in conforming ND.
    double hdiv_gamma = 0.0;
    // Boundary tangential penalty (Nitsche-style):
    //   (Cw / h_F) * int_{dOmega} (n x u).(n x v) dS
    // Penalises the tangential component of u on the boundary.  For RT the
    // normal component u.n is strongly enforced via essential BCs; this term
    // weakly controls the tangential trace.  A matching RHS consistency term
    // is added so the exact solution is not penalised.
    double hdiv_Cw = 100.0;

    // ---- Mesh and FE spaces ------------------------------------------------
    Mesh mesh(mesh_string.c_str(), 1, 1);
    for (int l = 0; l < refinements; l++)
        mesh.UniformRefinement();
    int dim = mesh.Dimension();

    // DG ~ L2, ND ~ H(curl), RT ~ H(div), CG ~ H1
    mfem::FiniteElementCollection *fec_DG = new mfem::L2_FECollection(order - 1, dim);
    mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order, dim);
    mfem::FiniteElementCollection *fec_RT = new mfem::RT_FECollection(order - 1, dim);
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
        std::cout << "Lid-driven cavity mode: nonzero BCs on attributes";
        for (int i = 0; i < lid_marker.Size(); i++)
            if (lid_marker[i]) std::cout << " " << i+1;
        std::cout << std::endl;
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
    hdiv::StokesSystem   hdiv_sys(RT, ND, DG, ess_tdof, 1./dt, viscosity,
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

    // ---- Visualisation -----------------------------------------------------
    double t     = 0.;
    int    cycle = 0;

    mfem::ParaViewDataCollection vtk_dc(
        "./data/visualisation/paraview/" + output_file, &mesh);
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
    hcurl::DirectSolver hcurl_solv(ND, CG, num_it_A1);
    hdiv::DirectSolver  hdiv_solv(RT, ND, DG, num_it_A2);

    // ---- CSV logging -------------------------------------------------------
    DualFieldCSVLogger csv(config, cycle, t, t, &ND, &RT,
                           hcurl_x.get_u(), hdiv_x.get_u(), hdiv_x.get_w(),
                           num_it_A1, num_it_A2);

    // ---- Time loop ---------------------------------------------------------
    for (t = dt, cycle = 1; t < T + tol; t += dt, cycle++)
    {
        csv.WriteRow();

        // -- H(curl): convection by w_hdiv (vorticity from H(div) system) ----
        hcurl_sys.Update(w_hdiv);
        hcurl_rhs.Update(hcurl_x.get_u(), t, 1./dt);
        hcurl_solv.SetOperator(hcurl_sys);
        hcurl_solv.Mult(hcurl_rhs, hcurl_x);
        // After this call hcurl_x (and therefore w_hcurl = curl(u1)) is at n+1

        // -- H(div): convection by w_hcurl (curl of freshly updated u1) ------
        hdiv_sys.Update(w_hcurl);
        hdiv_rhs.Update(hdiv_x.get_u(), t, 1./dt);
        hdiv_solv.SetOperator(hdiv_sys);
        hdiv_solv.Mult(hdiv_rhs, hdiv_x);

        if (visualisation > 0 && cycle % visualisation == 0)
        {
            vtk_dc.SetCycle(cycle);
            vtk_dc.SetTime(t);
            vtk_dc.Save();
        }
    }

    delete fec_RT;
    delete fec_ND;
    delete fec_CG;
    delete fec_DG;

    return 0;
}
