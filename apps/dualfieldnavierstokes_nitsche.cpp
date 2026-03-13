#include <iostream>
#include <boost/program_options.hpp>

#include "mfem.hpp"
#include "io.h"
#include "StokesOperators.h"

using namespace mfem;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char *argv[])
{
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

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------
    double viscosity   = config.get_viscosity();
    int    refinements = config.get_refinements();
    int    order       = config.get_order();
    int    visualisation = config.get_visualisation();
    double tol         = config.get_tol();
    double dt          = config.get_dt();
    double T           = config.get_T();
    double theta       = -1.;
    double Cw          = 1000.;
    std::string mesh_string  = config.get_mesh();
    std::string output_file  = config.get_outputfile();

    // ------------------------------------------------------------------
    // Mesh and FE spaces
    // ------------------------------------------------------------------
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

    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1; // all boundaries are essential for RT (u·n prescribed everywhere)

    mfem::Array<int> ess_tdof;
    RT.GetEssentialTrueDofs(ess_bdr, ess_tdof);

    hdiv::StokesSystem   hdiv_sys(RT, ND, DG, ess_tdof, 1./dt, viscosity);
    hdiv::StokesRHS      hdiv_rhs(RT, ND, DG, ess_tdof, config.get_exact_data("force_data"), config.get_exact_data("boundary_data_u"), viscosity, lid_marker_ptr);
    hdiv::StokesSolution hdiv_x(RT, ND, DG);

    hcurl::StokesSystem   hcurl_sys(ND, CG, 1./dt, viscosity, theta, Cw);
    hcurl::StokesRHS      hcurl_rhs(ND, CG, config.get_exact_data("force_data"), config.get_exact_data("boundary_data_u"), theta, Cw, viscosity, 0.0, lid_marker_ptr);
    hcurl::StokesSolution hcurl_x(ND, CG);

    mfem::VectorFunctionCoefficient u_init(3, config.get_exact_data("initial_data_u"));
    u_init.SetTime(0.);
    hdiv_x.get_u().ProjectCoefficient(u_init);
    hcurl_x.get_u().ProjectCoefficient(u_init);

    mfem::VectorGridFunctionCoefficient w1(&hdiv_x.get_w());
    mfem::CurlGridFunctionCoefficient   w2(&hcurl_x.get_u());

    double t     = 0.;
    int    cycle = 0;

    mfem::ParaViewDataCollection vtk_dc("./data/visualisation/paraview/" + output_file, &mesh);
    if (visualisation > 0)
    {
        vtk_dc.RegisterField("u1", &hcurl_x.get_u());
        vtk_dc.RegisterField("u2", &hdiv_x.get_u());
        vtk_dc.RegisterField("w1", &hdiv_x.get_w());
        vtk_dc.RegisterField("p0", &hcurl_x.get_p());
        vtk_dc.RegisterField("p3", &hdiv_x.get_p());
        vtk_dc.SetCycle(0);
        vtk_dc.SetTime(0.0);
        vtk_dc.Save();
    }

    hdiv::DirectSolver hdiv_solv(RT, ND, DG, num_it_A2);
    hcurl::DirectSolver hcurl_solv(ND, CG, num_it_A1);

    DualFieldCSVLogger csv(config, cycle, t, t, &ND, &RT, hcurl_x.get_u(), hdiv_x.get_u(), hdiv_x.get_w(), num_it_A1, num_it_A2);
    for (t = dt, cycle = 1; t < T + tol; t += dt, cycle++)
    {
        csv.WriteRow();

        // Solve H(curl) system
        hcurl_sys.Update(w1);
        hcurl_rhs.Update(hcurl_x.get_u(), t, 1./dt);

        hcurl_solv.SetOperator(hcurl_sys);
        hcurl_solv.Mult(hcurl_rhs, hcurl_x);

        // Solve H(div) system
        hdiv_sys.Update(w2);
        hdiv_rhs.Update(hdiv_x.get_u(), t, 1./dt);

        hdiv_solv.SetOperator(hdiv_sys);    // re-factor A with new convection
        hdiv_solv.Mult(hdiv_rhs, hdiv_x);

        if (visualisation > 0)
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
