#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <boost/program_options.hpp>

#include "mfem.hpp"
#include "BoundaryOperators.h"
#include "io.h" // SimulationConfig, EnergyCSVLogger
#include "StokesOperators.h"

using namespace mfem;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char *argv[])
{
    // ---- Parse command-line options with Boost BEFORE MPI_Init (recommended) ----
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
            // Just print from rank 0 later, but we don't know rank yet.
            // For now, print unconditionally (or move this after MPI_Init).
            std::cout << desc << "\n";
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error parsing command line: " << e.what() << "\n";
        return 1;
    }

    // Optionally only rank 0 prints what config it’s using
    std::cout << "Using config file: " << config_path << std::endl;

    // ---- Use the parsed config path ----
    DualFieldConfig config(config_path);
    config.PrintTree(config.get_tree());

    // ------------------------------------------------------------------
    // 0. Configuration
    // ------------------------------------------------------------------
    double viscosity = config.get_viscosity();
    int refinements = config.get_refinements();
    int order = config.get_order();
    int visualisation = config.get_visualisation();
    int printlevel = config.get_printlevel();
    double tol = config.get_tol();
    bool has_exact_u = config.has_exact_u();
    std::string mesh_string = config.get_mesh();
    std::string output_file = config.get_outputfile();
    std::string solver_type = config.get_solver();
    double theta = -1.;
    double Cw = 100.;
    double dt = config.get_dt();
    double T = config.get_T();

  
    // ------------------------------------------------------------------
    // 1. Mesh and FE spaces (PARALLEL)
    // ------------------------------------------------------------------
    Mesh mesh(mesh_string.c_str(), 1, 1);
    std::cout << mesh.GetNE() << std::endl;
    for (int l = 0; l < refinements; l++)
    {
        mesh.UniformRefinement();
    }
    std::cout << mesh.GetNE() << std::endl;
    int dim = mesh.Dimension();

    // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
    mfem::FiniteElementCollection *fec_DG = new mfem::L2_FECollection(order - 1, dim);
    mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order, dim);
    mfem::FiniteElementCollection *fec_RT = new mfem::RT_FECollection(order - 1, dim);
    mfem::FiniteElementCollection *fec_CG = new mfem::H1_FECollection(order, dim);
    mfem::FiniteElementSpace DG(&mesh, fec_DG);
    mfem::FiniteElementSpace ND(&mesh, fec_ND);
    mfem::FiniteElementSpace RT(&mesh, fec_RT);
    mfem::FiniteElementSpace CG(&mesh, fec_CG);


    int num_it_A1, num_it_A2;
    num_it_A1  = 0;
    num_it_A2 = 0;

    // ---- Boundary attribute marker (optional) ------------------------------
    mfem::Array<int> lid_marker;
    const mfem::Array<int> *lid_marker_ptr = nullptr;
    if (config.has_lid_attributes())
    {
        lid_marker = config.get_lid_marker(mesh.bdr_attributes.Max());
        lid_marker_ptr = &lid_marker;
    }

    // A1 blocks:
    hcurl::StokesSystem sys(ND, CG, 1./dt, viscosity, theta, Cw);
    hcurl::StokesRHS rhs(ND, CG, config.get_exact_data("force_data"), config.get_exact_data("boundary_data_u"),theta,Cw,viscosity, 0.0, lid_marker_ptr);
    hcurl::StokesSolution x(ND, CG);
    mfem::VectorFunctionCoefficient u_exact(3, config.get_exact_data("initial_data_u"));
    u_exact.SetTime(0.);
    x.get_u().ProjectCoefficient(u_exact);

    mfem::GridFunction u2(&RT);
    mfem::GridFunction w1(&ND);
    mfem::CurlGridFunctionCoefficient w2(&x.get_u());      // curl(u) for convection term
    mfem::VectorGridFunctionCoefficient u_coef(&x.get_u()); // u for Heumann upwind (beta=u)
   // mfem::GridFunction w2(&RT);
   // mfem::VectorFunctionCoefficient w_exact_coeff(3, config.get_exact_data("exact_data_w"));
   // w2.ProjectCoefficient(w_exact_coeff);

    double t = 0.;
    int cycle = 0;

    mfem::ParaViewDataCollection vtk_dc("./data/visualisation/paraview/" + output_file, &mesh);
    if (visualisation > 0)
    {
        vtk_dc.RegisterField("u1", &x.get_u()); // Register field for visualization
        //vtk_dc.RegisterField("u2", &v); // Register field for visualization
        //vtk_dc.RegisterField("w1", &w); // Register field for visualization
        //vtk_dc.RegisterField("w2", &w2); // Register field for visualization
        vtk_dc.RegisterField("p0", &x.get_p()); // Register field for visualization
        //vtk_dc.RegisterField("p3", &q); // Register field for visualization
        vtk_dc.SetCycle(0);             // Set initial cycle
        vtk_dc.SetTime(0.0);            // Set initial time
        vtk_dc.Save();                  // Save initial data
    }

    DualFieldCSVLogger csv(config,cycle, t, t, &ND, &RT, x.get_u(), u2, w1, num_it_A1, num_it_A2);
    for(t=dt, cycle=1; t<T+tol; t+=dt, cycle++){

        csv.WriteRow();

        sys.Update(w2, &u_coef);
        rhs.Update(x.get_u(),t,1./dt);


        hcurl::DirectSolver solv(ND,CG,num_it_A1);
        solv.SetOperator(sys);
        solv.Mult(rhs,x);
	//rhs.get_p().Print(std::cout);


        //if (visualisation > 0)
        {
            vtk_dc.SetCycle(cycle);     // Update cycle in ParaView
            vtk_dc.SetTime(t); // Update time in ParaView
            vtk_dc.Save();              // Save data
        }


    }


    delete fec_ND;
    delete fec_CG;

    return 0;
}
