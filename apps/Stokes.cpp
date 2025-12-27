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
    NitscheStokesConfig config(config_path);
    config.PrintTree(config.get_tree());

    // ------------------------------------------------------------------
    // 0. Configuration
    // ------------------------------------------------------------------
    double mass = config.get_mass();
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
    double Cw = 0.;

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
    FiniteElementCollection *fec_ND = new ND_FECollection(order, dim);
    FiniteElementCollection *fec_CG = new H1_FECollection(order, dim);

    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);


    int num_it_A1;
    num_it_A1  = 0;


    // A1 blocks:

    // A1 blocks:
    StokesSystem sys(ND, CG, mass, viscosity, 1., 100.);
    StokesRHS rhs(ND, CG, config.get_exact_data("force_data"), config.get_exact_data("exact_data_u"),1.,100.,viscosity);
    StokesSolution x(ND, CG);

    SchurSolver solv(ND,CG,mass,viscosity);
    NitscheStokesCSVLogger csv(config, x.get_u(), num_it_A1);
    //mfem::KLUSolver umfpack;

    solv.SetOperator(sys);
    solv.Mult(rhs,x);

    //num_it_A1 = solver->GetNumIterations();


    csv.WriteRow();

    delete fec_ND;
    delete fec_CG;

    return 0;
}
