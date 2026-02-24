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

    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1; // attribute 1 is essential

    mfem::Array<int> ess_tdof;
    RT.GetEssentialTrueDofs(ess_bdr, ess_tdof);


    hdiv::StokesSystem hdiv_sys(RT,ND,DG, ess_tdof, 1./dt, viscosity);
    hdiv::StokesRHS hdiv_rhs(RT, ND, DG, ess_tdof, config.get_exact_data("force_data"), config.get_exact_data("boundary_data_u"));
    hdiv::StokesSolution hdiv_x(RT, ND, DG);

    hcurl::StokesSystem hcurl_sys(ND, CG, 1./dt, viscosity, theta, Cw);
    hcurl::StokesRHS hcurl_rhs(ND, CG, config.get_exact_data("force_data"), config.get_exact_data("boundary_data_u"),theta,Cw,viscosity);
    hcurl::StokesSolution hcurl_x(ND, CG);

    mfem::VectorFunctionCoefficient u_init(3, config.get_exact_data("initial_data_u"));
    u_init.SetTime(0.);
    hdiv_x.get_u().ProjectCoefficient(u_init);
    hcurl_x.get_u().ProjectCoefficient(u_init);

    mfem::VectorGridFunctionCoefficient w1(&hdiv_x.get_w());
    mfem::CurlGridFunctionCoefficient w2(&hcurl_x.get_u());
   // mfem::GridFunction w2(&RT);
   // mfem::VectorFunctionCoefficient w_exact_coeff(3, config.get_exact_data("exact_data_w"));
   // w2.ProjectCoefficient(w_exact_coeff);

    double t = 0.;
    int cycle = 0;

    mfem::ParaViewDataCollection vtk_dc("./data/visualisation/paraview/" + output_file, &mesh);
    if (visualisation > 0)
    {
        vtk_dc.RegisterField("u1", &hcurl_x.get_u()); // Register field for visualization
        vtk_dc.RegisterField("u2", &hdiv_x.get_u()); // Register field for visualization
        vtk_dc.RegisterField("w1", &hdiv_x.get_w()); // Register field for visualization
        //vtk_dc.RegisterField("w2", &w2); // Register field for visualization
        vtk_dc.RegisterField("p0", &hcurl_x.get_p()); // Register field for visualization
        vtk_dc.RegisterField("p3", &hdiv_x.get_p()); // Register field for visualization
        vtk_dc.SetCycle(0);             // Set initial cycle
        vtk_dc.SetTime(0.0);            // Set initial time
        vtk_dc.Save();                  // Save initial data
    }

    DualFieldCSVLogger csv(config,cycle, t, t, &ND, &RT, hcurl_x.get_u(), hdiv_x.get_u(), hdiv_x.get_w(), num_it_A1, num_it_A2);
    for(t=dt, cycle=1; t<T+tol; t+=dt, cycle++){

        csv.WriteRow();

        // Solve H(curl) system
        hcurl_sys.Update(w1);
        hcurl_rhs.Update(hcurl_x.get_u(),t,1./dt);


        hcurl::SchurSolver hcurl_solv(ND,CG,1./dt,viscosity,num_it_A1,tol);
        hcurl_solv.SetOperator(hcurl_sys);    
        hcurl_solv.Mult(hcurl_rhs,hcurl_x);


        // Solve H(div) system

        hdiv_sys.Update(w2);
        hdiv_rhs.Update(hdiv_x.get_u(),t,1./dt);

        mfem::GMRESSolver hdiv_solv;
        //std::cout << "test17\n";
        //mfem::SparseMatrix *mono_mat = sys.CreateMonolithic();
        //std::cout << "test18\n";
        hdiv_solv.SetOperator(hdiv_sys);
        hdiv_solv.SetPrintLevel(1);
        hdiv_solv.SetRelTol(1e-5);
        hdiv_solv.SetAbsTol(1e-12);
        hdiv_solv.SetKDim(300);
        hdiv_solv.SetMaxIter(1000000);

        //X.Print(std::cout);
        //B.Print(std::cout);
        hdiv_solv.Mult(hdiv_rhs,hdiv_x);
        num_it_A2 = hdiv_solv.GetNumIterations();
	//rhs.get_p().Print(std::cout);


        //if (visualisation > 0)
        {
            vtk_dc.SetCycle(cycle);     // Update cycle in ParaView
            vtk_dc.SetTime(t); // Update time in ParaView
            vtk_dc.Save();              // Save data
        }


    }

    delete fec_RT;
    delete fec_ND;
    delete fec_CG;
    delete fec_DG;

    return 0;
}
