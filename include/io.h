#include "mfem.hpp"                            // For MFEM types like mfem::Vector
#include <boost/property_tree/ptree.hpp>       // For Boost Property Tree
#include <boost/property_tree/json_parser.hpp> // For Boost JSON parsing
#include <iostream>                            // For std::cerr and I/O
#include <fstream>                             // For file output
#include <cstdlib>                             // For std::system
#include <dlfcn.h>                             // For dynamic library loading (dlopen, dlsym, dlclose)

// Class to manage the simulation configuration
class SimulationConfig
{
public:
    void PrintTree(const boost::property_tree::ptree &pt, int depth = 0)
    {
        std::string indent(depth * 2, ' ');
        for (const auto &node : pt)
        {
            std::cout << indent << node.first;
            if (!node.second.data().empty())
            {
                std::cout << " = " << node.second.data();
            }
            std::cout << std::endl;

            PrintTree(node.second, depth + 1); // recurse into children
        }
    }
    // Constructor that reads configuration from a JSON file
    explicit SimulationConfig(const std::string &filename)
    {
        // Read the JSON file into a property tree
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(filename, tree);

        int depth = 1;
        PrintTree(tree, depth);

        // Populate member variables from the JSON configuration
        mesh = tree.get<std::string>("mesh");
        outputfile = tree.get<std::string>("outputfile");
        dt = tree.get<double>("dt");
        T = tree.get<double>("T");
        refinements = tree.get<int>("refinements",0);
        order = tree.get<int>("order",1);
        visualisation = tree.get<int>("visualisation",0);
        velocity_code = tree.get<std::string>("velocity","");
        boundary_data_u_code = tree.get<std::string>("boundary_data_u","");
        boundary_data_A_code = tree.get<std::string>("boundary_data_A","");
        initial_data_u_code = tree.get<std::string>("initial_data_u","");
        initial_data_A_code = tree.get<std::string>("initial_data_A","");
        viscosity = tree.get<double>("viscosity",0);

        // Generate code and load the dynamic library
        PrepareCodeAndLibrary();
    }

    // Getter methods for configuration parameters
    double get_dt() const { return dt; }
    double get_T() const { return T; }
    double get_viscosity() const { return viscosity; }
    int get_refinements() const { return refinements; }
    int get_order() const { return order; }
    int get_visualisation() const { return visualisation; }
    std::string get_mesh() const { return mesh; }
    std::string get_outputfile() const { return outputfile; }

    // Methods to interface with the dynamic library
    void velocity_data(const mfem::Vector &x, mfem::Vector &out)
    {
        out.SetSize(x.Size());
        velocity_data_func(x.GetData(), 0., out.GetData(), x.Size());
    }

    void boundary_data_u(const mfem::Vector &x, double t, mfem::Vector &out)
    {
        out.SetSize(x.Size());
        boundary_data_u_func(x.GetData(), t, out.GetData(), x.Size());
    }

    void initial_data_u(const mfem::Vector &x, mfem::Vector &out)
    {
        out.SetSize(x.Size());
        initial_data_u_func(x.GetData(), out.GetData(), x.Size());
    }

    mfem::real_t boundary_data_A(const mfem::Vector &x, double t)
    {
        double val;
        boundary_data_A_func(x.GetData(), t, &val, x.Size());
        return val;
    }

    mfem::real_t initial_data_A(const mfem::Vector &x)
    {
        double out;
        initial_data_A_func(x.GetData(), &out, x.Size());
        return out;
    }

private:
    // Configuration parameters loaded from the JSON file
    std::string mesh;
    std::string outputfile;
    double dt;
    double T;
    double viscosity;
    int refinements;
    int order;
    int visualisation;

    // Function pointers for the dynamically loaded functions
    typedef void (*InitialDataFunc)(double *, double *, int);
    typedef void (*VelocityDataFunc)(double *, double, double *, int);
    typedef void (*BoundaryDataFunc)(double *, double, double *, int);

    InitialDataFunc initial_data_u_func, initial_data_A_func;
    VelocityDataFunc velocity_data_func;
    BoundaryDataFunc boundary_data_u_func, boundary_data_A_func;

    // Code snippets for custom functions loaded from the JSON
    std::string velocity_code;
    std::string boundary_data_u_code;
    std::string initial_data_u_code;
    std::string boundary_data_A_code;
    std::string initial_data_A_code;

    // Method to generate code and load the dynamic library
    void PrepareCodeAndLibrary()
    {
        // Generate C++ code that includes the user-provided function code
        std::ofstream file("generated_initial_condition.cpp");
        file << R"(
            #include <cmath>
        
            extern "C" {                
                void initial_data_u(double* x, double* out, int dim) {
                    )" +
                    initial_data_u_code + R"(
                }

                void initial_data_A(double* x, double* out, int dim) {
                    )" +
                    initial_data_A_code + R"(
                }
        
                void velocity_data(double* x, double t, double *out, int dim) {
                    )" +
                    velocity_code + R"(
                }
        
                void boundary_data_u(double* x, double t, double* out, int dim) {
                    )" +
                    boundary_data_u_code + R"(
                }

                void boundary_data_A(double* x, double t, double* out, int dim) {
                    )" +
                    boundary_data_A_code + R"(
                }
            }
        )";
        file.close();

        // Compile the generated code into a shared library
        const char *compile_command = "g++ -shared -fPIC -o libinitial_condition.so generated_initial_condition.cpp";
        int result = std::system(compile_command);

        // Check if compilation was successful
        if (result != 0)
        {
            std::cerr << "Failed to compile the initial condition library!" << std::endl;
            return;
        }

        // Load the compiled shared library
        void *handle = dlopen("libinitial_condition.so", RTLD_LAZY);
        if (!handle)
        {
            std::cerr << "Failed to load library: " << dlerror() << std::endl;
            return;
        }

        // Load the function pointers for initial_data, velocity_data, and boundary_data
        initial_data_u_func = (InitialDataFunc)dlsym(handle, "initial_data_u");
        if (!initial_data_u_func)
        {
            std::cerr << "Failed to load initial data function: " << dlerror() << std::endl;
            dlclose(handle);
            return;
        }

        initial_data_A_func = (InitialDataFunc)dlsym(handle, "initial_data_A");
        if (!initial_data_A_func)
        {
            std::cerr << "Failed to load initial data function: " << dlerror() << std::endl;
            dlclose(handle);
            return;
        }

        velocity_data_func = (VelocityDataFunc)dlsym(handle, "velocity_data");
        if (!velocity_data_func)
        {
            std::cerr << "Failed to load velocity function: " << dlerror() << std::endl;
            dlclose(handle);
            return;
        }

        boundary_data_u_func = (BoundaryDataFunc)dlsym(handle, "boundary_data_u");
        if (!boundary_data_u_func)
        {
            std::cerr << "Failed to load boundary data function: " << dlerror() << std::endl;
            dlclose(handle);
            return;
        }

        boundary_data_A_func = (BoundaryDataFunc)dlsym(handle, "boundary_data_A");
        if (!boundary_data_A_func)
        {
            std::cerr << "Failed to load boundary data function: " << dlerror() << std::endl;
            dlclose(handle);
            return;
        }
    }
};
