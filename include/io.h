#include "mfem.hpp"                            // For MFEM types like mfem::Vector
#include <boost/property_tree/ptree.hpp>       // For Boost Property Tree
#include <boost/property_tree/json_parser.hpp> // For Boost JSON parsing
#include <iostream>                            // For std::cerr and I/O
#include <fstream>                             // For file output
#include <cstdlib>                             // For std::system
#include <dlfcn.h>                             // For dlopen, dlsym, dlclose
#include <mpi.h>                               // For MPI_Comm, MPI_Barrier

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
        : dt(0.0), T(0.0), viscosity(0.0), refinements(0), order(1), visualisation(0), printlevel(0), tol(1e-8), initial_data_u_func(nullptr), boundary_data_u_func(nullptr), force_data_func(nullptr), initial_data_w_func(nullptr), lib_handle(nullptr)
    {
        // Read the JSON file into a property tree
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(filename, tree);

        int depth = 1;
        PrintTree(tree, depth);

        // Populate member variables from the JSON configuration
        mesh = tree.get<std::string>("mesh");
        outputfile = tree.get<std::string>("outputfile");
        solver = tree.get<std::string>("solver");
        dt = tree.get<double>("dt");
        T = tree.get<double>("T");
        refinements = tree.get<int>("refinements", 0);
        order = tree.get<int>("order", 1);
        visualisation = tree.get<int>("visualisation", 0);
        tol = tree.get<double>("tol", 1e-8);

        boundary_data_u_code = tree.get<std::string>("boundary_data_u", "");
        // BUG FIX: this was overwriting boundary_data_u_code before
        force_data_code = tree.get<std::string>("force_data", "");
        initial_data_u_code = tree.get<std::string>("initial_data_u", "");
        initial_data_w_code = tree.get<std::string>("initial_data_w", "");
        exact_data_u_code = tree.get<std::string>("exact_data_u", "");
        has_exact_u_solution = !exact_data_u_code.empty();

        viscosity = tree.get<double>("viscosity", 0);
        printlevel = tree.get<int>("printlevel", 0);

        // NOTE: we DO NOT compile/load the library in the constructor anymore.
        // Call InitializeLibrary(rank, comm) after MPI_Init on all ranks.
    }

    ~SimulationConfig()
    {
        if (lib_handle)
        {
            dlclose(lib_handle);
            lib_handle = nullptr;
        }
    }

    // Must be called by ALL ranks, AFTER MPI_Init.
    // rank 0: generates and compiles libinitial_condition.so
    // all ranks: wait, then dlopen + dlsym
    void InitializeLibrary(int rank, MPI_Comm comm)
    {
        if (rank == 0)
        {
            // Generate C++ code that includes the user-provided function code
            std::ofstream file("generated_initial_condition.cpp");
            file <<
                R"(#include <cmath>

extern "C" {

    void initial_data_u(double* x, double* out, int dim) {
)"
                 << initial_data_u_code <<
                R"(
    }

    void initial_data_w(double* x, double* out, int dim) {
)"
                 << initial_data_w_code <<
                R"(
    }

    void boundary_data_u(double* x, double t, double* out, int dim) {
)"
                 << boundary_data_u_code <<
                R"(
    }

    void force_data(double* x, double t, double* out, int dim) {
)"
                 << force_data_code <<
                R"(
    }
)";
            file <<
                R"(    void exact_data_u(double* x, double t, double* out, int dim) {
)";
            if (has_exact_u_solution)
            {
                file << exact_data_u_code << "\n";
            }
            else
            {
                // Safe default if accidentally called
                file << "for (int i = 0; i < dim; ++i) out[i] = 0.0;\n";
            }
            file <<
                R"(    }

} // extern "C"
)";
            file.close();

            // Compile the generated code into a shared library
            const char *compile_command =
                "g++ -shared -fPIC -o libinitial_condition.so generated_initial_condition.cpp";
            int result = std::system(compile_command);

            if (result != 0)
            {
                std::cerr << "Rank 0: Failed to compile the initial condition library! "
                          << "(exit code " << result << ")\n";
                // Still continue to the barrier so other ranks don't hang.
            }
        }

        // Ensure all ranks wait until rank 0 is done writing & compiling
        MPI_Barrier(comm);

        // Load the compiled shared library (all ranks)
        lib_handle = dlopen("./libinitial_condition.so", RTLD_LAZY);
        if (!lib_handle)
        {
            std::cerr << "Rank " << rank
                      << ": Failed to load library: " << dlerror() << std::endl;
            return;
        }

        // Load the function pointers
        initial_data_u_func =
            reinterpret_cast<SpaceDataFunc>(dlsym(lib_handle, "initial_data_u"));
        if (!initial_data_u_func)
        {
            std::cerr << "Rank " << rank
                      << ": Failed to load initial_data_u: " << dlerror() << std::endl;
            dlclose(lib_handle);
            lib_handle = nullptr;
            return;
        }

        initial_data_w_func =
            reinterpret_cast<SpaceDataFunc>(dlsym(lib_handle, "initial_data_w"));
        if (!initial_data_w_func)
        {
            std::cerr << "Rank " << rank
                      << ": Failed to load initial_data_w: " << dlerror() << std::endl;
            dlclose(lib_handle);
            lib_handle = nullptr;
            return;
        }

        boundary_data_u_func =
            reinterpret_cast<SpaceTimeDataFunc>(dlsym(lib_handle, "boundary_data_u"));
        if (!boundary_data_u_func)
        {
            std::cerr << "Rank " << rank
                      << ": Failed to load boundary_data_u: " << dlerror() << std::endl;
            dlclose(lib_handle);
            lib_handle = nullptr;
            return;
        }

        force_data_func =
            reinterpret_cast<SpaceTimeDataFunc>(dlsym(lib_handle, "force_data"));
        if (!force_data_func)
        {
            std::cerr << "Rank " << rank
                      << ": Failed to load force_data: " << dlerror() << std::endl;
            dlclose(lib_handle);
            lib_handle = nullptr;
            return;
        }

        exact_data_u_func =
            reinterpret_cast<SpaceTimeDataFunc>(dlsym(lib_handle, "exact_data_u"));
        if (!force_data_func)
        {
            std::cerr << "Rank " << rank
                      << ": Failed to load exact_data_u: " << dlerror() << std::endl;
            dlclose(lib_handle);
            lib_handle = nullptr;
            return;
        }
    }

    // Getter methods for configuration parameters
    double get_dt() const { return dt; }
    double get_T() const { return T; }
    double get_viscosity() const { return viscosity; }
    int get_refinements() const { return refinements; }
    int get_order() const { return order; }
    int get_visualisation() const { return visualisation; }
    int get_printlevel() const { return printlevel; }
    double get_tol() const { return tol; }
    std::string get_mesh() const { return mesh; }
    std::string get_outputfile() const { return outputfile; }
    std::string get_solver() const { return solver; }
    bool has_exact_u() const { return has_exact_u_solution; }

    // Methods to interface with the dynamic library
    void boundary_data_u(const mfem::Vector &x, double t, mfem::Vector &out)
    {
        out.SetSize(x.Size());
        if (!boundary_data_u_func)
        {
            std::cerr << "boundary_data_u_func is null (library not initialized?)\n";
            out = 0.0;
            return;
        }
        boundary_data_u_func(x.GetData(), t, out.GetData(), x.Size());
    }

    void force_data(const mfem::Vector &x, double t, mfem::Vector &out)
    {
        out.SetSize(x.Size());
        if (!force_data_func)
        {
            std::cerr << "force_data_func is null (library not initialized?)\n";
            out = 0.0;
            return;
        }
        force_data_func(x.GetData(), t, out.GetData(), x.Size());
    }

    void initial_data_u(const mfem::Vector &x, mfem::Vector &out)
    {
        out.SetSize(x.Size());
        if (!initial_data_u_func)
        {
            std::cerr << "initial_data_u_func is null (library not initialized?)\n";
            out = 0.0;
            return;
        }
        initial_data_u_func(x.GetData(), out.GetData(), x.Size());
    }

    void initial_data_w(const mfem::Vector &x, mfem::Vector &out)
    {
        out.SetSize(x.Size());
        if (!initial_data_w_func)
        {
            std::cerr << "initial_data_w_func is null (library not initialized?)\n";
            out = 0.0;
            return;
        }
        initial_data_w_func(x.GetData(), out.GetData(), x.Size());
    }
    void exact_data_u(const mfem::Vector &x, double t, mfem::Vector &out)
    {
        out.SetSize(x.Size());
        if (!exact_data_u_func)
        {
            std::cerr << "exact_data_u is null (library not initialized?)\n";
            return;
        }
        exact_data_u_func(x.GetData(), t, out.GetData(), x.Size());
    }

private:
    // Configuration parameters loaded from the JSON file
    std::string mesh;
    std::string outputfile;
    std::string solver;
    double dt;
    double T;
    double viscosity;
    int refinements;
    int order;
    int visualisation;
    int printlevel;
    double tol;

    // Function pointers for the dynamically loaded functions
    typedef void (*SpaceDataFunc)(double *, double *, int);
    typedef void (*SpaceTimeDataFunc)(double *, double, double *, int);

    SpaceDataFunc initial_data_u_func;
    SpaceTimeDataFunc boundary_data_u_func;
    SpaceTimeDataFunc force_data_func;
    SpaceDataFunc initial_data_w_func;
    SpaceTimeDataFunc exact_data_u_func;

    bool has_exact_u_solution;

    void *lib_handle; // handle from dlopen

    // Code snippets for custom functions loaded from the JSON
    std::string boundary_data_u_code;
    std::string force_data_code;
    std::string initial_data_u_code;
    std::string initial_data_w_code;
    std::string exact_data_u_code;
};

class EnergyCSVLogger
{
public:
    EnergyCSVLogger(SimulationConfig &config,
                    mfem::Operator &M_op,
                    mfem::Operator &N_op,
                    mfem::GridFunction &u,
                    mfem::GridFunction &v,
                    mfem::GridFunction &w,
                    mfem::GridFunction &z,
                    int &num_it_A1,
                    int &num_it_A2)
        : M_op_(M_op),
          N_op_(N_op),
          u_(u), v_(v), w_(w), z_(z),
          M_u_(u.Size()),
          M_w_(w.Size()),
          N_v_(v.Size()),
          N_z_(z.Size()),
          num_it_A1_(num_it_A1),
          num_it_A2_(num_it_A2),
          config_(config)
    {
        std::string output_file = config.get_outputfile();
        std::string csv_path = std::string("./out/") + output_file + std::string("_vars.csv");
        csv_ = std::ofstream(csv_path, std::ios::out);

        if (!csv_)
        {
            std::cerr << "[warn] Failed to open CSV (for truncation): "
                      << csv_path << std::endl;
            return;
        }

        std::cout << "[info] CSV opened (truncated): " << csv_path << std::endl;
        csv_ << "cycle,time_full,time_half,num_it_A1,num_it_A2,||u1||,||u2||,u1*w1,u2*w2";
        if (config.has_exact_u())
        {
            csv_ << ",u1_err_L2,u2_err_L2";
        }
        csv_ << std::endl;
        csv_.flush();
    }

    bool IsOpen() const { return csv_.is_open(); }

    void WriteRow(int cycle, double t_full, double t_half)
    {
        if (!csv_)
        {
            return;
        }

        // Matrix-free applications
        M_op_.Mult(u_, M_u_); // M u
        M_op_.Mult(w_, M_w_); // M w
        N_op_.Mult(v_, N_v_); // N v
        N_op_.Mult(z_, N_z_); // N z

        // Inner products as dot products
        double u1_norm = u_ * M_u_; // (u, M u)
        double u2_norm = v_ * N_v_; // (v, N v)
        double u1w1 = u_ * M_w_;    // (u, M w)
        double u2w2 = v_ * N_z_;    // (v, N z)

        csv_ << cycle << ","
             << std::setprecision(15) << std::fixed
             << t_full << "," << t_half << ","
             << num_it_A1_ << "," << num_it_A2_ << ","
             << u1_norm << "," << u2_norm << ","
             << u1w1 << "," << u2w2;
        if (config_.has_exact_u())
        {
            std::function<void(const mfem::Vector &, double, mfem::Vector &)> exact_data_u =
                std::bind(&SimulationConfig::exact_data_u, &config_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
            mfem::VectorFunctionCoefficient u_exact_coeff(3, exact_data_u);
            csv_ << "," << u_.ComputeL2Error(u_exact_coeff) << "," << v_.ComputeL2Error(u_exact_coeff);
        }
        csv_ << std::endl;
        csv_.flush();
    }

private:
    mfem::Operator &M_op_;
    mfem::Operator &N_op_;
    mfem::GridFunction &u_;
    mfem::GridFunction &v_;
    mfem::GridFunction &w_;
    mfem::GridFunction &z_;
    int &num_it_A1_;
    int &num_it_A2_;

    mfem::Vector M_u_, M_w_, N_v_, N_z_;
    std::ofstream csv_;
    SimulationConfig &config_;
};
