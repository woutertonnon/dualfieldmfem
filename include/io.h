#include "mfem.hpp"                            // For MFEM types like mfem::Vector
#include <boost/property_tree/ptree.hpp>       // For Boost Property Tree
#include <boost/property_tree/json_parser.hpp> // For Boost JSON parsing
#include <iostream>                            // For std::cerr and I/O
#include <fstream>                             // For file output
#include <cstdlib>                             // For std::system
#include <dlfcn.h>                             // For dlopen, dlsym, dlclose
// #include <mpi.h>                               // For MPI_Comm, MPI_Barrier
#include <chrono>
#include <filesystem>
#include <typeindex>
#include <unistd.h>
#include <stdio.h>

class SimulationConfig
{
public:
    SimulationConfig(const std::string &filename)
    {
        boost::property_tree::read_json(filename, tree_);
    }

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

    boost::property_tree::ptree &get_tree() { return tree_; };

    template <typename T>
    T get_value(std::string variable, const T default_value) { return tree_.get<T>(variable.data(), default_value); };

    template <typename T>
    T get_value(std::string variable) { return tree_.get<T>(variable.data()); };

    std::function<void(const mfem::Vector &, double, mfem::Vector &)>& get_exact_data(std::string function_name) { return functions_.at(function_name); };

    void InitializeLibrary(std::initializer_list<std::string> function_names)
    {
        // Get job ID to get unique name
        auto pid = getpid();
        library_name_ = std::string("config_library_") + std::to_string(pid); // + std::string(".cpp");
        std::string library_name_cpp = library_name_ + std::string(".cpp");
        std::string library_name_so  = library_name_ + std::string(".so");
        // Generate C++ code that includes the user-provided function code
        std::ofstream file(library_name_cpp);
        file << R"(#include <cmath>

                   extern "C" {)";
        for (std::string function_name : function_names)
            file << "void " << function_name << R"((double* x, double t, double* out, int dim){)" << tree_.get<std::string>(function_name, "") << R"(};)";

        file << R"(} // extern "C")";
        file.close();

        std::string cmd = "g++ -O2 -fPIC -shared -o " + library_name_so + " " + library_name_cpp;
        int rc = std::system(cmd.c_str());
        if (rc != 0)
        {
            std::cerr << "Failed to compile generated library. cmd: " << cmd << "\n";
            return;
        }

        lib_handle_ = dlopen(library_name_so.data(), RTLD_LAZY);
        if (!lib_handle_)
        {
            std::cerr << ": Failed to load library: " << dlerror() << std::endl;
            return;
        }

        // Load the function pointers
        for (std::string function_name : function_names)
        {
            lib_func_handles_.insert({function_name, reinterpret_cast<SpaceTimeDataFunc>(dlsym(lib_handle_, function_name.data()))});
            if (!lib_func_handles_.at(function_name))
            {
                std::cerr << ": Failed to load initial_data_u: " << dlerror() << std::endl;
                dlclose(lib_handle_);
                lib_handle_ = nullptr;
                return;
            }
            functions_.insert({function_name,
                               [this, function_name](const mfem::Vector &x, double t, mfem::Vector &v)
                               {
                                   this->lib_func_handles_.at(function_name)(x.GetData(), t, v.GetData(), x.Size());
                                   return;
                               }});
        }
	    for (const auto& [key, value] : functions_) 
        std::cout << key << " => "  << '\n';
    }

    ~SimulationConfig()
    {
        if (lib_handle_)
        {
            dlclose(lib_handle_);
            lib_handle_ = nullptr;
        }
        std::filesystem::remove(library_name_+std::string(".so"));
        std::filesystem::remove(library_name_+std::string(".cpp"));
    }

private:
    boost::property_tree::ptree tree_;
    std::string library_name_;
    void *lib_handle_;

    typedef void (*SpaceTimeDataFunc)(double *, double, double *, int);
    std::map<std::string, SpaceTimeDataFunc> lib_func_handles_;
    std::map<std::string, std::function<void(const mfem::Vector &, double, mfem::Vector &)>> functions_;
};

// Class to manage the simulation configuration
class DualFieldConfig : public SimulationConfig
{
public:
    // Constructor that reads configuration from a JSON file
    DualFieldConfig(const std::string &filename)
        : SimulationConfig(filename),
          mesh(get_value<std::string>("mesh")),
          outputfile(get_value<std::string>("outputfile")),
          solver(get_value<std::string>("solver")),
          dt(get_value("dt", 0.02)),
          T(get_value("T", 1.)),
          refinements(get_value("refinements", 10)),
          order(get_value("order", 1)),
          visualisation(get_value("visualisation", 0)),
          tol(get_value("tol", 1e-8)),
          viscosity(get_value("viscosity", 0.)),
          printlevel(get_value("printlevel", 0)),
          has_exact_u_solution(!get_value<std::string>("exact_data_u", "").empty())
    {
        std::initializer_list<std::string> function_names({"force_data", "initial_data_u", "initial_data_w", "exact_data_u", "exact_data_w"});
        InitializeLibrary(function_names);
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
    std::string get_outputfile() { return outputfile; }
    std::string get_solver() const { return solver; }
    bool has_exact_u() const { return has_exact_u_solution; }

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
    bool has_exact_u_solution;
};

class NitscheStokesConfig : public SimulationConfig
{
public:
    // Constructor that reads configuration from a JSON file
    NitscheStokesConfig(const std::string &filename)
        : SimulationConfig(filename),
          mesh(get_value<std::string>("mesh")),
          outputfile(get_value<std::string>("outputfile")),
          solver(get_value<std::string>("solver")),
          refinements(get_value("refinements", 10)),
          order(get_value("order", 1)),
          visualisation(get_value("visualisation", 0)),
          tol(get_value("tol", 1e-8)),
          mass(get_value("mass",0.)),
          viscosity(get_value("viscosity", 0.)),
          printlevel(get_value("printlevel", 0)),
          has_exact_u_solution(!get_value<std::string>("exact_data_u", "").empty())
    {
        std::initializer_list<std::string> function_names({"force_data", "exact_data_u"});
        InitializeLibrary(function_names);
    }

    // Getter methods for configuration parameters
    double get_mass() const {return mass;}
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

private:
    // Configuration parameters loaded from the JSON file
    std::string mesh;
    std::string outputfile;
    std::string solver;
    double dt;
    double T;
    double mass;
    double viscosity;
    int refinements;
    int order;
    int visualisation;
    int printlevel;
    double tol;
    bool has_exact_u_solution;
};

class CSVLogger
{
public:
    CSVLogger(std::string output_file)
    {
        std::string csv_path = std::string("./out/data/") + output_file + std::string("_vars.csv");
        std::filesystem::path dir = std::filesystem::path(csv_path).parent_path();
        if (!std::filesystem::exists(dir))
        {
            throw std::runtime_error("Directory does not exist: " + dir.string());
        }
        csv_ = std::ofstream(csv_path, std::ios::out);
        if (!csv_)
        {
            std::cerr << "[warn] Failed to open CSV (for truncation): "
                      << csv_path << std::endl;
            return;
        }
        std::cout << "[info] CSV opened (truncated): " << csv_path << std::endl;
    };

    bool IsOpen() const { return csv_.is_open(); }

    std::ofstream &get_ofstream()
    {

        if (!csv_)
            std::runtime_error("[warn] Failed to open CSV;\n");
        return csv_;
    };

    double MatrixConservedVariable(mfem::Operator &M, mfem::Vector u)
    {
        if (M.Height() != M.Width())
        {
            throw std::runtime_error("CSVLogger::MatrixConservedVariable(): Matrix not symmetric!");
        }
        mfem::Vector M_u(u.Size());
        M.Mult(u, M_u);
        return u * M_u;
    }

    double MatrixConservedVariable(mfem::Vector v, mfem::Operator &M, mfem::Vector u)
    {
        if (M.Height() != v.Size() || M.Width() != u.Size())
        {
            throw std::runtime_error("CSVLogger::MatrixConservedVariable(): Dimensions of v, M, and u do not match!");
        }
        mfem::Vector M_u(M.Height());
        M.Mult(u, M_u);
        return v * M_u;
    }

    virtual void WriteRow() = 0;

private:
    std::ofstream csv_;
};

class DualFieldCSVLogger : public CSVLogger
{
public:
    DualFieldCSVLogger(DualFieldConfig &config,
                       int &cycle,
                       double &t_full,
                       double &t_half,
                       mfem::Operator &M_op,
                       mfem::Operator &N_op,
                       mfem::GridFunction &u,
                       mfem::GridFunction &v,
                       mfem::GridFunction &w,
                       mfem::GridFunction &z,
                       int &num_it_A1,
                       int &num_it_A2)
        : CSVLogger(config.get_outputfile()),
          cycle_(cycle),
          t_full_(t_full),
          t_half_(t_half),
          M_op_(M_op),
          N_op_(N_op),
          u_(u), v_(v), w_(w), z_(z),
          num_it_A1_(num_it_A1),
          num_it_A2_(num_it_A2),
          time_(std::chrono::time_point(std::chrono::steady_clock::now())),
          config_(config)
    {
        get_ofstream() << "runtime_it,cycle,time_full,time_half,num_it_A1,num_it_A2,||u1||,||u2||,u1*w1,u2*w2";
        if (config.has_exact_u())
        {
            get_ofstream() << ",u1_err_L2,u2_err_L2";
        }
        get_ofstream() << std::endl;
        get_ofstream().flush();
    };

    void WriteRow()
    {
        // Inner products as dot products
        double u1_norm = MatrixConservedVariable(M_op_, u_);  // (u, M u)
        double u2_norm = MatrixConservedVariable(N_op_, v_);  // (v, N v)
        double u1w1 = MatrixConservedVariable(u_, M_op_, w_); // (u, M w)
        double u2w2 = MatrixConservedVariable(v_, N_op_, z_); // (v, N z)

        std::chrono::duration<double> runtime_it = std::chrono::steady_clock::now() - time_;
        time_ = std::chrono::steady_clock::now();

        get_ofstream() << runtime_it.count() << "," << cycle_ << ","
                       << std::setprecision(15) << std::fixed
                       << t_full_ << "," << t_half_ << ","
                       << num_it_A1_ << "," << num_it_A2_ << ","
                       << u1_norm << "," << u2_norm << ","
                       << u1w1 << "," << u2w2;
        if (config_.has_exact_u())
        {
            mfem::VectorFunctionCoefficient u_exact_coeff(3, config_.get_exact_data("exact_data_u"));
            u_exact_coeff.SetTime(t_full_);
            get_ofstream() << "," << u_.ComputeL2Error(u_exact_coeff) << "," << v_.ComputeL2Error(u_exact_coeff);
        }
        get_ofstream() << std::endl;
        get_ofstream().flush();
    };

private:
    int &cycle_;
    double &t_full_;
    double &t_half_;
    mfem::Operator &M_op_;
    mfem::Operator &N_op_;
    mfem::GridFunction &u_;
    mfem::GridFunction &v_;
    mfem::GridFunction &w_;
    mfem::GridFunction &z_;
    int &num_it_A1_;
    int &num_it_A2_;
    std::chrono::time_point<std::chrono::steady_clock> time_;
    DualFieldConfig config_;
};

class NitscheStokesCSVLogger : public CSVLogger
{
public:
    NitscheStokesCSVLogger(NitscheStokesConfig &config,
                           mfem::GridFunction &u,
                           int &num_it_solver)
        : CSVLogger(config.get_outputfile()),
          config_(config),
          u_(u),
          num_it_solver_(num_it_solver),
          time_(std::chrono::time_point(std::chrono::steady_clock::now()))
    {
        get_ofstream() << "runtime_it,num_it_solver";
        if (config.has_exact_u())
        {
            get_ofstream() << ",u1_err_L2";
        }
        get_ofstream() << std::endl;
        get_ofstream().flush();
    };

    void WriteRow()
    {
        std::chrono::duration<double> runtime_it = std::chrono::steady_clock::now() - time_;
        time_ = std::chrono::steady_clock::now();

        get_ofstream() << runtime_it.count() << "," << num_it_solver_;
        if (config_.has_exact_u())
        {
            mfem::VectorFunctionCoefficient u_exact_coeff(3, config_.get_exact_data("exact_data_u"));
            get_ofstream() << "," << u_.ComputeL2Error(u_exact_coeff);
        }
        get_ofstream() << std::endl;
        get_ofstream().flush();
    };

private:
    NitscheStokesConfig config_;
    mfem::GridFunction &u_;
    int &num_it_solver_;
    std::chrono::time_point<std::chrono::steady_clock> time_;
};
