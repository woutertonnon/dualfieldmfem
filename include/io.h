#include "mfem.hpp"                            // For MFEM types like mfem::Vector
#include <boost/property_tree/ptree.hpp>       // For Boost Property Tree
#include <boost/property_tree/json_parser.hpp> // For Boost JSON parsing
#include <iostream>                            // For std::cerr and I/O
#include <fstream>                             // For file output
#include <cstdlib>                             // For std::system
#include <dlfcn.h>                             // For dlopen, dlsym, dlclose
// #include <mpi.h>                               // For MPI_Comm, MPI_Barrier
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
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
    const boost::property_tree::ptree &get_tree_const() const { return tree_; };

    template <typename T>
    T get_value(std::string variable, const T default_value) const { return get_tree_const().get<T>(variable.data(), default_value); };

    template <typename T>
    T get_value(std::string variable) const { return get_tree_const().get<T>(variable.data()); };

    std::function<void(const mfem::Vector &, double, mfem::Vector &)>& get_exact_data(std::string function_name) { 
        try{ 
            return functions_.at(function_name); } 
        catch (const std::out_of_range& e) {
            throw std::runtime_error("get_exact_data(): data not found: " + function_name);
        }
    };

    void InitializeLibrary(std::initializer_list<std::string> function_names)
    {
        CleanupGeneratedLibraryArtifacts();

        auto pid = getpid();
        const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        static std::atomic<std::uint64_t> counter{0};
        const auto serial = counter.fetch_add(1, std::memory_order_relaxed);

        std::error_code ec;
        std::filesystem::create_directories("./tmp", ec);
        if (ec)
        {
            throw std::runtime_error("Failed to create ./tmp for generated config library: " + ec.message());
        }

        const std::string library_stem =
            std::string("config_library_") + std::to_string(pid) +
            "_" + std::to_string(now) + "_" + std::to_string(serial);
        library_cpp_path_ = std::filesystem::path("./tmp") / (library_stem + ".cpp");
        library_so_path_  = std::filesystem::path("./tmp") / (library_stem + ".so");

        // Generate C++ code that includes the user-provided function code
        std::ofstream file(library_cpp_path_);
        if (!file)
        {
            CleanupGeneratedLibraryArtifacts();
            throw std::runtime_error("Failed to open generated source file: " + library_cpp_path_.string());
        }
        file << R"(#include <cmath>

                   extern "C" {)";
        for (std::string function_name : function_names)
            file << "void " << function_name << R"((double* x, double t, double* out, int dim){)" << tree_.get<std::string>(function_name, "") << R"(};)";

        file << R"(} // extern "C")";
        file.close();

        std::string cmd = "g++ -O2 -fPIC -shared -o \"" + library_so_path_.string() + "\" \"" + library_cpp_path_.string() + "\"";
        int rc = std::system(cmd.c_str());
        if (rc != 0)
        {
            CleanupGeneratedLibraryArtifacts();
            throw std::runtime_error("Failed to compile generated library. cmd: " + cmd);
        }

        lib_handle_ = dlopen(library_so_path_.c_str(), RTLD_LAZY);
        if (!lib_handle_)
        {
            const char *err = dlerror();
            CleanupGeneratedLibraryArtifacts();
            throw std::runtime_error(std::string("Failed to load generated library: ") + (err ? err : "unknown"));
        }

        // Load the function pointers
        for (std::string function_name : function_names)
        {
            lib_func_handles_.insert({function_name, reinterpret_cast<SpaceTimeDataFunc>(dlsym(lib_handle_, function_name.data()))});
            if (!lib_func_handles_.at(function_name))
            {
                const char *err = dlerror();
                CleanupGeneratedLibraryArtifacts();
                throw std::runtime_error(std::string("Failed to load generated symbol '") + function_name + "': " + (err ? err : "unknown"));
            }
            functions_.insert({function_name,
                               [this, function_name](const mfem::Vector &x, double t, mfem::Vector &v)
                               {
                                   this->lib_func_handles_.at(function_name)(x.GetData(), t, v.GetData(), x.Size());
                                    return;
                                }});
        }
    }

    ~SimulationConfig()
    {
        CleanupGeneratedLibraryArtifacts();
    }

private:
    void CleanupGeneratedLibraryArtifacts() noexcept
    {
        if (lib_handle_)
        {
            dlclose(lib_handle_);
            lib_handle_ = nullptr;
        }

        std::error_code ec;
        if (!library_so_path_.empty())
        {
            std::filesystem::remove(library_so_path_, ec);
        }
        if (!library_cpp_path_.empty())
        {
            std::filesystem::remove(library_cpp_path_, ec);
        }
    }

    boost::property_tree::ptree tree_;
    std::filesystem::path library_cpp_path_;
    std::filesystem::path library_so_path_;
    void *lib_handle_ = nullptr;

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
          tol(get_value("tol", 1e-10)),
          viscosity(get_value("viscosity", 0.)),
          printlevel(get_value("printlevel", 0)),
          has_exact_u_solution(!get_value<std::string>("exact_data_u", "").empty())
    {
        std::initializer_list<std::string> function_names({"force_data", "initial_data_u", "initial_data_w", "exact_data_u", "exact_data_w", "boundary_data_u"});
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

    /// True when the config specifies lid_attributes (attribute-based BCs).
    bool has_lid_attributes() const {
        auto child = get_tree_const().get_child_optional("lid_attributes");
        return child.has_value();
    }

    /// Build an MFEM boundary-attribute marker from lid_attributes.
    /// marker[i-1] = 1 for every attribute i listed in the JSON array.
    /// Size is max_bdr_attr (typically mesh.bdr_attributes.Max()).
    mfem::Array<int> get_lid_marker(int max_bdr_attr) const {
        mfem::Array<int> marker(max_bdr_attr);
        marker = 0;
        for (auto &item : get_tree_const().get_child("lid_attributes"))
        {
            int attr = item.second.get_value<int>();
            if (attr >= 1 && attr <= max_bdr_attr)
                marker[attr - 1] = 1;
        }
        return marker;
    }

    /// True when the config specifies outflow_attributes (consistent-Nitsche
    /// "do-nothing" pressure outflow; e.g. the channel outlet).
    bool has_outflow_attributes() const {
        auto child = get_tree_const().get_child_optional("outflow_attributes");
        return child.has_value();
    }

    /// Boundary-attribute marker for the outflow faces (same convention as
    /// get_lid_marker).
    mfem::Array<int> get_outflow_marker(int max_bdr_attr) const {
        mfem::Array<int> marker(max_bdr_attr);
        marker = 0;
        if (!has_outflow_attributes())
            return marker;
        for (auto &item : get_tree_const().get_child("outflow_attributes"))
        {
            int attr = item.second.get_value<int>();
            if (attr >= 1 && attr <= max_bdr_attr)
                marker[attr - 1] = 1;
        }
        return marker;
    }

    /// Pressure penalty gamma for the consistent-Nitsche outflow (tunable;
    /// larger => p closer to 0 and u.n freer at the outlet).
    double get_outflow_penalty() const {
        return get_value("outflow_penalty", 1.0);
    }

    /// Flow-around-cylinder QoI: boundary attribute of the cylinder (0 disables
    /// the drag/lift/pressure-drop output), mean inflow speed Ubar and diameter
    /// D used to non-dimensionalise the force coefficients.
    int    get_qoi_cylinder_attribute() const {
        return get_value("qoi_cylinder_attribute", 0);
    }
    double get_qoi_Ubar() const { return get_value("qoi_Ubar", 1.0); }
    double get_qoi_diameter() const { return get_value("qoi_diameter", 0.1); }

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
          tol(get_value("tol", 1e-10)),
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
                       mfem::FiniteElementSpace *ND,
                       mfem::FiniteElementSpace *RT,
                       mfem::GridFunction &u1,
                       mfem::GridFunction &u2,
                       mfem::GridFunction &w1,
                       int &num_it_A1,
                       int &num_it_A2)
        : CSVLogger(config.get_outputfile()),
          cycle_(cycle),
          t_full_(t_full),
          t_half_(t_half),
          ND(ND),
          RT(RT),
          mass_ND_(ND),
          mass_RT_(RT),
          mass_curl_RT_(RT,ND),
          u_(u1), v_(u2), w_(w1),
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

        mass_ND_.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
        mass_ND_.Assemble();
        mass_ND_.Finalize();

        mass_RT_.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
        mass_RT_.Assemble();
        mass_RT_.Finalize();

        mass_curl_RT_.AddDomainIntegrator(new mfem::MixedVectorWeakCurlIntegrator());
        mass_curl_RT_.Assemble();
        mass_curl_RT_.Finalize();



    };

    void WriteRow()
    {
        // Inner products as dot products
        double u1_norm = MatrixConservedVariable(mass_ND_, u_);  // (u, M u)
        double u2_norm = MatrixConservedVariable(mass_RT_, v_);  // (v, N v)
        double u1w1 = MatrixConservedVariable(u_, mass_ND_, w_); // (u, M w)
        double u2w2 = MatrixConservedVariable(u_, mass_curl_RT_, v_); // (v, N z)

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
            mfem::VectorFunctionCoefficient u1_exact_coeff(3, config_.get_exact_data("exact_data_u"));
            u1_exact_coeff.SetTime(t_full_);

            mfem::VectorFunctionCoefficient u2_exact_coeff(3, config_.get_exact_data("exact_data_u"));
            u2_exact_coeff.SetTime(t_half_);

            get_ofstream() << "," << u_.ComputeL2Error(u1_exact_coeff)
                           << "," << v_.ComputeL2Error(u2_exact_coeff);
        }
        get_ofstream() << std::endl;
        get_ofstream().flush();
    };

private:
    int &cycle_;
    double &t_full_;
    double &t_half_;
    mfem::FiniteElementSpace *ND, *RT;
    mfem::BilinearForm mass_ND_;
    mfem::BilinearForm mass_RT_;
    mfem::MixedBilinearForm mass_curl_RT_;
    mfem::GridFunction &u_;
    mfem::GridFunction &v_;
    mfem::GridFunction &w_;
    int &num_it_A1_;
    int &num_it_A2_;
    std::chrono::time_point<std::chrono::steady_clock> time_;
    DualFieldConfig config_;
};

// CSV logger for the H(curl)-H(curl) dual-field Navier-Stokes system.
//
// Both velocity fields u1, u2 live in the same ND (H(curl)) space.  The
// quantities logged are the kinetic energies ||u1||^2 and ||u2||^2 (via the
// ND mass matrix), the L2 error against the exact solution (when available),
// and per-system iteration counts.
class HCurlDualFieldCSVLogger : public CSVLogger
{
public:
    HCurlDualFieldCSVLogger(DualFieldConfig &config,
                            int &cycle,
                            double &t,
                            mfem::FiniteElementSpace *ND,
                            mfem::GridFunction &u1,
                            mfem::GridFunction &u2,
                            int &num_it1,
                            int &num_it2)
        : CSVLogger(config.get_outputfile()),
          cycle_(cycle), t_(t),
          ND_(ND),
          mass_ND_(ND),
          u1_(u1), u2_(u2),
          num_it1_(num_it1), num_it2_(num_it2),
          time_(std::chrono::steady_clock::now()),
          config_(config)
    {
        get_ofstream() << "runtime_it,cycle,time,num_it1,num_it2,||u1||,||u2||";
        if (config.has_exact_u())
            get_ofstream() << ",u1_err_L2,u2_err_L2";
        get_ofstream() << std::endl;
        get_ofstream().flush();

        mass_ND_.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
        mass_ND_.Assemble();
        mass_ND_.Finalize();
    }

    void WriteRow() override
    {
        double u1_norm = MatrixConservedVariable(mass_ND_, u1_);
        double u2_norm = MatrixConservedVariable(mass_ND_, u2_);

        std::chrono::duration<double> runtime_it = std::chrono::steady_clock::now() - time_;
        time_ = std::chrono::steady_clock::now();

        get_ofstream() << runtime_it.count() << "," << cycle_ << ","
                       << std::setprecision(15) << std::fixed << t_ << ","
                       << num_it1_ << "," << num_it2_ << ","
                       << u1_norm << "," << u2_norm;
        if (config_.has_exact_u())
        {
            mfem::VectorFunctionCoefficient u_exact(3, config_.get_exact_data("exact_data_u"));
            u_exact.SetTime(t_);
            get_ofstream() << "," << u1_.ComputeL2Error(u_exact)
                           << "," << u2_.ComputeL2Error(u_exact);
        }
        get_ofstream() << std::endl;
        get_ofstream().flush();
    }

private:
    int &cycle_;
    double &t_;
    mfem::FiniteElementSpace *ND_;
    mfem::BilinearForm mass_ND_;
    mfem::GridFunction &u1_, &u2_;
    int &num_it1_, &num_it2_;
    std::chrono::time_point<std::chrono::steady_clock> time_;
    DualFieldConfig config_;
};

// CSV logger for the H(div)-H(div) dual-field Navier-Stokes system.
//
// Both velocity fields u1, u2 live in the same RT (H(div)) space.  The
// quantities logged are the kinetic energies ||u1||^2 and ||u2||^2 (via the
// RT mass matrix), the L2 error against the exact solution (when available),
// and per-system iteration counts.
class HDivDualFieldCSVLogger : public CSVLogger
{
public:
    HDivDualFieldCSVLogger(DualFieldConfig &config,
                           int &cycle,
                           double &t,
                           mfem::FiniteElementSpace *RT,
                           mfem::GridFunction &u1,
                           mfem::GridFunction &u2,
                           int &num_it1,
                           int &num_it2)
        : CSVLogger(config.get_outputfile()),
          cycle_(cycle), t_(t),
          RT_(RT),
          mass_RT_(RT),
          u1_(u1), u2_(u2),
          num_it1_(num_it1), num_it2_(num_it2),
          time_(std::chrono::steady_clock::now()),
          config_(config)
    {
        get_ofstream() << "runtime_it,cycle,time,num_it1,num_it2,||u1||,||u2||";
        if (config.has_exact_u())
            get_ofstream() << ",u1_err_L2,u2_err_L2";
        get_ofstream() << std::endl;
        get_ofstream().flush();

        mass_RT_.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
        mass_RT_.Assemble();
        mass_RT_.Finalize();
    }

    void WriteRow() override
    {
        double u1_norm = MatrixConservedVariable(mass_RT_, u1_);
        double u2_norm = MatrixConservedVariable(mass_RT_, u2_);

        std::chrono::duration<double> runtime_it = std::chrono::steady_clock::now() - time_;
        time_ = std::chrono::steady_clock::now();

        get_ofstream() << runtime_it.count() << "," << cycle_ << ","
                       << std::setprecision(15) << std::fixed << t_ << ","
                       << num_it1_ << "," << num_it2_ << ","
                       << u1_norm << "," << u2_norm;
        if (config_.has_exact_u())
        {
            mfem::VectorFunctionCoefficient u_exact(3, config_.get_exact_data("exact_data_u"));
            u_exact.SetTime(t_);
            get_ofstream() << "," << u1_.ComputeL2Error(u_exact)
                           << "," << u2_.ComputeL2Error(u_exact);
        }
        get_ofstream() << std::endl;
        get_ofstream().flush();
    }

private:
    int &cycle_;
    double &t_;
    mfem::FiniteElementSpace *RT_;
    mfem::BilinearForm mass_RT_;
    mfem::GridFunction &u1_, &u2_;
    int &num_it1_, &num_it2_;
    std::chrono::time_point<std::chrono::steady_clock> time_;
    DualFieldConfig config_;
};

// CSV logger for the semi-Lagrangian single-field Navier-Stokes solver.
//
// The single velocity field u lives in ND (H(curl)).  Logs kinetic energy
// ||u1||^2 (via the ND mass matrix), L2 error (when exact solution is
// available), and iteration count.  Column names are chosen to match
// simbench_core/postprocess.py expectations.
class SingleFieldCSVLogger : public CSVLogger
{
public:
    SingleFieldCSVLogger(DualFieldConfig &config,
                         int &cycle,
                         double &t,
                         mfem::FiniteElementSpace *ND,
                         mfem::GridFunction &u,
                         int &num_it,
                         const double *advect_time_s = nullptr,
                         const double *solve_time_s = nullptr,
                         const double *trace_time_s = nullptr,
                         const double *split_time_s = nullptr,
                         const double *interior_integral_time_s = nullptr,
                         const double *boundary_integral_time_s = nullptr,
                         const long long *split_calls = nullptr,
                         const long long *total_segments = nullptr,
                         const double *edge_thread_min_s = nullptr,
                         const double *edge_thread_avg_s = nullptr,
                         const double *edge_thread_max_s = nullptr,
                         const double *edge_thread_imbalance = nullptr,
                         const int *edge_threads_active = nullptr,
                         const long long *edge_thread_edges_min = nullptr,
                         const long long *edge_thread_edges_max = nullptr,
                         const double *edge_thread_cpu_min_s = nullptr,
                         const double *edge_thread_cpu_avg_s = nullptr,
                         const double *edge_thread_cpu_max_s = nullptr,
                         const double *edge_thread_cpu_util_min = nullptr,
                         const double *edge_thread_cpu_util_avg = nullptr,
                         const double *edge_thread_cpu_util_max = nullptr,
                         const double *advect_time_min_s = nullptr,
                         const double *advect_time_max_s = nullptr,
                         const double *solve_time_min_s = nullptr,
                         const double *solve_time_max_s = nullptr,
                         const double *comm_time_s = nullptr,
                         const long long *gk_fallbacks = nullptr,
                         const int *rebalance_fired = nullptr,
                         const double *advect_imbalance = nullptr)
        : CSVLogger(config.get_outputfile()),
          cycle_(cycle), t_(t),
          ND_(ND),
          mass_ND_(ND),
          u_(u),
          num_it_(num_it),
          advect_time_s_(advect_time_s),
          solve_time_s_(solve_time_s),
          advect_time_min_s_(advect_time_min_s),
          advect_time_max_s_(advect_time_max_s),
          solve_time_min_s_(solve_time_min_s),
          solve_time_max_s_(solve_time_max_s),
          comm_time_s_(comm_time_s),
          gk_fallbacks_(gk_fallbacks),
          rebalance_fired_(rebalance_fired),
          advect_imbalance_(advect_imbalance),
          trace_time_s_(trace_time_s),
          split_time_s_(split_time_s),
          interior_integral_time_s_(interior_integral_time_s),
          boundary_integral_time_s_(boundary_integral_time_s),
          split_calls_(split_calls),
          total_segments_(total_segments),
          edge_thread_min_s_(edge_thread_min_s),
          edge_thread_avg_s_(edge_thread_avg_s),
          edge_thread_max_s_(edge_thread_max_s),
          edge_thread_imbalance_(edge_thread_imbalance),
          edge_threads_active_(edge_threads_active),
          edge_thread_edges_min_(edge_thread_edges_min),
          edge_thread_edges_max_(edge_thread_edges_max),
          edge_thread_cpu_min_s_(edge_thread_cpu_min_s),
          edge_thread_cpu_avg_s_(edge_thread_cpu_avg_s),
          edge_thread_cpu_max_s_(edge_thread_cpu_max_s),
          edge_thread_cpu_util_min_(edge_thread_cpu_util_min),
          edge_thread_cpu_util_avg_(edge_thread_cpu_util_avg),
          edge_thread_cpu_util_max_(edge_thread_cpu_util_max),
          time_(std::chrono::steady_clock::now()),
          config_(config)
    {
        get_ofstream() << "runtime_it,cycle,time_full,num_it,||u1||";
        if (advect_time_s_ && solve_time_s_)
            get_ofstream() << ",time_advect_dofs_s,time_solve_s";
        if (advect_time_min_s_ && advect_time_max_s_ &&
            solve_time_min_s_ && solve_time_max_s_ &&
            comm_time_s_)
            get_ofstream() << ",time_advect_dofs_min_s,time_advect_dofs_max_s,"
                              "time_solve_min_s,time_solve_max_s,time_comm_s";
        if (gk_fallbacks_)
            get_ofstream() << ",gk_fallbacks";
        if (advect_imbalance_)
            get_ofstream() << ",advect_imbalance";
        if (rebalance_fired_)
            get_ofstream() << ",rebalance_fired";
        if (trace_time_s_ && split_time_s_ && interior_integral_time_s_ && boundary_integral_time_s_ && split_calls_ && total_segments_)
            get_ofstream() << ",time_trace_departure_s,time_split_line_s,time_interior_integral_s,time_boundary_integral_s,split_calls,total_segments";
        if (edge_thread_min_s_ && edge_thread_avg_s_ && edge_thread_max_s_ && edge_thread_imbalance_ && edge_threads_active_ && edge_thread_edges_min_ && edge_thread_edges_max_)
            get_ofstream() << ",edge_thread_min_s,edge_thread_avg_s,edge_thread_max_s,edge_thread_imbalance,edge_threads_active,edge_thread_edges_min,edge_thread_edges_max";
        if (edge_thread_cpu_min_s_ && edge_thread_cpu_avg_s_ && edge_thread_cpu_max_s_ && edge_thread_cpu_util_min_ && edge_thread_cpu_util_avg_ && edge_thread_cpu_util_max_)
            get_ofstream() << ",edge_thread_cpu_min_s,edge_thread_cpu_avg_s,edge_thread_cpu_max_s,edge_thread_cpu_util_min,edge_thread_cpu_util_avg,edge_thread_cpu_util_max";
        if (config.has_exact_u())
            get_ofstream() << ",u1_err_L2";
        get_ofstream() << std::endl;
        get_ofstream().flush();

        mass_ND_.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
        mass_ND_.Assemble();
        mass_ND_.Finalize();
    }

    void WriteRow() override
    {
        double u1_norm = std::sqrt(MatrixConservedVariable(mass_ND_, u_));

        std::chrono::duration<double> runtime_it = std::chrono::steady_clock::now() - time_;
        time_ = std::chrono::steady_clock::now();

        get_ofstream() << runtime_it.count() << "," << cycle_ << ","
                       << std::setprecision(15) << std::fixed << t_ << ","
                       << num_it_ << ","
                       << u1_norm;
        if (advect_time_s_ && solve_time_s_)
            get_ofstream() << "," << *advect_time_s_ << "," << *solve_time_s_;
        if (advect_time_min_s_ && advect_time_max_s_ &&
            solve_time_min_s_ && solve_time_max_s_ &&
            comm_time_s_)
            get_ofstream() << "," << *advect_time_min_s_ << "," << *advect_time_max_s_
                           << "," << *solve_time_min_s_ << "," << *solve_time_max_s_
                           << "," << *comm_time_s_;
        if (gk_fallbacks_)
            get_ofstream() << "," << *gk_fallbacks_;
        if (advect_imbalance_)
            get_ofstream() << "," << *advect_imbalance_;
        if (rebalance_fired_)
            get_ofstream() << "," << *rebalance_fired_;
        if (trace_time_s_ && split_time_s_ && interior_integral_time_s_ && boundary_integral_time_s_ && split_calls_ && total_segments_)
            get_ofstream() << "," << *trace_time_s_ << "," << *split_time_s_ << ","
                           << *interior_integral_time_s_ << "," << *boundary_integral_time_s_ << ","
                           << *split_calls_ << "," << *total_segments_;
        if (edge_thread_min_s_ && edge_thread_avg_s_ && edge_thread_max_s_ && edge_thread_imbalance_ && edge_threads_active_ && edge_thread_edges_min_ && edge_thread_edges_max_)
            get_ofstream() << "," << *edge_thread_min_s_ << "," << *edge_thread_avg_s_ << ","
                           << *edge_thread_max_s_ << "," << *edge_thread_imbalance_ << ","
                           << *edge_threads_active_ << "," << *edge_thread_edges_min_ << ","
                           << *edge_thread_edges_max_;
        if (edge_thread_cpu_min_s_ && edge_thread_cpu_avg_s_ && edge_thread_cpu_max_s_ && edge_thread_cpu_util_min_ && edge_thread_cpu_util_avg_ && edge_thread_cpu_util_max_)
            get_ofstream() << "," << *edge_thread_cpu_min_s_ << "," << *edge_thread_cpu_avg_s_ << ","
                           << *edge_thread_cpu_max_s_ << "," << *edge_thread_cpu_util_min_ << ","
                           << *edge_thread_cpu_util_avg_ << "," << *edge_thread_cpu_util_max_;
        if (config_.has_exact_u())
        {
            // The exact-velocity coefficient must match the spatial dimension of
            // the velocity space (2 in 2D, 3 in 3D); a hardcoded 3 breaks the
            // L2-error evaluation on 2D meshes.
            const int sdim = ND_->GetMesh()->Dimension();
            mfem::VectorFunctionCoefficient u_exact(sdim, config_.get_exact_data("exact_data_u"));
            u_exact.SetTime(t_);
            get_ofstream() << "," << u_.ComputeL2Error(u_exact);
        }
        get_ofstream() << std::endl;
        get_ofstream().flush();
    }

private:
    int &cycle_;
    double &t_;
    mfem::FiniteElementSpace *ND_;
    mfem::BilinearForm mass_ND_;
    mfem::GridFunction &u_;
    int &num_it_;
    const double *advect_time_s_;
    const double *solve_time_s_;
    const double *advect_time_min_s_;
    const double *advect_time_max_s_;
    const double *solve_time_min_s_;
    const double *solve_time_max_s_;
    const double *comm_time_s_;
    const long long *gk_fallbacks_;
    const int *rebalance_fired_;
    const double *advect_imbalance_;
    const double *trace_time_s_;
    const double *split_time_s_;
    const double *interior_integral_time_s_;
    const double *boundary_integral_time_s_;
    const long long *split_calls_;
    const long long *total_segments_;
    const double *edge_thread_min_s_;
    const double *edge_thread_avg_s_;
    const double *edge_thread_max_s_;
    const double *edge_thread_imbalance_;
    const int *edge_threads_active_;
    const long long *edge_thread_edges_min_;
    const long long *edge_thread_edges_max_;
    const double *edge_thread_cpu_min_s_;
    const double *edge_thread_cpu_avg_s_;
    const double *edge_thread_cpu_max_s_;
    const double *edge_thread_cpu_util_min_;
    const double *edge_thread_cpu_util_avg_;
    const double *edge_thread_cpu_util_max_;
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
