#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include "mfem.hpp"
#include "io.h"

// MEHC scheme for dirichlet problem
// essential BC at Hdiv and Hcurl of dual system only


// primal: A1*x=b1
// [M_dt+R1  CT_Re    G] [u]   [(M_dt-R1)*u - CT_Re*z  + f]
// [C        N_n      0] [z] = [             0            ]
// [GT       0        0] [p]   [             0            ]
//
// dual: A2*y=b2
// [N_dt+R2  C_Re     DT_n] [v]   [(N_dt-R2)*u - C_Re*w + f]
// [CT       M_n      0   ] [w] = [            0           ]
// [D        0        0   ] [q]   [            0           ]


struct Parameters {
    // double Re_inv = 1/100.; // = 1/Re 
    double Re_inv = 0.; // = 1/Re 
    // double Re_inv = 1.; // = 1/Re 
    // double Re_inv = 1./1600.; // = 1/Re 
    double dt     = 0.0001;
    double tmax   = 0.0001;
    int ref_steps = 5;
    int init_ref  = 0;
    int order     = 1;
    double tol    = 1e-3;
    // std::string outputfile = "out/rawdata/dirichlet-conv-Re1-sTGV.txt";
    // std::string outputfile = "out/rawdata/dirichlet-conv-invisc-sTGV.txt";
    // std::string outputfile = "out/rawdata/dirichlet-conv-Re1-sin.txt";
    // std::string outputfile = "out/rawdata/dirichlet-conv-invisc-sin.txt";
    // std::string outputfile = "out/rawdata/dirichlet-conv-Re1-sin2.txt";
    // std::string outputfile = "out/rawdata/dirichlet-conv-invisc-sin2.txt";
    // std::string outputfile = "out/rawdata/dirichlet-conv-Re1-sTGV2.txt";
    // std::string outputfile = "out/rawdata/dirichlet-conv-invisc-sTGV2.txt";
    // std::string outputfile = "out/rawdata/dirichlet-conv-Re1-dTGV.txt";
    // std::string outputfile = "out/rawdata/dirichlet-conv-invisc-dTGV.txt";

    std::string outputfile = "out/rawdata/dirichlet-conv-test1.txt";
    const char* mesh_file = "extern/mfem/data/periodic-cube.mesh";
    // const char* mesh_file = "extern/mfem-4.5/data/ref-square.mesh";
};


void PrintVector3(mfem::Vector vec, int stride=1, 
                  int start=0, int stop=0, int prec=3);
void u_t(const mfem::Vector &x, double t, mfem::Vector &v);
void w_t(const mfem::Vector &x, double t, mfem::Vector &v);
void f_t(const mfem::Vector &x, double t, mfem::Vector &v); 


int main(int argc, char *argv[]) {
    SimulationConfig config("/scratch/users/wtonnon/VisualStudioProjects/dualfieldmfem/data/config/example2.json");
    
    // Parse configuration parameters
    double dt = config.get_dt();                  // Time step
    double T = config.get_T();                    // Total time
    double viscosity = config.get_viscosity();
    int refinements = config.get_refinements();   // Number of mesh refinements
    int order = config.get_order();               // Finite element order
    int visualisation = config.get_visualisation(); // Visualisation level
    std::string mesh_string = config.get_mesh(); // Path to mesh file
    std::string output_file = config.get_outputfile(); // Output file for results
    

    std::function<void(const mfem::Vector&, double, mfem::Vector&)> boundary_data_u = 
        std::bind(&SimulationConfig::boundary_data_u, &config, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    std::function<void(const mfem::Vector&, mfem::Vector&)> initial_data_u = 
        std::bind(&SimulationConfig::initial_data_u, &config, std::placeholders::_1, std::placeholders::_2);
    std::function<void(const mfem::Vector&, mfem::Vector&)> initial_data_w = 
        std::bind(&SimulationConfig::initial_data_w, &config, std::placeholders::_1, std::placeholders::_2);

    // simulation parameters
    Parameters param;
    double Re_inv = 0*viscosity; 
    double tmax   = T;
    int ref_steps = 0;
    int init_ref  = 2;
    double tol    = 1e-12;




    // loop over refinement steps to check convergence
    
    // start timer
    auto start = std::chrono::high_resolution_clock::now();

    // output file 
    std::string outputfile = param.outputfile;
    std::ofstream file;
    file.precision(6);
    // file.open(outputfile);
    file.open(outputfile, std::ios::app);

    // mesh
    const char *mesh_file = param.mesh_file;
    mfem::Mesh mesh(mesh_file, 1, 1); 
    int dim = mesh.Dimension(); 
    int l;
    for (l = 0; l<refinements; l++) {mesh.UniformRefinement();} 



    // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
    mfem::FiniteElementCollection *fec_DG = new mfem::L2_FECollection(order-1,dim);
    mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order,dim);
    mfem::FiniteElementCollection *fec_RT = new mfem::RT_FECollection(order-1,dim);
    mfem::FiniteElementCollection *fec_CG = new mfem::H1_FECollection(order,dim);
    mfem::FiniteElementSpace DG(&mesh, fec_DG);
    mfem::FiniteElementSpace ND(&mesh, fec_ND);
    mfem::FiniteElementSpace RT(&mesh, fec_RT);
    mfem::FiniteElementSpace CG(&mesh, fec_CG);

    // Initialize time-stepping variables
    double t = 0.;
    int cycle = 0;

    // unkowns and gridfunctions
    mfem::GridFunction u(&ND); //u = 4.3;
    mfem::GridFunction z(&RT); //z = 5.3;
    mfem::GridFunction p(&CG); p=0.; //p = 6.3;
    mfem::GridFunction v(&RT); //v = 3.;
    mfem::GridFunction w(&ND); //w = 3.; 
    mfem::GridFunction q(&DG); q=0.; //q = 9.3;

    // initial condition
    mfem::VectorFunctionCoefficient u_coeff(dim, initial_data_u);
    mfem::VectorFunctionCoefficient w_coeff(dim, initial_data_w); 
    u_coeff.SetTime(t);
    w_coeff.SetTime(t);
    u.ProjectCoefficient(u_coeff);
    v.ProjectCoefficient(u_coeff);
    z.ProjectCoefficient(w_coeff);
    w.ProjectCoefficient(w_coeff);


    // Set up ParaView data collection for visualization
    mfem::ParaViewDataCollection vtk_dc("/scratch/users/wtonnon/VisualStudioProjects/dualfieldmfem/data/visualisation/paraview/" + output_file, &mesh);
    if (visualisation > 0)
    {
        vtk_dc.RegisterField("u1", &u); // Register field for visualization
        vtk_dc.RegisterField("u2", &v); // Register field for visualization
        vtk_dc.RegisterField("w1", &w); // Register field for visualization
        vtk_dc.RegisterField("w2", &z); // Register field for visualization
        vtk_dc.RegisterField("p0", &p); // Register field for visualization
        vtk_dc.RegisterField("p3", &q); // Register field for visualization
        vtk_dc.SetCycle(0);                  // Set initial cycle
        vtk_dc.SetTime(0.0);                 // Set initial time
        vtk_dc.Save();                       // Save initial data
    }

    // linearform for forcing term
    mfem::VectorFunctionCoefficient f_coeff(dim, f_t);
    f_coeff.SetTime(t);
    mfem::LinearForm f1(&ND);
    mfem::LinearForm f2(&RT);
    f1.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coeff)); //=(f,v)
    f2.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coeff)); //=(f,v)
    f1.Assemble();
    f2.Assemble(); 



  

    // system size
    int size_1 = u.Size() + z.Size() + p.Size();
    int size_2 = v.Size() + w.Size() + q.Size();
    
    // initialize solution vectors
    mfem::Vector x(size_1);
    mfem::Vector y(size_2);
    x.SetVector(u,0);
    x.SetVector(z,u.Size());
    x.SetVector(p,u.Size()+z.Size());
    y.SetVector(v,0);
    y.SetVector(w,v.Size());
    y.SetVector(q,v.Size()+w.Size());

    // helper dofs
    mfem::Array<int> u_dofs (u.Size());
    mfem::Array<int> z_dofs (z.Size());
    mfem::Array<int> p_dofs (p.Size());
    mfem::Array<int> v_dofs (v.Size());
    mfem::Array<int> w_dofs (w.Size());
    mfem::Array<int> q_dofs (q.Size());
    std::iota(&u_dofs[0], &u_dofs[u.Size()], 0);
    std::iota(&z_dofs[0], &z_dofs[z.Size()], u.Size());
    std::iota(&p_dofs[0], &p_dofs[p.Size()], u.Size()+z.Size());
    std::iota(&v_dofs[0], &v_dofs[v.Size()], 0);
    std::iota(&w_dofs[0], &w_dofs[w.Size()], v.Size());
    std::iota(&q_dofs[0], &q_dofs[q.Size()], v.Size()+w.Size());

    // Matrix M
    mfem::BilinearForm blf_M(&ND);
    blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator()); //=(u,v)
    blf_M.Assemble();
    blf_M.Finalize();
    mfem::SparseMatrix M_n(blf_M.SpMat());
    mfem::SparseMatrix M_dt;
    M_dt = M_n;
    M_dt *= 1/dt;
    M_n *= -1.;
    M_dt.Finalize();
    M_n.Finalize();
    
    // Matrix N
    mfem::BilinearForm blf_N(&RT);
    blf_N.AddDomainIntegrator(new mfem::VectorFEMassIntegrator()); //=(u,v)
    blf_N.Assemble();
    blf_N.Finalize();
    mfem::SparseMatrix N_n(blf_N.SpMat());
    mfem::SparseMatrix N_dt;
    N_dt = N_n;
    N_dt *= 1/dt;
    N_n *= -1.;
    N_dt.Finalize();
    N_n.Finalize();



    // Matrix C
    mfem::MixedBilinearForm blf_C(&ND, &RT);
    blf_C.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator()); //=(curl u,v)
    blf_C.Assemble();
    blf_C.Finalize();
    mfem::SparseMatrix C(blf_C.SpMat());
    mfem::SparseMatrix *CT;
    mfem::SparseMatrix C_Re;
    mfem::SparseMatrix CT_Re;
    CT = Transpose(C);
    C_Re = C;
    CT_Re = *CT; 
    C_Re *= Re_inv/2.;
    CT_Re *= Re_inv/2.;
    C.Finalize();
    CT->Finalize();
    C_Re.Finalize();
    CT_Re.Finalize();

    // Matrix D
    mfem::MixedBilinearForm blf_D(&RT, &DG);
    blf_D.AddDomainIntegrator(new mfem::MixedScalarDivergenceIntegrator()); //=(div u,v)
    blf_D.Assemble();
    blf_D.Finalize();
    mfem::SparseMatrix D(blf_D.SpMat());
    mfem::SparseMatrix *DT_n;
    DT_n = Transpose(D);
    *DT_n *= -1.;
    D.Finalize();
    DT_n->Finalize();

    // Matrix G
    mfem::MixedBilinearForm blf_G(&CG, &ND);
    blf_G.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator()); //=(grad u,v)
    blf_G.Assemble();
    blf_G.Finalize();
    mfem::SparseMatrix G(blf_G.SpMat());
    mfem::SparseMatrix *GT;
    GT = Transpose(G);
    G.Finalize();
    GT->Finalize();    


    // initialize system matrices
    mfem::Array<int> offsets_1 (4);
    offsets_1[0] = 0;
    offsets_1[1] = u.Size();
    offsets_1[2] = z.Size();
    offsets_1[3] = p.Size();
    offsets_1.PartialSum(); // exclusive scan
    mfem::BlockOperator A1(offsets_1);
    mfem::Array<int> offsets_2 (4);
    offsets_2[0] = 0;
    offsets_2[1] = v.Size();
    offsets_2[2] = w.Size();
    offsets_2[3] = q.Size();
    offsets_2.PartialSum();
    mfem::BlockMatrix A2(offsets_2);

    // initialize rhs
    mfem::Vector b1(size_1);
    mfem::Vector b1sub(u.Size());
    mfem::Vector b2(size_2); 
    mfem::Vector b2sub(v.Size());

    ////////////////////////////////////////////////////////////////////
    // EULERSTEP: code up to the loop computes euler step for primal sys
    ////////////////////////////////////////////////////////////////////

    // Matrix MR_eul for eulerstep
    mfem::MixedBilinearForm blf_MR_eul(&ND,&ND); 
    mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
    mfem::ConstantCoefficient two_over_dt(2.0/dt);
    blf_MR_eul.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(two_over_dt)); //=(u,v)
    blf_MR_eul.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
    blf_MR_eul.Assemble();
    blf_MR_eul.Finalize();
    mfem::SparseMatrix MR_eul(blf_MR_eul.SpMat());
    MR_eul.Finalize();
    
    // CT for eulerstep
    mfem::SparseMatrix CT_eul = CT_Re;
    CT_eul *= 2;
    CT_eul.Finalize();

    // assemble and solve system
    A1.SetBlock(0,0, &MR_eul);
    A1.SetBlock(0,1, &CT_eul);
    A1.SetBlock(0,2, &G);
    A1.SetBlock(1,0, &C);
    A1.SetBlock(1,1, &N_n);
    A1.SetBlock(2,0, GT);

    // update b1, b2 for eulerstep
    b1 = 0.0;
    b1sub = 0.0;
    M_dt.AddMult(u,b1sub,2);
    b1.AddSubVector(f1,0);
    b1.AddSubVector(b1sub,0);

    // Transposition
    mfem::TransposeOperator AT1 (&A1);
    mfem::ProductOperator ATA1 (&AT1,&A1,false,false);
    mfem::Vector ATb1 (size_1);
    A1.MultTranspose(b1,ATb1);

    // solve 
    int iter = 100000000;  
    mfem::MINRES(ATA1, ATb1, x, 0, iter, tol*tol, tol*tol);

    // extract solution values u,z,p from eulerstep
    x.GetSubVector(u_dofs, u);
    x.GetSubVector(z_dofs, z);
    x.GetSubVector(p_dofs, p);


    // --- CSV output for monitoring six variables (u,z,p,v,w,q) ---
    // CSV path: out/<output_file>_vars.csv
    std::string csv_path = std::string("out/") + output_file + std::string("_vars.csv");
    // Always truncate/clear the CSV at program start so we don't append old runs.
    std::ofstream csv(csv_path, std::ios::out);
    if (csv)
    {
        std::cout << "[info] CSV opened (truncated): " << csv_path << std::endl;
        // write CSV header
        csv << "cycle,time_full,time_half,||u1||,||u2||,(u1,w1),(u2,w2)\n";
        // write initial condition row (cycle, t=0)
        csv << cycle << "," << std::setprecision(15) << std::fixed << t << "," << t << ","
            << blf_M.InnerProduct(u, u) << "," << blf_N.InnerProduct(v,v) << "," <<  ","
            << blf_M.InnerProduct(u,w)  << "," << blf_N.InnerProduct(v,z) << "\n";
        csv.flush();
    }
    else
    {
        std::cerr << "[warn] Failed to open CSV (for truncation): " << csv_path << std::endl;
    }

    // time loop
    int barWidth = 70;
    while (t < T - dt / 2.) {
        // Print progress bar for simulation
        float progress = t / T;
        if (progress < 1.0)
        {
            std::cout << "[";
            int pos = barWidth * progress;
            for (int i = 0; i < barWidth; ++i)
            {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << "% \r";
            std::cout.flush();
        }
        t += dt;
        cycle++;
        ////////////////////////////////////////////////////////////////////
        // DUAL FIELD
        ////////////////////////////////////////////////////////////////////

        // update R2 
        mfem::MixedBilinearForm blf_R2(&RT,&RT);
        mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
        blf_R2.AddDomainIntegrator(
            new mfem::MixedCrossProductIntegrator(z_gfcoeff)); //=(wxu,v)
        blf_R2.Assemble();
        blf_R2.Finalize();
        mfem::SparseMatrix R2(blf_R2.SpMat());
        R2 *= 1./2.;
        R2.Finalize();

        // update NR
        mfem::MixedBilinearForm blf_NR(&RT,&RT); 
        blf_NR.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(two_over_dt)); //=(u,v)
        blf_NR.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(z_gfcoeff)); //=(wxu,v)
        blf_NR.Assemble();
        blf_NR.Finalize();
        mfem::SparseMatrix NR(blf_NR.SpMat());
        NR *= 1./2.;
        NR.Finalize();

        // update A2
        A2.SetBlock(0,0, &NR);
        A2.SetBlock(0,1, &C_Re);
        A2.SetBlock(0,2, DT_n);
        A2.SetBlock(1,0, CT);
        A2.SetBlock(1,1, &M_n);
        A2.SetBlock(2,0, &D);

        // update f2
        f_coeff.SetTime(t-dt);
        mfem::LinearForm f2(&RT);
        f2.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coeff)); //=(f,v)
        f2.Assemble(); 
        
        // update b2
        b2 = 0.0;
        b2sub = 0.0;
        N_dt.AddMult(v,b2sub);
        R2.AddMult(v,b2sub,-1);
        C_Re.AddMult(w,b2sub,-1);
        b2.AddSubVector(f2,0); 
        b2.AddSubVector(b2sub,0);

        // remove unnecessary equations from matrix corresponding to essdofs

        // Transposition
        mfem::TransposeOperator AT2 (&A2);
        mfem::ProductOperator ATA2 (&AT2,&A2,false,false);
        mfem::Vector ATb2 (size_2);
        A2.MultTranspose(b2,ATb2);

        // solve  
        mfem::MINRES(ATA2, ATb2, y, 1, iter, tol*tol, tol*tol);
        y.GetSubVector(v_dofs, v);
        y.GetSubVector(w_dofs, w);
        y.GetSubVector(q_dofs, q);                

        ////////////////////////////////////////////////////////////////////
        // PRIMAL FIELD
        ////////////////////////////////////////////////////////////////////

        // update R1
        mfem::MixedBilinearForm blf_R1(&ND,&ND);
        mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
        blf_R1.AddDomainIntegrator(
            new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
        blf_R1.Assemble();
        blf_R1.Finalize();
        mfem::SparseMatrix R1(blf_R1.SpMat());
        R1 *= 1./2.;
        R1.Finalize();

        // update MR
        mfem::MixedBilinearForm blf_MR(&ND,&ND); 
        blf_MR.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(two_over_dt)); //=(u,v)
        blf_MR.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
        blf_MR.Assemble();
        blf_MR.Finalize();
        mfem::SparseMatrix MR(blf_MR.SpMat());
        MR *= 1./2.;
        MR.Finalize();

        // update A1
        A1.SetBlock(0,0, &MR);
        A1.SetBlock(0,1, &CT_Re);
        A1.SetBlock(0,2, &G);
        A1.SetBlock(1,0, &C);
        A1.SetBlock(1,1, &N_n);
        A1.SetBlock(2,0, GT);

        // update f1
        f_coeff.SetTime(t);
        mfem::LinearForm f1(&ND);
        f1.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coeff)); //=(f,v)
        f1.Assemble();

        // update boundary integral for primal reynolds term#
        mfem::LinearForm lform_zxn(&ND);
        w_coeff.SetTime(t);
        lform_zxn.AddBoundaryIntegrator(new mfem::VectorFEBoundaryTangentLFIntegrator(w_coeff)); 
        lform_zxn.Assemble();
        lform_zxn *= -1.*Re_inv; // minus!

        // update boundary integral for primal div-free cond
        mfem::LinearForm lform_un(&CG);
        u_coeff.SetTime(t);
        lform_un.AddBoundaryIntegrator(new mfem::BoundaryNormalLFIntegrator(u_coeff)); 
        lform_un.Assemble();

        // update b1
        b1 = 0.0;
        b1sub = 0.0;
        M_dt.AddMult(u,b1sub);
        R1.AddMult(u,b1sub,-1);
        CT_Re.AddMult(z,b1sub,-1);
        b1.AddSubVector(b1sub,0);
        b1.AddSubVector(f1,0);
        b1.AddSubVector(lform_zxn, 0); // NEU
        b1.AddSubVector(lform_un, u.Size() + z.Size());

        // Transposition
        mfem::TransposeOperator AT1 (&A1);
        mfem::ProductOperator ATA1 (&AT1,&A1,false,false);
        mfem::Vector ATb1 (size_1);
        A1.MultTranspose(b1,ATb1);

        // solve 
        // Choose SuperLU as the linear solver
        //mfem::SuperLUSolver superlu;
        //superlu.SetOperator(ATA1);

        // Optionally:
        //superlu.SetPrintStatistics(true);
        // superlu.SetEquilibriate(true); // depending on the MFEM version

        //superlu.Mult(ATb1, x);
        //mfem::HypreParMatrix *ATA1_ptr = (&ATA1).As<mfem::HypreParMatrix>()->GetSystemMatrix();
        //mfem::MUMPSSolver mumps(ATA1);
        //mumps.Mult(ATb1, x);
        mfem::UMFPackSolver solver;
        solver.SetOperator(ATA1);
        solver.Mult(ATb1, x);
        //x.Print(std::cout);
        mfem::Vector res(ATb1);
        ATA1.AddMult(x,res,-1.);
        std::cout<< "Error UMFPack: " << res.Norml2()/ATb1.Norml2() << std::endl;

        mfem::MINRES(ATA1, ATb1, x, 1, iter, tol*tol, tol*tol);
        x.GetSubVector(u_dofs, u);
        x.GetSubVector(z_dofs, z);
        x.GetSubVector(p_dofs, p);

        // Append norms to CSV for this timestep (after full primal solve)
        if (csv)
        {
            csv << cycle << "," << std::setprecision(15) << std::fixed << t << "," << t-0.5*dt << ","
            << blf_M.InnerProduct(u, u) << "," << blf_N.InnerProduct(v,v) << "," <<  ","
            << blf_M.InnerProduct(u,w)  << "," << blf_N.InnerProduct(v,z) << "\n";
            csv.flush();
        }     //   std::cout << "[info] CSV opened: " << csv_path << std::endl;


        // Save data to ParaView if visualization is enabled
        if (visualisation > 0)
        {
            vtk_dc.SetCycle(cycle);             // Update cycle in ParaView
            vtk_dc.SetTime(dt * cycle);         // Update time in ParaView
            vtk_dc.Save();                      // Save data
        }

    } // time loop



    // convergence error 
    u_coeff.SetTime(t);
    mfem::VectorGridFunctionCoefficient v_gfcoeff(&v);
    double err_L2_u = u.ComputeL2Error(u_coeff);
    double err_L2_v = v.ComputeL2Error(u_coeff);
    double err_L2_diff = u.ComputeL2Error(v_gfcoeff);

    // print and save convergence error 
    std::cout << "L2err of u = "<< err_L2_u<<"\n";
    std::cout << "L2err of v = "<< err_L2_v<<"\n";
    std::cout << "L2err(u-v) = "<< err_L2_diff <<"\n";

    // runtime
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = 1000*(end - start);
    std::cout << "runtime = " << duration.count() << "ms" << std::endl;


    // free memory
    delete fec_DG;
    delete fec_CG;
    delete fec_ND;
    delete fec_RT;

    // close file
    file.close();
    if (csv.is_open()) { csv.close(); }

    // refinement loop
}



/////////////////////////////////////////////////////////////////////
// simple sin, funktioniert
// void u_t(const mfem::Vector &x, double t, mfem::Vector &returnvalue) { 

//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     returnvalue(0) = std::sin(Y);
//     returnvalue(1) = std::sin(Z);
//     returnvalue(2) = 0.;
// }
// void w_t(const mfem::Vector &x, double t, mfem::Vector &returnvalue) { 
   
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     returnvalue(0) = -std::cos(Z);
//     returnvalue(1) = 0.;
//     returnvalue(2) = -std::cos(Y);
// }
// void f_t(const mfem::Vector &x, double t, mfem::Vector &returnvalue) { 
//     Parameters param;
//     double Re_inv = param.Re_inv; // = 1/Re 
    
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     returnvalue(0) = std::sin(Y)*Re_inv + std::cos(Y)*std::sin(Z);
//     returnvalue(1) = -std::cos(Y)*std::sin(Y) + std::sin(Z)*Re_inv;
//     returnvalue(2) = - std::cos(Z)*std::sin(Z);
// }



/////////////////////////////////////////////////////////////////////
// sTGV, funktioniert
// void u_t(const mfem::Vector &x, double t, mfem::Vector &returnvalue) {
   
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;
   
//     returnvalue(0) =     std::cos(X)*std::sin(Y);
//     returnvalue(1) = -1* std::sin(X)*std::cos(Y);
//     returnvalue(2) = 0;
// }

// void w_t(const mfem::Vector &x, double t, mfem::Vector &returnvalue) { 
   
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;
   
//     returnvalue(0) = 0;
//     returnvalue(1) = 0;
//     returnvalue(2) = -2* std::cos(X) * std::cos(Y);
// }

// void f_t(const mfem::Vector &x, double t, mfem::Vector &returnvalue) { 
   
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     Parameters param;
//     double Re_inv = param.Re_inv;

//     returnvalue(0) = 2.*Re_inv*std::cos(X)*std::sin(Y) - 2.*std::sin(X)*std::cos(X)*std::cos(Y)*std::cos(Y);
//     returnvalue(1) = -2.*Re_inv*std::sin(X)*std::cos(Y) -2.*std::cos(X)*std::cos(X)*std::sin(Y)*std::cos(Y);
//     returnvalue(2) = 0.;
// }




/////////////////////////////////////////////////////////////////////
// dTGV (dynamic TGV), funktioniert
void u_t(const mfem::Vector &x, double t, mfem::Vector &returnvalue) {
    
    Parameters param;
    double Re_inv = param.Re_inv; 

    double F = std::exp(-2*Re_inv*t);
   
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;
   
    returnvalue(0) =     std::cos(X)*std::sin(Y) * F;
    returnvalue(1) = -1* std::sin(X)*std::cos(Y) * F;
    returnvalue(2) = 0;
}

void w_t(const mfem::Vector &x, double t, mfem::Vector &returnvalue) { 
    
    Parameters param;
    double Re_inv = param.Re_inv; 

    double F = std::exp(-2*Re_inv*t);
   
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;
   
    returnvalue(0) = 0;
    returnvalue(1) = 0;
    returnvalue(2) = -2*F* std::cos(X) * std::cos(Y);
}

void f_t(const mfem::Vector &x, double t, mfem::Vector &returnvalue) { 
    
    Parameters param;
    double Re_inv = param.Re_inv; 

    double Fof2 = std::exp(-4*Re_inv*t);
   
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;

    returnvalue(0) = 0.;
    returnvalue(1) = 0.;
    returnvalue(2) = 0.;
}



/////////////////////////////////////////////////////////////////////
// decaying bump solution, doesnt work yet
// TODO: fix time, fix everything, boudnary int, and forcing (doesnt converge)
// void u_0(const mfem::Vector &x, mfem::Vector &returnvalue) {
   
//     double F = 1.;
   
//     double pi = 3.14159265358979323846;
//     double C = 10;
//     double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;
    
//     double cos = std::cos(C*(X*X+Y*Y+Z*Z));
//     double cos2 = cos*cos;
   
//     returnvalue(0) = F*cos2;
//     returnvalue(1) = 0;
//     returnvalue(2) = 0;
// }

// void u_t(const mfem::Vector &x, double t, mfem::Vector &returnvalue) {
    
//     Parameters param;
//     double Re_inv = param.Re_inv; 
//     double tmax   = param.tmax;
//     double F = std::exp(-1*Re_inv*tmax);
   
//     double pi = 3.14159265358979323846;
//     double C = 10;
//     double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;
    
//     double cos = std::cos(C*(X*X+Y*Y+Z*Z));
//     double cos2 = cos*cos;

//     if (X*X + Y*Y + Z*Z < R*R) {   
//         returnvalue(0) = F*cos2;
//         returnvalue(1) = 0;
//         returnvalue(2) = 0;
//     }
//     else {
//         returnvalue(0) = 0;
//         returnvalue(1) = 0;
//         returnvalue(2) = 0;
//     }
// }

// void w_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
//     double F = 1.;
   
//     double pi = 3.14159265358979323846;
//     double C = 10;
//     double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     double sinof2 = std::sin(2*C*(X*X+Y*Y+Z*Z));

//     if (X*X + Y*Y + Z*Z < R*R) {  
//         returnvalue(0) = 0;
//         returnvalue(1) = - 2*C* Z *F * sinof2;
//         returnvalue(2) = + 2*C* Y *F * sinof2; 
//     }
//     else {
//         returnvalue(0) = 0;
//         returnvalue(1) = 0;
//         returnvalue(2) = 0;
//     }
// }

// void f(const mfem::Vector &x, mfem::Vector &returnvalue) { 
    
//     Parameters param;
//     double Re_inv = param.Re_inv; 
//     double tmax   = param.tmax;
//     double F = std::exp(-1*Re_inv*tmax); 
   
//     double pi = 3.14159265358979323846;
//     double C = 10;
//     double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;
    
//     // double cos = std::cos(C*(X*X+Y*Y+Z*Z));
//     // double sin = std::sin(C*(X*X+Y*Y+Z*Z));
//     // double cos2 = cos*cos;
//     // double cos3 = cos*cos*cos;
//     // double sinof2 = std::sin(2*C*(X*X+Y*Y+Z*Z));
//     // double cosof2 = std::cos(2*C*(X*X+Y*Y+Z*Z));

//     // double Fof2 = std::exp(-2*Re_inv*tmax); 
//     // double temp1 = 4*C* Fof2 ;
//     // double Fpos = std::exp(+1*Re_inv*tmax); 
//     // double temp2 = 2*C*X*Fpos*cosof2 - 1/Re_inv *sin*cos3;

//     double EE = 2.7182818284590452353;
//     double Rey = 1/Re_inv;
//     double CC = C;
//     double t = tmax;


//     if (X*X + Y*Y + Z*Z < R*R) {  
//         // returnvalue(0) = F * ( 4*C*(sinof2+2*C*(Y*Y+Z*Z)*cosof2) - cos2) * Re_inv;
//         // returnvalue(1) = -temp1* Y * temp2 * Re_inv;
//         // returnvalue(2) = -temp1* Z * temp2 * Re_inv;
        
//         returnvalue(0) = (-std::pow(std::cos(CC*(std::pow(X,2) + std::pow(Y,2) + std::pow(Z,2))),2) + 
//                             4*CC*(2*CC*(std::pow(Y,2) + std::pow(Z,2))*std::cos(2*CC*(std::pow(X,2) + std::pow(Y,2) + std::pow(Z,2))) + 
//                             std::sin(2*CC*(std::pow(X,2) + std::pow(Y,2) + std::pow(Z,2)))))/(std::pow(EE,t/Rey)*Rey);
//         returnvalue(1) = (-4*CC*Y*(2*CC*std::pow(EE,t/Rey)*X*std::cos(2*CC*(std::pow(X,2) + std::pow(Y,2) + std::pow(Z,2))) - 
//                             Rey*std::pow(std::cos(CC*(std::pow(X,2) + std::pow(Y,2) + std::pow(Z,2))),3)*
//                              std::sin(CC*(std::pow(X,2) + std::pow(Y,2) + std::pow(Z,2)))))/(std::pow(EE,(2*t)/Rey)*Rey);
//         returnvalue(2) = (-4*CC*Z*(2*CC*std::pow(EE,t/Rey)*X*std::cos(2*CC*(std::pow(X,2) + std::pow(Y,2) + std::pow(Z,2))) - 
//                             Rey*std::pow(std::cos(CC*(std::pow(X,2) + std::pow(Y,2) + std::pow(Z,2))),3)*
//                              std::sin(CC*(std::pow(X,2) + std::pow(Y,2) + std::pow(Z,2)))))/(std::pow(EE,(2*t)/Rey)*Rey);
//     }
//     else {
//         returnvalue(0) = 0;
//         returnvalue(1) = 0;
//         returnvalue(2) = 0;
//     }
// }