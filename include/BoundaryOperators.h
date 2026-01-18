#ifndef SEMILAGRANGE0FORMS_BOUNDARYOPERATORS_H
#define SEMILAGRANGE0FORMS_BOUNDARYOPERATORS_H

#include <mfem.hpp>




class NormalJumpIntegrator : public mfem::BilinearFormIntegrator
{
private:
    mfem::Vector vec, pointflux;
#ifndef MFEM_THREAD_SAFE
    mfem::Vector D;
    mfem::DenseMatrix curlshape, curlshape_dFt, M;
    mfem::DenseMatrix te_curlshape, te_curlshape_dFt;
    mfem::DenseMatrix vshape, projcurl;
#endif

protected:
    mfem::Coefficient *Q;
    mfem::DiagonalMatrixCoefficient *DQ;
    mfem::MatrixCoefficient *MQ;
    double alpha, beta, gamma;

    // PA extension
    mfem::Vector pa_data;
    const mfem::DofToQuad *mapsO;       ///< Not owned. DOF-to-quad map, open.
    const mfem::DofToQuad *mapsC;       ///< Not owned. DOF-to-quad map, closed.
    const mfem::GeometricFactors *geom; ///< Not owned
    int dim, ne, nq, dofs1D, quad1D;
    bool symmetric = true; ///< False if using a nonsymmetric matrix coefficient

public:

    NormalJumpIntegrator() : alpha(1.), beta(1.), gamma(1.)
    {
        Q = NULL;
        DQ = NULL;
        MQ = NULL;
    }


    NormalJumpIntegrator(double eps) : alpha(eps), beta(eps), gamma(eps)
    {
        Q = NULL;
        DQ = NULL;
        MQ = NULL;
    }

    NormalJumpIntegrator(double alpha, double beta, double gamma) : alpha(alpha), beta(beta), gamma(gamma)
    {
        Q = NULL;
        DQ = NULL;
        MQ = NULL;
    }
    /// Construct a bilinear form integrator for Nedelec elements
    NormalJumpIntegrator(mfem::Coefficient &q, const mfem::IntegrationRule *ir = NULL) : BilinearFormIntegrator(ir), Q(&q), DQ(NULL), MQ(NULL) {}
    NormalJumpIntegrator(mfem::DiagonalMatrixCoefficient &dq,
                     const mfem::IntegrationRule *ir = NULL) : BilinearFormIntegrator(ir), Q(NULL), DQ(&dq), MQ(NULL) {}
    NormalJumpIntegrator(mfem::MatrixCoefficient &mq, const mfem::IntegrationRule *ir = NULL) : BilinearFormIntegrator(ir), Q(NULL), DQ(NULL), MQ(&mq) {}

    void AssembleFaceMatrix(const mfem::FiniteElement &el1, 
                            const mfem::FiniteElement &el2,
                            mfem::FaceElementTransformations &Trans, 
                            mfem::DenseMatrix &elmat);

    //virtual void AssembleElementMatrix2(const mfem::FiniteElement &trial_fe,
    //                                    const mfem::FiniteElement &test_fe,
    //                                    mfem::ElementTransformation &Trans,
    //                                    mfem::DenseMatrix &elmat);

    //virtual void ComputeElementFlux(const mfem::FiniteElement &el,
    //                                mfem::ElementTransformation &Trans,
    //                                mfem::Vector &u, const mfem::FiniteElement &fluxelem,
    //                                mfem::Vector &flux, bool with_coef,
    //                                const mfem::IntegrationRule *ir = NULL);

    //virtual mfem::real_t ComputeFluxEnergy(const mfem::FiniteElement &fluxelem,
    //                                       mfem::ElementTransformation &Trans,
    //                                       mfem::Vector &flux, mfem::Vector *d_energy = NULL);

    using BilinearFormIntegrator::AssemblePA;

    void AssemblePA(const mfem::FiniteElementSpace &)
    {
        MFEM_ABORT("BilinearFormIntegrator::AssemblePA(fes)\n"
                   "   is not implemented for this class.");
    };
    void AddMultPA(const mfem::Vector &, mfem::Vector &) const
    {
        MFEM_ABORT("BilinearFormIntegrator:AddMultPA:(...)\n"
                   "   is not implemented for this class.");
    }
    void AssembleDiagonalPA(mfem::Vector &)
    {
        MFEM_ABORT("BilinearFormIntegrator::AssembleDiagonalPA(...)\n"
                   "   is not implemented for this class.");
    }

    const mfem::Coefficient *GetCoefficient() const { return Q; }
};


class GradJumpIntegrator : public mfem::BilinearFormIntegrator
{
private:
    mfem::Vector vec, pointflux;
#ifndef MFEM_THREAD_SAFE
    mfem::Vector D;
    mfem::DenseMatrix curlshape, curlshape_dFt, M;
    mfem::DenseMatrix te_curlshape, te_curlshape_dFt;
    mfem::DenseMatrix vshape, projcurl;
#endif

protected:
    mfem::Coefficient *Q;
    mfem::DiagonalMatrixCoefficient *DQ;
    mfem::MatrixCoefficient *MQ;
    double alpha, beta, gamma;

    // PA extension
    mfem::Vector pa_data;
    const mfem::DofToQuad *mapsO;       ///< Not owned. DOF-to-quad map, open.
    const mfem::DofToQuad *mapsC;       ///< Not owned. DOF-to-quad map, closed.
    const mfem::GeometricFactors *geom; ///< Not owned
    int dim, ne, nq, dofs1D, quad1D;
    bool symmetric = true; ///< False if using a nonsymmetric matrix coefficient

public:

    GradJumpIntegrator() : alpha(1.), beta(1.), gamma(1.)
    {
        Q = NULL;
        DQ = NULL;
        MQ = NULL;
    }


    GradJumpIntegrator(double eps) : alpha(eps), beta(eps), gamma(eps)
    {
        Q = NULL;
        DQ = NULL;
        MQ = NULL;
    }

    GradJumpIntegrator(double alpha, double beta, double gamma) : alpha(alpha), beta(beta), gamma(gamma)
    {
        Q = NULL;
        DQ = NULL;
        MQ = NULL;
    }
    /// Construct a bilinear form integrator for Nedelec elements
    GradJumpIntegrator(mfem::Coefficient &q, const mfem::IntegrationRule *ir = NULL) : BilinearFormIntegrator(ir), Q(&q), DQ(NULL), MQ(NULL) {}
    GradJumpIntegrator(mfem::DiagonalMatrixCoefficient &dq,
                     const mfem::IntegrationRule *ir = NULL) : BilinearFormIntegrator(ir), Q(NULL), DQ(&dq), MQ(NULL) {}
    GradJumpIntegrator(mfem::MatrixCoefficient &mq, const mfem::IntegrationRule *ir = NULL) : BilinearFormIntegrator(ir), Q(NULL), DQ(NULL), MQ(&mq) {}

    void AssembleFaceMatrix(const mfem::FiniteElement &el1, 
                            const mfem::FiniteElement &el2,
                            mfem::FaceElementTransformations &Trans, 
                            mfem::DenseMatrix &elmat);

    //virtual void AssembleElementMatrix2(const mfem::FiniteElement &trial_fe,
    //                                    const mfem::FiniteElement &test_fe,
    //                                    mfem::ElementTransformation &Trans,
    //                                    mfem::DenseMatrix &elmat);

    //virtual void ComputeElementFlux(const mfem::FiniteElement &el,
    //                                mfem::ElementTransformation &Trans,
    //                                mfem::Vector &u, const mfem::FiniteElement &fluxelem,
    //                                mfem::Vector &flux, bool with_coef,
    //                                const mfem::IntegrationRule *ir = NULL);

    //virtual mfem::real_t ComputeFluxEnergy(const mfem::FiniteElement &fluxelem,
    //                                       mfem::ElementTransformation &Trans,
    //                                       mfem::Vector &flux, mfem::Vector *d_energy = NULL);

    using BilinearFormIntegrator::AssemblePA;

    void AssemblePA(const mfem::FiniteElementSpace &)
    {
        MFEM_ABORT("BilinearFormIntegrator::AssemblePA(fes)\n"
                   "   is not implemented for this class.");
    };
    void AddMultPA(const mfem::Vector &, mfem::Vector &) const
    {
        MFEM_ABORT("BilinearFormIntegrator:AddMultPA:(...)\n"
                   "   is not implemented for this class.");
    }
    void AssembleDiagonalPA(mfem::Vector &)
    {
        MFEM_ABORT("BilinearFormIntegrator::AssembleDiagonalPA(...)\n"
                   "   is not implemented for this class.");
    }

    const mfem::Coefficient *GetCoefficient() const { return Q; }
};



class WouterIntegrator : public mfem::BilinearFormIntegrator
{
private:
    mfem::Vector vec, pointflux;
#ifndef MFEM_THREAD_SAFE
    mfem::Vector D;
    mfem::DenseMatrix curlshape, curlshape_dFt, M;
    mfem::DenseMatrix te_curlshape, te_curlshape_dFt;
    mfem::DenseMatrix vshape, projcurl;
#endif

protected:
    mfem::Coefficient *Q;
    mfem::DiagonalMatrixCoefficient *DQ;
    mfem::MatrixCoefficient *MQ;
    double factor_, theta_, Cw_;

    // PA extension
    mfem::Vector pa_data;
    const mfem::DofToQuad *mapsO;       ///< Not owned. DOF-to-quad map, open.
    const mfem::DofToQuad *mapsC;       ///< Not owned. DOF-to-quad map, closed.
    const mfem::GeometricFactors *geom; ///< Not owned
    int dim, ne, nq, dofs1D, quad1D;
    bool symmetric = true; ///< False if using a nonsymmetric matrix coefficient

public:

    WouterIntegrator(double theta, double Cw, double factor = 1.) : factor_(factor), theta_(theta), Cw_(Cw)
    {
        Q = NULL;
        DQ = NULL;
        MQ = NULL;
    }

    /// Construct a bilinear form integrator for Nedelec elements

    /* Given a particular Finite Element, compute the
       element curl-curl matrix elmat */
    virtual void AssembleElementMatrix(const mfem::FiniteElement &el,
                                       mfem::ElementTransformation &Trans,
                                       mfem::DenseMatrix &elmat);

    void AssembleFaceMatrix(const mfem::FiniteElement &el1, 
                            const mfem::FiniteElement &el2,
                            mfem::FaceElementTransformations &Trans, 
                            mfem::DenseMatrix &elmat);

    //virtual void AssembleElementMatrix2(const mfem::FiniteElement &trial_fe,
    //                                    const mfem::FiniteElement &test_fe,
    //                                    mfem::ElementTransformation &Trans,
    //                                    mfem::DenseMatrix &elmat);

    //virtual void ComputeElementFlux(const mfem::FiniteElement &el,
    //                                mfem::ElementTransformation &Trans,
    //                                mfem::Vector &u, const mfem::FiniteElement &fluxelem,
    //                                mfem::Vector &flux, bool with_coef,
    //                                const mfem::IntegrationRule *ir = NULL);

    //virtual mfem::real_t ComputeFluxEnergy(const mfem::FiniteElement &fluxelem,
    //                                       mfem::ElementTransformation &Trans,
    //                                       mfem::Vector &flux, mfem::Vector *d_energy = NULL);

    using BilinearFormIntegrator::AssemblePA;

    void AssemblePA(const mfem::FiniteElementSpace &)
    {
        MFEM_ABORT("BilinearFormIntegrator::AssemblePA(fes)\n"
                   "   is not implemented for this class.");
    };
    void AddMultPA(const mfem::Vector &, mfem::Vector &) const
    {
        MFEM_ABORT("BilinearFormIntegrator:AddMultPA:(...)\n"
                   "   is not implemented for this class.");
    }
    void AssembleDiagonalPA(mfem::Vector &)
    {
        MFEM_ABORT("BilinearFormIntegrator::AssembleDiagonalPA(...)\n"
                   "   is not implemented for this class.");
    }

    const mfem::Coefficient *GetCoefficient() const { return Q; }
};

class WouterLFIntegrator : public mfem::LinearFormIntegrator
{
   mfem::Vector shape;
   mfem::VectorCoefficient &Q;
   double factor_, theta_, Cw_;
   int oa, ob;
public:
   /** @brief Constructs a boundary integrator with a given Coefficient @a QG.
       Integration order will be @a a * basis_order + @a b. */
   WouterLFIntegrator(double theta, double Cw, mfem::VectorCoefficient &QG, double factor = 1., int a = 1, int b = 1)
      : factor_(factor), theta_(theta), Cw_(Cw), Q(QG), oa(a), ob(b) { }
 

 
   /** Given a particular boundary Finite Element and a transformation (Tr)
       computes the element boundary vector, elvect. */
   virtual void AssembleRHSElementVect(const mfem::FiniteElement &el,
                                       mfem::ElementTransformation &Tr,
                                       mfem::Vector &elvect);
   virtual void AssembleRHSElementVect(const mfem::FiniteElement &el,
                                       mfem::FaceElementTransformations &Tr,
                                       mfem::Vector &elvect);

    
 
   using LinearFormIntegrator::AssembleRHSElementVect;
};

#endif
