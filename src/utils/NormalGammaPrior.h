#ifndef NORMAL_GAMMA_PRIOR
#define NORMAL_GAMMA_PRIOR

class NormalGammaPrior
{
public:
	NormalGammaPrior(void);
	NormalGammaPrior(double mu, double n, double nu, double S) {
		m_mu0 = mu;
		m_n0  = n;
		m_nu0 = nu;
		m_S0  = S;

		// the expectation
		m_dMu = m_mu0; 
		m_dTau = m_nu0 / m_S0;// / 4;
	}
	~NormalGammaPrior(void);

	// perform mean-field updates (return the KL-divergence)
	double mean_field(double &dMu, double &dTau, const double *dWExp, const int &nDim);

	double mean_field2(double &dMu, double &dTau, const double *dWExp, const int &nDim);

	// draw a random sample from a posterior NormalGamma distribution.
	void draw_sample_prior(double &dMu, double &dTau);

	// draw a random sample from a posterior NormalGamma distribution.
	void draw_sample_post(double &dMu, double &dTau, const double *dWExp, const int &nDim);

	// draw random samples from a standard distribution
	double nextGamma();
	double nextGamma(double alpha);
	double nextGamma(double alpha, double beta);

	double nextGaussian();
	double nextGaussian(double m, double s2);

private:
	double compute_kl(double dMu, double dN, double dNu, double dS);
	double nextGamma(double alpha, double beta, double lambda);
public:
	// the 4 hyper-parameters.
	double m_mu0;
	double m_n0;
	double m_nu0;
	double m_S0;

private:
	double m_nextGaussian;
	bool haveNextGaussian;
	double m_dTau;
	double m_dMu;
};

#endif