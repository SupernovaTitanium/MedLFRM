#include "NormalGammaPrior.h"
#include "utils.h"
#include "cokus.h"
#include <math.h>

NormalGammaPrior::NormalGammaPrior(void)
{
	haveNextGaussian = false;
	m_mu0 = 0;
	m_nu0 = 1;
	m_n0 = 10;
	m_S0 = 0.1;

	// the expectation
	m_dMu = m_mu0; 
	m_dTau = m_nu0 * m_S0 / 4;
}

NormalGammaPrior::~NormalGammaPrior(void)
{
}

double NormalGammaPrior::mean_field(double &dMu, double &dTau, const double *dWExp, const int &nDim)
{
	double dOldTau = m_dTau;
	double dWMean = 0;
	for ( int k=0; k<nDim; k++ ) {
		dWMean += dWExp[k] / nDim;
	}
	double dWMeanSq = dWMean * dWMean;
	double dWMeanSqExp = 1 / (nDim * dOldTau) + dWMeanSq;

	// compute the expectation of S (square error)
	double dSExp = nDim / dOldTau + nDim * dWMeanSqExp;
	for ( int k=0; k<nDim; k++ ) {
		dSExp += dWExp[k] * (dWExp[k] - 2*dWMean);
	}
	double newS = m_S0 + dSExp + (nDim * m_n0 * (dWMeanSq - 2*m_mu0*dWMean + m_mu0*m_mu0)) / (nDim + m_n0);
	double newMu = (nDim * dWMean + m_mu0 * m_n0) / (nDim + m_n0);
	double newN  = m_n0 + nDim;
	double newNu = m_nu0 + nDim;

	// compute the KL-divergence from two norm-gamma distribution.
	double dKLDivergence = compute_kl(newMu, newN, newNu, newS);

	//// update the expected hyper-parameters.
	//m_mu0 = newMu;
	//m_n0  = newN;
	//m_nu0 = newNu;
	//m_S0  = newS;

	//// update the mean of Tau.
	//m_dTau = m_nu0 * m_S0 / 4;
	//m_dMu  = m_mu0;
	m_dTau = newNu / newS;// / 4;
	m_dMu  = newMu;

	dMu = m_dMu;
	dTau = m_dTau;

	return dKLDivergence;
}

double NormalGammaPrior::mean_field2(double &dMu, double &dTau, const double *dWExp, const int &nDim)
{
	double dOldTau = m_dTau;
	double dWMean = 0;
	for ( int k=0; k<nDim; k++ ) {
		dWMean += dWExp[k] / nDim;
	}
	double dWMeanSq = dWMean * dWMean;
	double dWMeanSqExp = 1 / (nDim * dOldTau) + dWMeanSq;

	// compute the expectation of S (square error)
	double dSExp = nDim / dOldTau + nDim * dWMeanSqExp;
	for ( int k=0; k<nDim; k++ ) {
		dSExp += dWExp[k] * (dWExp[k] - 2*dWMean);
	}
	double newS = m_S0 + dSExp + (nDim * m_n0 * (dWMeanSq - 2*m_mu0*dWMean + m_mu0*m_mu0)) / (nDim + m_n0);
	double newMu = (nDim * dWMean + m_mu0 * m_n0) / (nDim + m_n0);
	double newN  = m_n0 + nDim;
	double newNu = m_nu0 + nDim;

	// compute the KL-divergence from two norm-gamma distribution.
	double dKLDivergence = compute_kl(newMu, newN, newNu, newS);

	//// update the expected hyper-parameters.
	//m_mu0 = newMu;
	//m_n0  = newN;
	//m_nu0 = newNu;
	//m_S0  = newS;

	//// update the mean of Tau.
	//m_dTau = m_nu0 * m_S0 / 4;
	//m_dMu  = m_mu0;
	m_dTau = newNu / newS;// / 4;
	m_dMu  = newMu;

	dMu = m_dMu;
	dTau = m_dTau;

	return dKLDivergence;
}

// compute the KL-divergence from a normal-gamma to the prior.
double NormalGammaPrior::compute_kl(double dMu, double dN, double dNu, double dS)
{
	double dRes = 0;

	double dTauExp = dNu / dS; // / 4;
	double dLogTauExp = digamma(0.5*dNu) + log(2/dS); //log(0.5*dS);
	double dMuSqExp = dMu*dMu + 1/(dN * dTauExp);

	dRes = 0.5*(dNu * log(0.5*dS) - m_nu0 * log(0.5*m_S0) + log(dN/m_n0)) 
			- ( log_gamma(0.5*dNu) - log_gamma(0.5*m_nu0) )
			+ 0.5 * (dNu - m_nu0) * dLogTauExp
			- 0.5 * (dS - m_S0) * dTauExp
			- 0.5 * dTauExp * ( (dN-m_n0)*dMuSqExp - 2*(dN*dMu - m_n0*m_mu0)*dMu + (dN*dMu*dMu - m_n0*m_mu0*m_mu0) );

	return dRes;
}

void NormalGammaPrior::draw_sample_prior(double &dMu, double &dTau)
{
	dTau = nextGamma(0.5*m_nu0, 2/m_S0);
	dMu  = nextGaussian(m_mu0, 1/(m_n0 * dTau));

	m_dTau = dTau;
}

void NormalGammaPrior::draw_sample_post(double &dMu, double &dTau, const double *dWExp, const int &nDim)
{
	double dOldTau = m_dTau;
	double *dW = new double[nDim];
	double dWMean = 0;
	// draw a sample of W
	for ( int k=0; k<nDim; k++ ) {
		haveNextGaussian = false; // sample from a difference Gaussian
		dW[k] = nextGaussian(dWExp[k], dOldTau);
		dWMean += dW[k] / nDim;
	}

	double dS = 0;
	for ( int k=0; k<nDim; k++ ) {
		dS += (dW[k] - dWMean) * (dW[k] - dWMean);
	}
	
	// update the hyper-parameters of the prior.
	double newMu = (nDim * dWMean + m_mu0 * m_n0) / (nDim + m_n0);
	double newN  = nDim + m_n0;
	double newNu = m_nu0 + nDim;
	double newS  = dS + m_S0 + nDim*m_n0*pow(dWMean-m_mu0, 2.0) / (nDim + m_n0);
	
	m_mu0 = newMu;
	m_n0  = newN;
	m_nu0 = newNu;
	m_S0  = newS;

	// draw a sample from the updated prior
	draw_sample_prior(dMu, dTau);
	m_dTau = dTau;
	m_dMu = dMu;
}

// generate Gamma(1,1)
// E(X)=1 ; Var(X)=1
/** Return a random double drawn from a Gamma distribution with mean 1.0 and variance 1.0. */
double NormalGammaPrior::nextGamma() 
{
	return nextGamma(1,1,0);
}

/** Return a random double drawn from a Gamma distribution with mean alpha and variance 1.0. */
double NormalGammaPrior::nextGamma(double alpha) 
{
	return nextGamma(alpha,1,0);
}

/** Return a random double drawn from a Gamma distribution with mean alpha*beta and variance alpha*beta^2. */
////(from: http://java2s.com/Code/Java/Development-Class/ReturnasamplefromtheGammaPoissionGaussiandistributionwithparameterIA.htm)
double NormalGammaPrior::nextGamma(double alpha, double beta) {
	return nextGamma(alpha, beta, 0);
}

/** Return a random double drawn from a Gamma distribution with mean alpha*beta+lamba and variance alpha*beta^2. */
double NormalGammaPrior::nextGamma(double alpha, double beta, double lambda) 
{
	double gamma=0;
	if (alpha <= 0 || beta <= 0) {
		printf("alpha and beta must be strictly positive.");
		exit(0);
	}
	if (alpha < 1) {
		double b,p;
		bool flag=false;
		b= 1 + alpha * exp( -1.0 );
		while(!flag) {
			p = b * myrand(); //nextUniform();
			if (p>1) {
				gamma = -log((b-p)/alpha);
				if (myrand()/*nextUniform()*/ <= pow(gamma, alpha-1)) 
					flag=true;
			} else {
				gamma = pow(p, 1/alpha);
				if (myrand() <= exp(-gamma)) 
					flag=true;
			}
		}
	} else if (alpha == 1) {
		gamma = - log( myrand() );
	} else {
		double y = - log( myrand() );
		while ( myrand() > pow(y*exp(1 - y), alpha - 1))
			y = - log( myrand() );
		gamma = alpha * y;
	}

	return beta*gamma + lambda;
}


/** Return a random double drawn from a Gaussian distribution with mean 0 and variance 1. */
double NormalGammaPrior::nextGaussian() 
{
	if (!haveNextGaussian) {
		double v1 = myrand();
		double v2 = myrand();
		double x1,x2;
		x1 = sqrt( -2*log(v1) ) * cos( 2 * M_PI * v2);
		x2 = sqrt( -2*log(v1) ) * sin( 2 * M_PI * v2);
		m_nextGaussian = x2;
		haveNextGaussian = true;
		return x1;
	} else {
		haveNextGaussian=false;
		return m_nextGaussian;
	}
}

/** Return a random double drawn from a Gaussian distribution with mean m and variance s2. */
double NormalGammaPrior::nextGaussian(double m, double s2)
{
	return nextGaussian() * sqrt(s2) + m;
}