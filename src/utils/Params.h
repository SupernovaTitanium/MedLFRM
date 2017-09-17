#pragma once
#include <string>


class Params
{
public:
	Params(void);
public:
	~Params(void);

	void read_settings(char* filename);
	void write_settings(char* filename);

public:
	float EM_CONVERGED;
	int EM_MAX_ITER;
	float INITIAL_ALPHA;
	float INITIAL_C1;
	float INITIAL_C2;

	int NLABELS;
	int NFOLDS;
	int FOLDIX;
	float DELTA_ELL;
	int PHI_DUALOPT;

	int VAR_MAX_ITER;
	float VAR_CONVERGED;

	int SVM_ALGTYPE;       // the algorithm type for SVM
	int T;                 // T for truncated variational inference
	float SIGMA;
	int ORG_FEATURE;
	int ESTIMATE_SIGMA;
	int BIASED_HYPERPLANE;
	int DISPLAY;
	int TRAIN_RATIO;
	int SYMMETRIC;
	int NUM_RELATION;
	int NUM_FEATURES;
	char *model;
	char *label;
	char *file_root;

	char *train_filename;   // the file names of training & testing data sets
	char *test_filename;
	char *res_filename;
	int NTERMS;
	int NDOCS;
	int train_size;

	int STOCHASTIC_NU;
	int STOCHASTIC_PHI;
	int PHI_ITER;
	float FORGETRATE_NU;
	float FORGETRATE_PHI;
};
