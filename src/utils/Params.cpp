// (C) Copyright 2011, Jun Zhu (junzhu [at] cs [dot] cmu [dot] edu)

// This file is part of Logistic.

// Logistic is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// Logistic is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
#include "Params.h"
#include <string>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
using namespace std;

Params::Params(void)
{
	train_filename = new char[512];
	test_filename  = new char[512];
	res_filename   = new char[512];
	file_root      = new char[512];
	label = new char[512];
	model = new char[512];
	T = 20;
	SIGMA = 1.0;
	BIASED_HYPERPLANE = 0;
	DISPLAY = 1;
	INITIAL_C1 = 1;
	ORG_FEATURE = 0;
	TRAIN_RATIO = 80;
	SYMMETRIC = 0;
	DELTA_ELL = 4;
	NUM_FEATURES = 0;
	NFOLDS = 0;
}

Params::~Params(void)
{
	delete[] train_filename;
	delete[] test_filename;
	delete[] res_filename;
	delete[] file_root;
	delete[] label;
	delete[] model;
}

void Params::read_settings(char* filename)
{
	char file[512];
	strcpy(file, filename);
	printf("Settings File: %s\n", file);
	FILE* fileptr;
	fileptr = fopen(filename, "r");
	//printf("read start\n");
	if (fileptr != NULL) {
    fscanf(fileptr, "%s\n", label);
    fscanf(fileptr, "%s\n", model);
    fscanf(fileptr, "display %d\n", &DISPLAY);
    fscanf(fileptr, "stochastic nu %d\n", &STOCHASTIC_NU);
    fscanf(fileptr, "stochastic phi %d\n", &STOCHASTIC_PHI);
    fscanf(fileptr, "phi iter %d\n", &PHI_ITER);
    fscanf(fileptr, "original feature %d\n", &ORG_FEATURE);
    fscanf(fileptr, "var max iter %d\n", &VAR_MAX_ITER);
    fscanf(fileptr, "var convergence %f\n", &VAR_CONVERGED);
    fscanf(fileptr, "em max iter %d\n", &EM_MAX_ITER);
    fscanf(fileptr, "em convergence %f\n", &EM_CONVERGED);
    fscanf(fileptr, "KL-C %f\n", &INITIAL_C1);
    fscanf(fileptr, "svm-C %f\n", &INITIAL_C2);
    fscanf(fileptr, "init alpha %f\n", &INITIAL_ALPHA);
    fscanf(fileptr, "svm_alg_type %d\n", &SVM_ALGTYPE);
    fscanf(fileptr, "biased_hyperplane %d\n", &BIASED_HYPERPLANE);
    fscanf(fileptr, "truncated-T %d\n", &T);
    fscanf(fileptr, "phi-dual-opt %d\n", &PHI_DUALOPT);
    fscanf(fileptr, "forgetting rate nu %f\n", &FORGETRATE_NU);
    fscanf(fileptr, "forgetting rate phi %f\n", &FORGETRATE_PHI);

    //printf("read complete half..\n");

    fscanf(fileptr, "train_file: %s\n", train_filename);
    fscanf(fileptr, "test_file: %s\n", test_filename);
    fscanf(fileptr, "class-num: %d\n", &NLABELS);
    fscanf(fileptr, "term-num: %d\n", &NTERMS);
    fscanf(fileptr, "doc-num: %d\n", &NDOCS);
    fscanf(fileptr, "relation-num: %d\n", &NUM_RELATION);
    fscanf(fileptr, "feature-num: %d\n", &NUM_FEATURES);
    fscanf(fileptr, "train-size: %d\n", &train_size);
    fscanf(fileptr, "overall_res: %s\n", res_filename);
    fscanf(fileptr, "relation_root: %s\n", file_root);

    fclose(fileptr);
	}
}

void Params::write_settings(char* filename)
{
	FILE* fileptr;
	// char alpha_action[100];
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "dislay %d\n", DISPLAY);
	fprintf(fileptr, "original feature %d\n", ORG_FEATURE);
	fprintf(fileptr, "var max iter %d\n", VAR_MAX_ITER);
	fprintf(fileptr, "var convergence %f\n", VAR_CONVERGED);
	fprintf(fileptr, "em max iter %d\n", EM_MAX_ITER);
	fprintf(fileptr, "em convergence %f\n", EM_CONVERGED);
	fprintf(fileptr, "KL-C %f\n", INITIAL_C1);
	fprintf(fileptr, "svm-C %f\n", INITIAL_C2);
	fprintf(fileptr, "init alpha %f\n", INITIAL_ALPHA);
	fprintf(fileptr, "svm_alg_type %d\n", SVM_ALGTYPE);
	fprintf(fileptr, "biased_hyperplane %d\n", BIASED_HYPERPLANE);
	fprintf(fileptr, "truncated-T %d\n", T);
	fprintf(fileptr, "phi-dual-opt %d\n", PHI_DUALOPT);

	fprintf(fileptr, "train_file: %s\n", train_filename);
	fprintf(fileptr, "test_file: %s\n", test_filename);
	fprintf(fileptr, "class-num: %d\n", NLABELS);
	fprintf(fileptr, "term-num: %d\n", NTERMS);
	fprintf(fileptr, "doc-num: %d\n", NDOCS);
	fprintf(fileptr, "relation-num: %d\n", NUM_RELATION);
	fprintf(fileptr, "train-size: %d\n", train_size);
	fprintf(fileptr, "overall_res: %s\n", res_filename);
	fprintf(fileptr, "relation_root: %s\n", file_root);

	fclose(fileptr);
}
