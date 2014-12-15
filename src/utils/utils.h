#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <math.h>
#include <utility>
#include <float.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

// #define M_PI 3.141592653589793238462643
#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)
#define NUM_INIT 1

#define NEWTON_THRESH 1e-5
#define MAX_ALPHA_ITER 1000
#define LAG 10
#define MIN_SIGMAX 1e-5;

//double get_runtime(void);

double log_sum(double log_a, double log_b);
double trigamma(double x);
double digamma(double x);
double log_gamma(double x);
double _lgamma(const double &x);
double entropy_beta(const double &alpha, const double &beta);

int argmax(double* x, const int &n);
double dotprod(double *a, double *b, const int&n);
double sum(double *a, const int&n);
double l2norm2(double *a, const int&n);
double l2norm2(double *a, double **A, double *b, const int&n);
double l2dist2(double *a, double *b, const int &n);
void matrixprod(double *a, double **A, double *res, const int&n);
void matrixprod(double **A, double *a, double *res, const int&n);
void addmatrix(double **A, double *a, double *b, const int &n, double factor);
void addmatrix2(double **A, double *a, double *b, const int &n, double factor);
void addvec(double *res, double *a, const int &n, double factor);
void addvec(double *res, double *a, const int &n);

double safe_entropy(double *dist, const int &K);
double safe_logist(double dVal);
void save_mat(char *filename, double **mat, const int &n1, const int &n2);
void load_mat(char *filename, double **mat, const int &n1, const int &n2);
void save_vec(char *filename, double *vec, const int &n);
void load_vec(char *filename, double *vec, const int &n);

double step_size(int iter, double tau, double kappa);

template <typename T1, typename T2>
bool pair_cmp_fst_gt(std::pair<T1, T2> a, std::pair<T1, T2> b) {
  return a.first > b.first;
}

#endif
