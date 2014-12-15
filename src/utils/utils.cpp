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

#include "utils.h"
#include <time.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//double get_runtime(void)
//{
//  /* returns the current processor time in hundredth of a second */
//  clock_t start;
//  start = clock();
//  return((double)start/((double)(CLOCKS_PER_SEC)/100.0));
//}

double safe_entropy(double *dist, const int &K)
{
  double dEnt = 0;

  for ( int k=0; k<K; k++ ) {
    if ( dist[k] > 1e-30 ) dEnt -= dist[k] * log( dist[k] );
  }

  return dEnt;
}

double safe_logist(double dVal)
{
  if ( 50 < dVal ) return 1;
  if ( -50 > dVal ) return 0;

  return 1 / (1 + exp(-dVal));
}

double log_sum(double log_a, double log_b)
{
  double dval = 0;

  if (log_a < log_b) {
    dval = log_b + log(1 + exp(log_a - log_b));
  } else {
    dval = log_a + log(1 + exp(log_b - log_a));
  }
  return dval;
}

double trigamma(double x)
{
  double p;
  int i;

  x=x+6;
  p=1/(x*x);
  p=(((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)
          *p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
  for (i=0; i<6 ;i++)
  {
    x=x-1;
    p=1/(x*x)+p;
  }
  return(p);
}

double digamma(double x)
{
  double p;
  x=x+6;
  p=1/(x*x);
  p=(((0.004166666666667*p-0.003968253986254)*p+
        0.008333333333333)*p-0.083333333333333)*p;
  p=p+log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
  return p;
}

double _lgamma(const double &x)
{
  double x0,x2,xp,gl,gl0;
  int n,k;
  static double a[] = {
    8.333333333333333e-02,
    -2.777777777777778e-03,
    7.936507936507937e-04,
    -5.952380952380952e-04,
    8.417508417508418e-04,
    -1.917526917526918e-03,
    6.410256410256410e-03,
    -2.955065359477124e-02,
    1.796443723688307e-01,
    -1.39243221690590
  };

  x0 = x;
  if (x <= 0.0) return 1e308;
  else if ((x == 1.0) || (x == 2.0)) return 0.0;
  else if (x <= 7.0) {
    n = (int)(7-x);
    x0 = x+n;
  }
  x2 = 1.0/(x0*x0);
  xp = 2.0*M_PI;
  gl0 = a[9];
  for (k=8;k>=0;k--) {
    gl0 = gl0*x2 + a[k];
  }
  gl = gl0/x0+0.5*log(xp)+(x0-0.5)*log(x0)-x0;
  if (x <= 7.0) {
    for (k=1;k<=n;k++) {
      gl -= log(x0-1.0);
      x0 -= 1.0;
    }
  }
  return gl;
}

double log_gamma(double x)
{
  double z=1/(x*x);

  x=x+6;
  z=(((-0.000595238095238*z+0.000793650793651)
        *z-0.002777777777778)*z+0.083333333333333)/x;
  z=(x-0.5)*log(x)-x+0.918938533204673+z-log(x-1)-
    log(x-2)-log(x-3)-log(x-4)-log(x-5)-log(x-6);
  return z;
}

double entropy_beta(const double &alpha, const double &beta)
{
  double dRes = (_lgamma(alpha)+_lgamma(beta) - _lgamma(alpha+beta))
    - (alpha-1)*digamma(alpha) - (beta-1)*digamma(beta)
    + (alpha+beta-2) * digamma(alpha+beta);

  return dRes;
}

int argmax(double* x, const int &n)
{
  double max = x[0];
  int argmax = 0;
  for (int i=1; i<n; i++) {
    if (x[i] > max) {
      max = x[i];
      argmax = i;
    }
  }

  return argmax;
}

double dotprod(double *a, double *b, const int&n)
{
  double dres = 0;
  for ( int i=0; i<n; i++ ) {
    dres += a[i] * b[i];
  }
  return dres;
}

double sum(double *a, const int &n)
{
  double dres = 0;
  for ( int i=0; i<n; i++ ) 
    dres += a[i];
  return dres;
}

double l2norm2(double *a, const int&n)
{
  return dotprod(a, a, n);
}

double l2norm2(double *a, double **A, double *b, const int&n)
{
  double dres = 0;
  for ( int i=0; i<n; i++ ) {
    dres += a[i] * dotprod(A[i], b, n);
  }
  return dres;
}

double l2dist2(double *a, double *b, const int &n)
{
  double dres = 0, dDiff = 0;
  for ( int i=0; i<n; i++ ) {
    dDiff = a[i] - b[i];
    dres += dDiff * dDiff;
  }
  return dres;
}

/* a vector times a (n x n) square matrix  */
void matrixprod(double *a, double **A, double *res, const int &n)
{
  for ( int i=0; i<n; i++ ) {
    res[i] = 0;
    for ( int j=0; j<n; j++ ) {
      res[i] += a[j] * A[j][i];
    }
  }
}
/* a (n x n) square matrix times a vector. */
void matrixprod(double **A, double *a, double *res, const int &n)
{
  for ( int i=0; i<n; i++ ) {
    res[i] = 0;
    for ( int j=0; j<n; j++ ) {
      res[i] += a[j] * A[i][j];
    }
  }
}

/* A + ab^\top*/
void addmatrix(double **A, double *a, double *b, const int &n, double factor)
{
  for (int i=0; i<n; i++ ) {
    for ( int j=0; j<n; j++ ) {
      A[i][j] += a[i] * b[j] * factor;
    }
  }
}

/* A + ab^\top + ba^\top*/
void addmatrix2(double **A, double *a, double *b, const int &n, double factor)
{
  for (int i=0; i<n; i++ ) {
    for ( int j=0; j<n; j++ ) {
      A[i][j] += (a[i] * b[j] + b[i] * a[j]) * factor;
    }
  }
}

/* res = res + a * factor */
void addvec(double *res, double *a, const int &n, double factor)
{
  for ( int i=0; i<n; i++ ) {
    res[i] += a[i] * factor;
  }
}

/* res = res + a */
void addvec(double *res, double *a, const int &n)
{
  for ( int i=0; i<n; i++ ) {
    res[i] += a[i];
  }
}

void save_mat(char *filename, double **mat, const int &n1, const int &n2)
{
  FILE *fptr = fopen(filename, "w");
  for ( int i=0; i<n1; i++ ) {
    double *dPtr = mat[i];
    for ( int j=0; j<n2; j++ ) {
      fprintf(fptr, "%.20f ", dPtr[j]);
    }
    fprintf(fptr, "\n");
  }
  fclose( fptr );
}

void load_mat(char *filename, double **mat, const int &n1, const int &n2)
{
  double dVal = 0;
  FILE *fptr = fopen(filename, "r");
  for ( int i=0; i<n1; i++ ) {
    double *dPtr = mat[i];
    for ( int j=0; j<n2; j++ ) {
      fscanf(fptr, "%lf ", &dVal);
      dPtr[j] = dVal;
    }
  }
  fclose( fptr );
}
void save_vec(char *filename, double *vec, const int &n)
{
  FILE *fptr = fopen(filename, "w");
  for ( int i=0; i<n; i++ ) {
    fprintf(fptr, "%.20f\n", vec[i]);
  }
  fclose( fptr );
}

void load_vec(char *filename, double *vec, const int &n)
{
  double dval = 0;
  FILE *fptr = fopen(filename, "r");
  for ( int i=0; i<n; i++ ) {
    fscanf(fptr, "%lf", &dval);
    vec[i] = dval;
  }
  fclose( fptr );
}

// the step size for stocastic variational inference
// (iter + tau) ^ -kappa
double step_size(int iter, double tau, double kappa) {
  return pow(iter + tau, -kappa);
}
