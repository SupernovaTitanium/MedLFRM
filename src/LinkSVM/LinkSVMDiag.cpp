#include "LinkSVMDiag.h"
#include <cassert>
#include <algorithm>
#include <set>
#include <boost/timer.hpp>
#include <boost/filesystem.hpp>
#include "../utils/simple_sparse_vec_hash.h"

using namespace std;
namespace fs = boost::filesystem;

#define INF HUGE_VAL

LinkSVMDiag::LinkSVMDiag(void) {
  _delay_nu = 1;
  _forgetting_rate_nu = 0;
  _delay_phi = 1;
  _forgetting_rate_phi = 0.9;
  m_pParam = NULL;
  m_dMu = NULL;
  m_dW = NULL;
  m_alpha = 1;
  m_var_nu = NULL;
  m_dNuExpSum = NULL;
  m_dNuExpProd = NULL;
  m_digamma_v1_ = NULL;
  m_digamma_v2_ = NULL;
  m_digamma_vsum_ = NULL;
  m_exp_feat_num_ = NULL;
  m_exp_feat_n_minus_num_ = NULL;
  m_sum_phi_ = NULL;
  m_dB = 0;
  truePos_ = NULL;
  falsePos_ = NULL;
  fvec_ = NULL;
  m_stochastic_nu = 0;
  m_stochastic_phi = 0;
  running_time_for_svm = 0.0;
  running_time_for_update_nu = 0.0;
  running_time_for_vi = 0.0;
  running_time_for_update_phi = 0.0;
  phi_iter = 1;
}

LinkSVMDiag::LinkSVMDiag(Params *param) {
  _delay_nu = 1;
  _forgetting_rate_nu = param->FORGETRATE_NU;
  _delay_phi = 1;
  _forgetting_rate_phi = param->FORGETRATE_PHI;
  m_pParam = param;
  m_dMu = NULL;
  m_dW = NULL;
  m_alpha = param->INITIAL_ALPHA;
  m_var_nu = NULL;
  m_dNuExpSum = NULL;
  m_dNuExpProd = NULL;
  m_digamma_v1_ = NULL;
  m_digamma_v2_ = NULL;
  m_digamma_vsum_ = NULL;
  m_exp_feat_num_ = NULL;
  m_exp_feat_n_minus_num_ = NULL;
  m_sum_phi_ = NULL;
  m_dB = 0;
  m_dWPriorVar = param->SIGMA;
  m_dDeltaEll = param->DELTA_ELL;
  truePos_ = NULL;
  falsePos_ = NULL;
  fvec_ = NULL;
  m_stochastic_nu = param->STOCHASTIC_NU;
  m_stochastic_phi = param->STOCHASTIC_PHI;
  running_time_for_svm = 0.0;
  running_time_for_update_nu = 0.0;
  running_time_for_update_phi = 0.0;
  running_time_for_vi = 0.0;
  phi_iter = param->PHI_ITER;
  mini_batch_svm_to = new int[m_stochastic_nu * m_stochastic_phi *
      m_pParam->VAR_MAX_ITER * 10];
  mini_batch_svm_from = new int[m_stochastic_nu * m_stochastic_phi *
      m_pParam->VAR_MAX_ITER * 10];
  mini_batch_svm_label = new int[m_stochastic_nu * m_stochastic_phi *
      m_pParam->VAR_MAX_ITER * 10];
}

LinkSVMDiag::~LinkSVMDiag(void) {
}

void LinkSVMDiag::new_model(int num_docs, int num_train_links) {
  m_nK = m_pParam->T;
  m_nLabelNum = m_pParam->NLABELS;
  m_dC = m_pParam->INITIAL_C1;

  m_dMu = (double *) malloc(sizeof(double) * num_train_links * m_nLabelNum);
  memset(m_dMu, 0, sizeof(double) * num_train_links * m_nLabelNum);

  // likelihood parameters
  m_nOrgFeatures = m_pParam->NUM_FEATURES * 2;
  m_nLatentFeatures = m_nK * 3; // change
  m_nSVMFeature = m_nLatentFeatures + m_nOrgFeatures;
  m_dW = (double ***) malloc(sizeof(double **) * m_nLabelNum);
  m_dOrgW = (double **) malloc(sizeof(double *) * m_nLabelNum);
  for (int y = 0; y < m_nLabelNum; y++) {
    m_dW[y] = (double **) malloc(sizeof(double *) * m_nK);
    m_dOrgW[y] = (double *) malloc(sizeof(double) * m_nOrgFeatures);
    for (int k = 0; k < m_nK; k++) {
      m_dW[y][k] = (double *) malloc(sizeof(double) * m_nK);
      memset(m_dW[y][k], 0, sizeof(double) * m_nK);
    }
    memset(m_dOrgW[y], 0, sizeof(double) * m_nOrgFeatures);
  }
  fvec_ = (double *) malloc(sizeof(double) * m_nSVMFeature);
  m_nData = num_docs;
  m_nTrainLinks = num_train_links;

  m_var_nu = (double **) malloc(sizeof(double *) * m_nK);
  for (int k = 0; k < m_nK; k++) {
    m_var_nu[k] = (double *) malloc(sizeof(double) * 2);
  }
  m_dNuExpSum = (double *) malloc(sizeof(double) * m_nK);
  m_dNuExpProd = (double *) malloc(sizeof(double) * m_nK);

  m_sum_phi_ = (double *) malloc(sizeof(double) * m_nK);
  m_digamma_v1_ = (double *) malloc(sizeof(double) * m_nK);
  m_digamma_v2_ = (double *) malloc(sizeof(double) * m_nK);
  m_digamma_vsum_ = (double *) malloc(sizeof(double) * m_nK);
  m_dQ_dist_ = (double **) malloc(sizeof(double *) * m_nK);
  for (int k = 0; k < m_nK; k++) {
    m_dQ_dist_[k] = (double *) malloc(sizeof(double) * (k + 1));
  }
  m_exp_feat_num_ = (double *) malloc(sizeof(double) * m_nK);
  m_exp_feat_n_minus_num_ = (double *) malloc(sizeof(double) * m_nK);

  dAVec = new double[m_nLabelNum];
  dBVec = new double[m_nLabelNum];
  dCVec = new double[m_nLabelNum];
  m_phi_old_ = (double *) malloc(sizeof(double) * m_nK);
}

void LinkSVMDiag::free_model() {
  if (m_dMu != NULL) free(m_dMu);

  for (int y = 0; y < m_nLabelNum; y++) {
    for (int k = 0; k < m_nK; k++) {
      free(m_dW[y][k]);
    }
    free(m_dW[y]);
  }
  free(m_dW);

  if (m_var_nu != NULL) {
    for (int k = 0; k < m_nK; k++) free(m_var_nu[k]);
    free(m_var_nu);
  }
  if (m_dNuExpSum != NULL) free(m_dNuExpSum);
  if (m_dNuExpProd != NULL) free(m_dNuExpProd);

  if (m_digamma_v1_ != NULL) free(m_digamma_v1_);
  if (m_digamma_v2_ != NULL) free(m_digamma_v2_);
  if (m_digamma_vsum_ != NULL) free(m_digamma_vsum_);
  if (m_dQ_dist_ != NULL) {
    for (int k = 0; k < m_nK; k++) {
      free(m_dQ_dist_[k]);
    }
    free(m_dQ_dist_);
  }
  if (m_exp_feat_num_ != NULL) free(m_exp_feat_num_);
  if (m_exp_feat_n_minus_num_ != NULL) free(m_exp_feat_n_minus_num_);
  if (m_sum_phi_ != NULL) free(m_sum_phi_);

  delete[] dAVec;
  delete[] dBVec;
  delete[] dCVec;
  free(m_phi_old_);
  if (truePos_ != NULL) free(truePos_);
  if (falsePos_ != NULL) free(falsePos_);
  if (fvec_ != NULL) free(fvec_);
}

double LinkSVMDiag::train(char *directory, Corpus *pC) {
  int d;
  boost::timer Timer;

  // initialize model
  new_model(pC->num_docs, pC->num_train_links);
  strcpy(m_dir, directory);

  // allocate variational parameters
  double **phi = (double **) malloc(sizeof(double *) * pC->num_docs);
  m_dYDist = (double **) malloc(sizeof(double *) * pC->num_docs);
  for (d = 0; d < pC->num_docs; d++) {
    Document *pDoc = &(pC->docs[d]);
    phi[d] = (double *) malloc(sizeof(double) * m_nK);
    m_dYDist[d] = (double *) malloc(sizeof(double) * pDoc->num_neighbors);
    for (int i = 0; i < pDoc->num_neighbors; i++) {
      m_dYDist[d][i] = pDoc->linkGnd[i];
    }
  }
  random_init(pC, phi);
  random_wvar_nu();
  random_w();

  char filename[512];
  if (m_pParam->DISPLAY == 1) {
    sprintf(filename, "%s/000", directory);
    save_model(filename, pC, phi);
  }

  // run expectation maximization
  sprintf(filename, "%s/lhood%.2f.dat", directory, _forgetting_rate_phi);
  FILE *likelihood_file = fopen(filename, "w");

  double lhood, obj_old = 0, converged = 1, dTstTime = 0;
  double dTstAcc_fst = 0, dF1_fst = 0, dAUC_fst = 0, dTstAcc = 0, dTstAUC = 0;
  double dBestTstAcc, dBestTstAUC = 0, dBestTstF1, dBestTrAUC = 0;
  int nIt = 0;
  while (/*((converged > m_pParam->EM_CONVERGED)
            || (nIt <= 2)) && */(nIt < m_pParam->EM_MAX_ITER)) {
    if (m_pParam->DISPLAY) printf("**** em iteration %d ****\n", nIt + 1);

    // e-step
    if (m_pParam->DISPLAY) printf("\t e-step (update phi, nu, gamma) \n");
    // random_init(pC, phi);
    //random_wvar_nu();
    if (m_stochastic_nu == 0) lhood = e_step(pC, m_var_nu, phi);
    else lhood = e_step(pC, m_var_nu, phi, m_stochastic_nu);

    // m-step
    if (m_pParam->DISPLAY) printf("\t learn svm and/or hyper-parameters \n");
    //learn_svm(pC, phi, NULL); //old, traditional svm.
    learn_svm(pC, phi, NULL, 0.1, m_dC, m_dC);
    //learn_svm_mini_batch(pC, phi, NULL, 0.1, m_dC, m_dC, nIt);
    //learn_svm_pegasos(pC, phi, nIt);
    // check for convergence
    double dobj = lhood;
    converged = fabs((obj_old - dobj) / (obj_old));
    //if (converged < 0) VAR_MAX_ITER = VAR_MAX_ITER * 2;
    obj_old = dobj;

    Document *pDoc;
    // get test time
    for (int i = 0; i < pC->num_docs; i++) {
      pDoc = &(pC->docs[i]);
      for (int j = 0; j < pDoc->num_neighbors; j++) {
        int jIx = pDoc->neighbors[j];
        if (!pDoc->bTrain[j]) continue;
        loss_aug_predict(pDoc, &(pC->docs[jIx]), phi[i], phi[jIx], j);
      }
    }

    boost::timer testTimer;
    update_ydist(pC, phi);
    printf("compute_auc_tr complete\n");
    dTstAcc = get_test_acc(pC, phi);
    printf("compute_test_acc complete\n");
    dTstAUC = compute_auc(pC, m_dYDist);
    dTstTime += testTimer.elapsed();
    printf("compute_auc complete\n");
    update_ydist_tr(pC, phi);
    double dTrAUC = compute_auc_tr(pC, m_dYDist);
    printf("compute_auc_tr complete\n");
    if (dTrAUC > (dBestTrAUC + 0.00000001)) {
      dBestTrAUC = dTrAUC;
      dBestTstAUC = dTstAUC;
      dBestTstAcc = dTstAcc;
      dBestTstF1 = m_dF1Score;
    }
    if (nIt == 0) {
      dTstAcc_fst = dTstAcc;
      dF1_fst = m_dF1Score;
      dAUC_fst = dTstAUC;
    }
    printf("calculate AUC complete\n");
    // output model and lhood
    fprintf(likelihood_file, "%.10f\t%.3f\t%.3f\t%.3f\t%.3f\t%5.5e\n",
        dobj, dTstAUC, dTstAcc, m_dF1Score, dTrAUC, converged);
    fflush(likelihood_file);
    if (m_pParam->DISPLAY && ((nIt % LAG) == 0)) {
      sprintf(filename, "%s/%d", directory, nIt + 1);
      save_model(filename, pC, phi);
    }
    nIt++;
  }
  double dTrainTime = Timer.elapsed() - dTstTime;
  fclose(likelihood_file);
  printf("Training time in (cpu-seconds): %.2f\n", dTrainTime);

  // output the final model
  sprintf(filename, "%s/final", directory);
  save_model(filename, pC, phi);

  // test data accuracy
  // Document *pDoc = NULL;
  printf("\n\n LinkSVMDiag: test accuracy: %.5f; AUC: %.4f\n\n", dTstAcc, dTstAUC);

  sprintf(filename, "%s/train-phi.dat", directory);
  save_mat(filename, phi, pC->num_docs, m_nK);

  // save the prediction performance
  sprintf(filename, "%s/evl-performance.dat", directory);
  dTstAcc = save_prediction(filename, pC, phi);

  //fs::path cur_path = fs::current_path().parent_path();
  sprintf(filename, "%s", m_pParam->res_filename);

  FILE *fileptr = fopen(filename, "a");
  fprintf(fileptr, "(K: %d; S_nu: %d; S_phi: %d, kappa: %.2f; F: %d; alpha: %.3f; C1: %.3f; C2: "
          "%.3f; Ell: %.3f): acc %.3f; f1: %.3f; auc: %.3f; fst_acc: %.3f; fst_f1: %.3f; fst_auc: %.3f;"
          "train: %.4f; test: %.4f\n",
      m_nK, m_pParam->STOCHASTIC_NU, m_pParam->STOCHASTIC_PHI, _forgetting_rate_phi, m_pParam->NFOLDS,
      m_pParam->INITIAL_ALPHA, m_dC, m_pParam->INITIAL_C2, m_dDeltaEll,
      dBestTstAcc, dBestTstF1, dBestTstAUC, dTstAcc_fst, dF1_fst, dAUC_fst,
      dTrainTime, dTstTime / nIt);
  fprintf(fileptr, "\t svm: %.4f, vi: %.4f, nu: %.4f, phi: %.4f\n", running_time_for_svm,
      running_time_for_vi, running_time_for_update_nu, running_time_for_update_phi);
  fclose(fileptr);

  for (d = 0; d < pC->num_docs; d++) {
    free(phi[d]);
    free(m_dYDist[d]);
  }
  free(phi);
  free(m_dYDist);

  return dBestTstAUC;
}

/*
* perform inference on documents and update sufficient statistics
*/
double LinkSVMDiag::e_step(Corpus *pC, double **var_nu, double **phi) {
  boost::timer Timer;
  Document *pDoc = NULL;
  // compute the expectation statistics
  compute_nu_stat(var_nu);
  for (int i = 0; i < pC->num_docs; i++) {
    pDoc = &(pC->docs[i]);
    for (int j = 0; j < pDoc->num_neighbors; j++) {
      int jIx = pDoc->neighbors[j];
      if (!pDoc->bTrain[j]) continue;

      loss_aug_predict(pDoc, &(pC->docs[jIx]), phi[i], phi[jIx], j);
    }
  }

  double converged = 1, obj_old = 1e70, obj = 0, lhood = 0;
  int var_iter = 0/*, wrd = 0, wcount = 0*/;
  while ((converged > m_pParam->VAR_CONVERGED) && ((var_iter < m_pParam->VAR_MAX_ITER)
      || (m_pParam->VAR_MAX_ITER == -1))) {
    var_iter++;

    // update nu
    update_nu(pC, var_nu, phi);
    compute_nu_stat(var_nu); // compute the expectation statistics

    // update phi without coupling
    for (int d = 0; d < pC->num_docs; d++) {
      if (m_stochastic_phi != 0)
        update_phi(pC, &(pC->docs[d]), phi, d, var_nu, m_stochastic_phi, var_iter);
      else
        update_phi(pC, &(pC->docs[d]), phi, d, var_nu);
    }

    lhood = compute_lhood(pC, phi, var_nu);
    obj = lhood;

    //assert(!isnan(lhood));
    converged = fabs(1 - obj / obj_old);
    obj_old = obj;

    printf("\t%.3f;  %.5f\n", obj, converged);
  }
  double dTrainTime = Timer.elapsed();
  running_time_for_vi += dTrainTime;
  return lhood;
}

void LinkSVMDiag::random_batch(int *target, int batch_size, int start, int end) {
  int len = end - start;
  assert(batch_size <= len);

  int rd = randomMT() % len;
  for (int i = 0; i < batch_size; i++) {
    target[i] = (rd + i) % len + start;
  }
}

double LinkSVMDiag::e_step(Corpus *pC, double **var_nu, double **phi, int batch_size) {
  mini_batch_svm_size = 0;

  Document *pDoc = NULL;
  boost::timer Timer;
  // compute the expectation statistics
  printf("compute_nu_stat\n");
  compute_nu_stat(var_nu);
  // compute_lhood(pC, phi, var_nu);

  double converged = 1, obj_old = 1e70, obj = 0, lhood = 0;
  int var_iter = 0/*, wrd = 0, wcount = 0*/;
  printf("update_nu\n");
  update_nu(pC, var_nu, phi);
  compute_nu_stat(var_nu);
  int *sto = new int[batch_size];

  while (/*(converged > m_pParam->VAR_CONVERGED) && */((var_iter < m_pParam->VAR_MAX_ITER)
      || (m_pParam->VAR_MAX_ITER == -1))) {
    random_batch(sto, batch_size, 0, pC->num_docs);

    // update phi without coupling
    for (int i = 0; i < batch_size; i++) {
      int d = sto[i];
      if (m_stochastic_phi != 0)
        update_phi(pC, &(pC->docs[d]), phi, d, var_nu, m_stochastic_phi, var_iter);
      else
        update_phi(pC, &(pC->docs[d]), phi, d, var_nu);
    }
    // update nu
    update_nu(pC, var_nu, phi, sto, batch_size, var_iter);
    compute_nu_stat(var_nu); // compute the expectation statistics

    lhood = compute_lhood(pC, phi, var_nu);
    obj = lhood;

    //assert(!isnan(lhood));
    converged = fabs(1 - obj / obj_old);
    obj_old = obj;

    printf("\t%.3f;  %.5f\n", obj, converged);

    var_iter++;
  }
  /*
    // update phi without coupling
    for (int d=0; d<pC->num_docs; d++) {
      if (m_stochastic_phi != 0)
        update_phi(pC, &(pC->docs[d]), phi, d, var_nu, m_stochastic_phi, var_iter);
      else
        update_phi(pC, &(pC->docs[d]), phi, d, var_nu);
    }
  */
  delete[]sto;


  double dTrainTime = Timer.elapsed();
  running_time_for_vi += dTrainTime;
  lhood = compute_lhood(pC, phi, var_nu);
  return lhood;
}

/*
 * compute lhood bound
 */
double LinkSVMDiag::compute_lhood(Corpus *pC, double **phi, double **var_nu) {
  double lhood = 0;
  Document *pDoc = NULL;
  double *phiPtr = NULL, *nuPtr = NULL, dPhiVal = 0/*, *wCovPtr=NULL*/;

  // KL-divergence between q(v) and p(v|alpha) (Beta(1, alpha))
  double dKLv = 0;
  for (int k = 0; k < m_nK; k++) {
    nuPtr = var_nu[k];

    double digamma_vsum = m_digamma_vsum_[k];
    dKLv += ((nuPtr[0] - m_alpha) * (m_digamma_v1_[k] - digamma_vsum)
        + (nuPtr[1] - 1) * (m_digamma_v2_[k] - digamma_vsum)
        - (_lgamma(nuPtr[0]) + _lgamma(nuPtr[1]) - _lgamma(nuPtr[0] + nuPtr[1]))
        - log(m_alpha));
  }
  lhood += dKLv;

  // KL-divergence between q(w) and p(w|lambda) (regularizer)
  double dKLw = get_2norm_w();
  lhood += dKLw * 0.5 / m_dWPriorVar;

  // KL-divergence between q(Z) and p(Z)
  double dKLz = 0;
  for (int d = 0; d < pC->num_docs; d++) {
    phiPtr = phi[d];
    for (int k = 0; k < m_nK; k++) {
      dPhiVal = phiPtr[k];
      dKLz -= (dPhiVal * m_dNuExpSum[k] + (1 - dPhiVal) * m_dNuExpProd[k]);

      if (1e-50 < dPhiVal) dKLz += dPhiVal * log(dPhiVal);
      if (1e-50 < 1 - dPhiVal) dKLz += (1 - dPhiVal) * log(1 - dPhiVal);
    }
  }
  lhood += dKLz;
  // hinge loss
  double dHingeLoss = 0;
  for (int i = 0; i < pC->num_docs; i++) {
    pDoc = &(pC->docs[i]);
    // printf("compute hinge loss for doc #%d\n", i);
    for (int j = 0; j < pDoc->num_neighbors; j++) {
      int jIx = pDoc->neighbors[j];
      if (jIx < 0) {
        printf("doc #%d", i);
        for (int k = 0; k < pDoc->num_neighbors; k++) {
          printf("%d ", pDoc->neighbors[k]);
        }
        printf("%d\n");
      }

      if (!pDoc->bTrain[j]) continue;

      if (pDoc->linkGnd[j] == 0)
        dHingeLoss += loss_aug_predict(pDoc, &(pC->docs[jIx]), phi[i], phi[jIx], j) * m_pParam->INITIAL_C1;
      else
        dHingeLoss += loss_aug_predict(pDoc, &(pC->docs[jIx]), phi[i], phi[jIx], j) * m_pParam->INITIAL_C2;
    }
  }
  lhood += dHingeLoss;

  //// entropy of y-distribution for test data
  //double yDistEnt = 0, dDistVal = 0;
  //for ( int i=0; i<pC->num_docs; i++ ) {
  //	pDoc = &(pC->docs[i]);
  //	yDistPtr = m_dYDist[i];
  //	for ( int j=0; j<pDoc->num_neighbors; j++ ) {
  //		if ( pDoc->bTrain[j] ) continue;

  //		dDistVal = yDistPtr[j];
  //		if ( dDistVal > 1e-50 && (1-dDistVal) > 1e-50 )
  //			yDistEnt += (dDistVal * log( dDistVal ) + (1 - dDistVal) * log(1 - dDistVal));
  //	}
  //}
  //lhood += yDistEnt;

  return (lhood);
}

// find the loss-augmented prediction for one document.
double LinkSVMDiag::loss_aug_predict(Document *doc, Document *doc2, double *phi, double *phi2, const int &j) {
  double dMargin = 0, dMaxScore = 0;

  doc->linkLossAug[j] = -1;
  for (int y = 0; y < m_nLabelNum; y++) {
    double dScore = loss(y, doc->linkGnd[j]) + compute_disc_fscore(y, doc, doc2, phi, phi2);

    if (doc->linkLossAug[j] == -1 || dScore > dMaxScore) {
      doc->linkLossAug[j] = y;
      dMaxScore = dScore;
    }

    if (y == doc->linkGnd[j])
      dMargin = 0 - dScore;
  }
  dMargin += dMaxScore;

  return dMargin;
}

double LinkSVMDiag::get_2norm_w() {
  double dRes = 0;

  for (int y = 0; y < m_nLabelNum; y++) {
    double **wPtr = m_dW[y];
    for (int i = 0; i < m_nK; i++) {
      for (int j = 0; j < m_nK; j++) {
        dRes += (wPtr[i][j] * wPtr[i][j]);
      }
    }
    double *wOrgPtr = m_dOrgW[y];
    for (int i = 0; i < m_nOrgFeatures; i++) {
      dRes += wOrgPtr[i] * wOrgPtr[i];
    }
  }

  return dRes;
}

double LinkSVMDiag::compute_disc_fscore(const int &y, Document *doc, Document *doc2, double *phi, double *phi2) {
  double dScore = 0;

  dScore = l2norm2(phi, m_dW[y], phi2, m_nK);

  if (doc == doc2) { // for the self-links.
    double **wPtr = m_dW[y];
    for (int k = 0; k < m_nK; k++) {
      dScore += wPtr[k][k] * phi[k] * (1 - phi[k]);
    }
  }

  double *wOrgPtr = m_dOrgW[y];
  for (int i = 0; i < doc->num_features; i++) {
    dScore += wOrgPtr[i] * doc->feature[i];
  }
  for (int i = 0; i < doc2->num_features; i++) {
    dScore += wOrgPtr[i] * doc2->feature[i];
  }

  return (dScore - m_dB);
}

void LinkSVMDiag::random_init(Corpus *pC, double **phi) {
  // initialize phi to be uniform plus a small random.
  for (int d = 0; d < pC->num_docs; d++) {
    double *phiPtr = phi[d];
    for (int k = 0; k < m_nK; k++) {
      phiPtr[k] = 0.5 + myrand() * 0.001;
    }
  }

  //// initialize y_dist for testing data
  //for ( int d=0; d<pC->num_docs; d++ ) {
  //	Document *pDoc = &(pC->docs[d]);
  //	double *yDistPtr = m_dYDist[d];
  //	for ( int i=0; i<pDoc->num_neighbors; i++ ) {
  //		if ( pDoc->bTrain[i] ) continue;

  //		int jIx = pDoc->neighbors[i];
  //		yDistPtr[i] = safe_logist( l2norm2(phi[d], m_dWMean, phi[jIx], m_nK) );
  //	}
  //}
}

void LinkSVMDiag::random_wvar_nu() {
  // initialize nu & gamma to be the priors.
  for (int k = 0; k < m_nK; k++) {
    m_var_nu[k][0] = m_alpha;
    m_var_nu[k][1] = 1;

    //for ( int m=0; m<m_nK; m++ )
    //	m_dWVar[k][m] = m_dWPriorVar / m_nK;
  }

  for (int y = 0; y < m_nLabelNum; y++) {
    double **wPtr = m_dW[y];
    for (int k = 0; k < m_nK; k++) {
      for (int m = 0; m < m_nK; m++) {
        wPtr[k][m] = myrand() * 0.01;
      }
    }
  }
}

void LinkSVMDiag::random_w() {
  for (int y = 0; y < m_nLabelNum; y++) {
    double **wPtr = m_dW[y];
    for (int k = 0; k < m_nK; k++) {
      for (int m = 0; m < m_nK; m++) {
        wPtr[k][m] = myrand() * 0.1;
      }
    }
  }
}

void LinkSVMDiag::update_phi(Corpus *pC, Document *pDoc, double **phi, const int &docId,
    double **var_nu) {
  //printf("id = %d\n", docId);
  boost::timer Timer;
  //FILE *fptr = fopen("phi_intermediate.txt", "a");
  double *phiPtr = phi[docId];
  for (int k = 0; k < m_nK; k++) {
    double dVal = 0;
    for (int j = 0; j < pDoc->num_neighbors; j++) {
      int jIx = pDoc->neighbors[j];
      if (!pDoc->bTrain[j]) continue;

      Document *pDoc2 = &(pC->docs[jIx]);
      if (jIx != docId) { // different entities
        loss_aug_predict(pDoc, pDoc2, phiPtr, phi[jIx], j);
        dVal += compute_mrgterm_right(pDoc, phi[jIx], j, k);

        int dIx = pDoc2->find_nix(docId);
        if (dIx > 0 && pDoc2->bTrain[dIx]) {
          loss_aug_predict(pDoc2, pDoc, phi[jIx], phiPtr, dIx);
          dVal += compute_mrgterm_left(pDoc2, phi[jIx], dIx, k);
        }
      } else { // same entity
        loss_aug_predict(pDoc, pDoc2, phiPtr, phi[jIx], j);
        dVal += compute_mrgterm_self(pDoc, phi[jIx], j, k);
      }
    }
    dVal += m_dNuExpSum[k] - m_dNuExpProd[k];
    //printf("%lf\n", dVal);
    //system("pause");
    phiPtr[k] = safe_logist(dVal);
  }
  double dTrainTime = Timer.elapsed();
  running_time_for_update_phi += dTrainTime;
}

void LinkSVMDiag::update_phi(Corpus *pC, Document *pDoc, double **phi, const int &docId,
    double **var_nu, const int batch_size, int iter) {
  //printf("id = %d\n", docId);
  boost::timer Timer;
  //printf("sto update phi b-size %d\n", batch_size);
  //FILE *fptr = fopen("phi_intermediate.txt", "a");
  double *phiPtr = phi[docId];

  int real_neighbors = 0, real_infer_links = 0;
  for (int j = 0; j < pDoc->num_neighbors; j++) {
    if (!pDoc->bTrain[j]) continue;
    real_neighbors++;
    int jIx = pDoc->neighbors[j];
    Document *pDoc2 = &(pC->docs[jIx]);
    if (jIx != docId) {
      int dIx = pDoc2->find_nix(docId);
      if (dIx > -1 && pDoc2->bTrain[dIx]) {
        real_infer_links++;
      }
    }
    real_infer_links++;
  }
  int real_batch_size = std::min(pDoc->num_neighbors, batch_size);
  //printf("rbs = %d, rn = %d, rif = %d\n", real_batch_size, real_neighbors, real_infer_links);
  for (int k = 0; k < m_nK; k++) {
    double dVal = 0;
    int infer_links = 0;
    //for (int i = 0, j = randomMT() % pDoc->num_neighbors;
    //  i < real_batch_size; i++, j = (j + randomMT()) % pDoc->num_neighbors) {
    for (int i = 0, j = (randomMT()) % (pDoc->num_neighbors);
         i < real_batch_size; j = (j + 1) % (pDoc->num_neighbors)) {

      if (!pDoc->bTrain[j]) {
        continue;
      }
      i++;
      int jIx = pDoc->neighbors[j];
      Document *pDoc2 = &(pC->docs[jIx]);

      if (jIx != docId) { // different entities
        loss_aug_predict(pDoc, pDoc2, phiPtr, phi[jIx], j);
        dVal += compute_mrgterm_right(pDoc, phi[jIx], j, k);


        infer_links++;
        int dIx = pDoc2->find_nix(docId);
        if (dIx > -1 && pDoc2->bTrain[dIx]) {
          loss_aug_predict(pDoc2, pDoc, phi[jIx], phiPtr, dIx);
          dVal += compute_mrgterm_left(pDoc2, phi[jIx], dIx, k);

          infer_links++;
        }
      } else { // same entity
        loss_aug_predict(pDoc, pDoc2, phiPtr, phi[jIx], j);
        dVal += compute_mrgterm_self(pDoc, phi[jIx], j, k);


        infer_links++;
      }
    }
    //printf("\nmini = %d\n", mini_batch_svm_size);

    dVal *= (double) real_infer_links / (double) infer_links;
    //printf("%lf\n", dVal);
    dVal += m_dNuExpSum[k] - m_dNuExpProd[k];

    //system("pause");
    double newPhi = safe_logist(dVal);
    // step size at iteration ``iter``
    double rou = step_size(iter, _delay_phi, _forgetting_rate_phi);
    phiPtr[k] = (1 - rou) * phiPtr[k] + rou * newPhi;
  }
  double dTrainTime = Timer.elapsed();
  running_time_for_update_phi += dTrainTime;
}

double LinkSVMDiag::compute_mrgterm_right(Document *doc, double *phi, const int &j, const int &k) {
  double dval = 0;
  int gnd = doc->linkGnd[j];

  if (gnd != doc->linkLossAug[j]) {// right product
    dval = (dotprod(m_dW[gnd][k], phi, m_nK) - dotprod(m_dW[doc->linkLossAug[j]][k], phi, m_nK));

    if (gnd == 0) dval *= m_pParam->INITIAL_C1;
    else dval *= m_pParam->INITIAL_C2;
  }

  return dval;
}

double LinkSVMDiag::compute_mrgterm_left(Document *doc, double *phi, const int &j, const int &k) {
  double dval = 0;
  int gnd = doc->linkGnd[j];

  if (gnd != doc->linkLossAug[j]) {// left product
    double **dGndW = m_dW[gnd];
    double **dLossAugW = m_dW[doc->linkLossAug[j]];
    for (int i = 0; i < m_nK; i++) {
      dval += (dGndW[i][k] - dLossAugW[i][k]) * phi[i];
    }

    if (gnd == 0) dval *= m_pParam->INITIAL_C1;
    else dval *= m_pParam->INITIAL_C2;
  }

  return dval;
}

double LinkSVMDiag::compute_mrgterm_self(Document *doc, double *phi, const int &j, const int &k) {
  double dval = 0;
  int gnd = doc->linkGnd[j];

  if (gnd != doc->linkLossAug[j]) {
    dval = (dotprod(m_dW[gnd][k], phi, m_nK) - dotprod(m_dW[doc->linkLossAug[j]][k], phi, m_nK));

    dval += (m_dW[gnd][k][k] - m_dW[doc->linkLossAug[j]][k][k]) * (1 - phi[k]);

    if (gnd == 0) dval *= m_pParam->INITIAL_C1;
    else dval *= m_pParam->INITIAL_C2;
  }

  return dval;
}

void LinkSVMDiag::update_nu(Corpus *pC, double **var_nu, double **phi) {
  // double *phiPtr = NULL;
  boost::timer Timer;
  // sum the expected number of features.
  memset(m_exp_feat_num_, 0, sizeof(double) * m_nK);
  for (int d = 0; d < pC->num_docs; d++) {
    addvec(m_exp_feat_num_, phi[d], m_nK);
  }
  for (int k = 0; k < m_nK; k++) {
    m_exp_feat_n_minus_num_[k] = pC->num_docs - m_exp_feat_num_[k];
  }

  // update nu parameters
  for (int k = 0; k < m_nK; k++) {
    double dVal1 = m_alpha, dVal2 = 1;

    for (int m = k; m < m_nK; m++) {
      dVal1 += m_exp_feat_num_[m];
      dVal2 += m_exp_feat_n_minus_num_[m] * m_dQ_dist_[m][k];
    }

    for (int m = k + 1; m < m_nK; m++) {
      double qDistSum = 0;
      for (int i = k + 1; i <= m; i++)
        qDistSum += m_dQ_dist_[m][i];
      dVal1 += m_exp_feat_n_minus_num_[m] * qDistSum;
    }

    // the shape parameter of Beta dist.
    var_nu[k][0] = dVal1;
    var_nu[k][1] = dVal2;
  }
  double dTrainTime = Timer.elapsed();
  running_time_for_update_nu += dTrainTime;
}

void LinkSVMDiag::update_nu(Corpus *pC, double **var_nu, double **phi, int *batch, int batch_size, int iter) {
  //double *phiPtr = NULL;
  boost::timer Timer;
  // sum the expected number of features.
  memset(m_exp_feat_num_, 0, sizeof(double) * m_nK);
  for (int d = 0; d < pC->num_docs; d++) {
    addvec(m_exp_feat_num_, phi[d], m_nK);
  }
  for (int k = 0; k < m_nK; k++) {
    m_exp_feat_n_minus_num_[k] = pC->num_docs - m_exp_feat_num_[k];
  }

  double *tmp = new double[m_nK];
  for (int i = 0; i < m_nK; i++) tmp[i] = 0;
  for (int i = 0; i < batch_size; i++) {
    addvec(tmp, phi[batch[i]], m_nK);
  }

  // update nu parameters
  for (int k = 0; k < m_nK; k++) {
    double dVal1 = m_alpha, dVal2 = 1;

    for (int m = k; m < m_nK; m++) {
      dVal1 += tmp[m] * pC->num_docs / batch_size;
      dVal2 += (pC->num_docs - tmp[m] * pC->num_docs / batch_size) * m_dQ_dist_[m][k];
    }

    for (int m = k + 1; m < m_nK; m++) {
      double qDistSum = 0;
      for (int i = k + 1; i <= m; i++)
        qDistSum += m_dQ_dist_[m][i];
      dVal1 += (pC->num_docs - tmp[m] * pC->num_docs / batch_size) * qDistSum;
    }


    // step size at iteration ``iter``
    double rou = step_size(iter, _delay_nu, _forgetting_rate_nu);

    // the shape parameter of Beta dist.
    var_nu[k][0] = (1 - rou) * var_nu[k][0] + rou * dVal1;
    var_nu[k][1] = (1 - rou) * var_nu[k][0] + rou * dVal2;

  }
  delete[]tmp;
  double dTrainTime = Timer.elapsed();
  running_time_for_update_nu += dTrainTime;
}

void LinkSVMDiag::compute_nu_stat(double **var_nu) {
  double dNuExpSum = 0, *nuPtr = NULL;
  for (int k = 0; k < m_nK; k++) {
    nuPtr = var_nu[k];
    m_digamma_v1_[k] = digamma(nuPtr[0]);
    m_digamma_v2_[k] = digamma(nuPtr[1]);
    m_digamma_vsum_[k] = digamma(nuPtr[0] + nuPtr[1]);
    dNuExpSum += (m_digamma_v1_[k] - m_digamma_vsum_[k]);

    // expectation of sum of log v
    m_dNuExpSum[k] = dNuExpSum;

    // expectation of log(1 - prod v)
    m_dNuExpProd[k] = compute_nu_exp_prod(k);
  }
}

// compute the auxiliary dist q and expectation of prod v
double LinkSVMDiag::compute_nu_exp_prod(const int &k) {
  double *qDist = m_dQ_dist_[k];
  double digamma_vsum_acc = 0;
  double digamma_v1_acc = 0;
  double qDistSum = 0;

  // compute the auxiliary distribution q.
  for (int i = 0; i <= k; i++) {
    digamma_vsum_acc += m_digamma_vsum_[i];
    qDist[i] = exp(m_digamma_v2_[i] + digamma_v1_acc - digamma_vsum_acc);
    qDistSum += qDist[i];
    digamma_v1_acc += m_digamma_v1_[i];
  }
  for (int i = 0; i <= k; i++) {
    qDist[i] /= qDistSum;
  }

  // compute the lower bound.
  double dBound = 0;
  for (int m = 0; m <= k; m++) {
    dBound += qDist[m] * m_digamma_v2_[m];
  }
  for (int m = 0; m < k; m++) {
    qDistSum = 0;
    for (int i = m + 1; i <= k; i++) {
      qDistSum += qDist[i];
    }
    dBound += qDistSum * m_digamma_v1_[m];

    dBound -= (qDistSum + qDist[m]) * m_digamma_vsum_[m];
  }
  dBound -= qDist[k] * m_digamma_vsum_[k];

  // entropy term.
  dBound += safe_entropy(qDist, (k + 1));

  return dBound;
}

void LinkSVMDiag::update_ydist(Corpus *pC, double **phi) {
  for (int d = 0; d < pC->num_docs; d++) {
    Document *pDoc = &(pC->docs[d]);
    double *yDistPtr = m_dYDist[d];
    double *phiPtr = phi[d];

    for (int i = 0; i < pDoc->num_neighbors; i++) {
      if (pDoc->bTrain[i]) continue;

      int jIx = pDoc->neighbors[i];

      double dval = compute_disc_fscore(1, pDoc, &(pC->docs[jIx]), phiPtr, phi[jIx]) -
          compute_disc_fscore(0, pDoc, &(pC->docs[jIx]), phiPtr, phi[jIx]);
      //l2norm2(phiPtr, m_dW[1], phi[jIx], m_nK) - l2norm2(phiPtr, m_dW[0], phi[jIx], m_nK);
      yDistPtr[i] = safe_logist(dval);
    }
  }
}

void LinkSVMDiag::update_ydist_tr(Corpus *pC, double **phi) {
  for (int d = 0; d < pC->num_docs; d++) {
    Document *pDoc = &(pC->docs[d]);
    double *yDistPtr = m_dYDist[d];
    double *phiPtr = phi[d];

    for (int i = 0; i < pDoc->num_neighbors; i++) {
      if (!pDoc->bTrain[i]) continue;

      int jIx = pDoc->neighbors[i];

      double dval = compute_disc_fscore(1, pDoc, &(pC->docs[jIx]), phiPtr, phi[jIx]) -
          compute_disc_fscore(0, pDoc, &(pC->docs[jIx]), phiPtr, phi[jIx]);
      //l2norm2(phiPtr, m_dW[1], phi[jIx], m_nK) - l2norm2(phiPtr, m_dW[0], phi[jIx], m_nK);
      yDistPtr[i] = safe_logist(dval);
    }
  }
}

void LinkSVMDiag::sample_init(Corpus *pC, double **phi, STRUCT_LEARN_PARM *struct_parm) {
  char buff[512];
  sprintf(buff, "%s/Feature_%d_%d_%d.txt", m_dir, (int) (100 * m_pParam->INITIAL_ALPHA),
      (int) (100 * m_pParam->INITIAL_C1), (int) (100 * m_pParam->INITIAL_C2));

  outputLowDimData(buff, pC, phi);

  /* read the training examples */
  m_sample = read_struct_examples(buff, struct_parm);
}

void LinkSVMDiag::sample_init2(Corpus *pC, double **phi, STRUCT_LEARN_PARM *struct_parm) {
  DOC **docs = (DOC **) my_malloc(sizeof(DOC *) * m_nTrainLinks);    /* feature vectors */
  double *label = (double *) my_malloc(sizeof(double) * m_nTrainLinks); /* target values */
  TOKEN *words = (TOKEN *) my_malloc(sizeof(TOKEN) * (m_nSVMFeature + 10));

  long dnum = 0;
  long totwords = 0;
  long num_classes = 0;

  for (int d = 0; d < pC->num_docs; d++) {
    Document *pDoc = &(pC->docs[d]);
    for (int k = 0; k < pDoc->num_neighbors; k++) {
      if (!pDoc->bTrain[k]) continue;
      int jIx = pDoc->neighbors[k];
      label[dnum] = pDoc->linkGnd[k] + 1;
      get_fvec(pDoc, &(pC->docs[jIx]), phi[d], phi[jIx], fvec_);

      long queryid = 0;
      long slackid = 0;
      double costfactor = 1;
      char *comment = NULL;
      long wpos = 0;
      for (int m = 0; m < m_nSVMFeature; m++) {
        (words[wpos]).wnum = m + 1;
        (words[wpos]).weight = fvec_[m];
        wpos++;
      }
      (words[wpos]).wnum = 0;

      if ((wpos > 0) && ((words[wpos - 1]).wnum > totwords))
        totwords = (words[wpos - 1]).wnum;
      if (totwords > MAXFEATNUM) {
        printf("\nMaximum feature number exceeds limit defined in MAXFEATNUM!\n");
        exit(1);
      }
      docs[dnum] = create_example(dnum, queryid, slackid, costfactor,
          create_svector(words, comment, 1.0));

      if (num_classes < (label[dnum] + 0.1))
        num_classes = label[dnum] + 0.1;

      dnum++;
    }
  }


  EXAMPLE *examples;
  examples = (EXAMPLE *) my_malloc(sizeof(EXAMPLE) * dnum);
  for (long i = 0; i < dnum; i++) {          /* copy docs over into new datastructure */
    examples[i].x.doc = docs[i];
    examples[i].y.classlabel = label[i] + 0.1;
    examples[i].y.scores = NULL;
    examples[i].y.num_classes = num_classes;
  }
  free(label);
  free(docs);
  //free(words);
  m_sample.n = dnum;
  m_sample.examples = examples;
  printf("# of training data: %d; class-num: %d\n", m_sample.n, (int) num_classes);
}

void LinkSVMDiag::set_init_param(STRUCT_LEARN_PARM *struct_parm, LEARN_PARM *learn_parm,
    KERNEL_PARM *kernel_parm, int *alg_type) {
  /* set default */
  (*alg_type) = DEFAULT_ALG_TYPE;
  struct_parm->C = -0.01;
  struct_parm->slack_norm = 1;
  struct_parm->epsilon = DEFAULT_EPS;
  struct_parm->custom_argc = 0;
  struct_parm->loss_function = DEFAULT_LOSS_FCT;
  struct_parm->loss_type = DEFAULT_RESCALING;
  struct_parm->newconstretrain = 100;
  struct_parm->ccache_size = 5;
  struct_parm->batch_size = 100;
  struct_parm->delta_ell = m_pParam->DELTA_ELL;

  strcpy(learn_parm->predfile, "trans_predictions");
  strcpy(learn_parm->alphafile, "");
  verbosity = 0;/*verbosity for svm_light*/
  struct_verbosity = 1; /*verbosity for struct learning portion*/
  learn_parm->biased_hyperplane = 1;
  learn_parm->remove_inconsistent = 0;
  learn_parm->skip_final_opt_check = 0;
  learn_parm->svm_maxqpsize = 10;
  learn_parm->svm_newvarsinqp = 0;
  learn_parm->svm_iter_to_shrink = -9999;
  learn_parm->maxiter = 100000;
  learn_parm->kernel_cache_size = 40;
  learn_parm->svm_c = 99999999;  /* overridden by struct_parm->C */
  learn_parm->eps = 0.001;       /* overridden by struct_parm->epsilon */
  learn_parm->transduction_posratio = -1.0;
  learn_parm->svm_costratio = 1.0;
  learn_parm->svm_costratio_unlab = 1.0;
  learn_parm->svm_unlabbound = 1E-5;
  learn_parm->epsilon_crit = 0.001;
  learn_parm->epsilon_a = 1E-10;  /* changed from 1e-15 */
  learn_parm->compute_loo = 0;
  learn_parm->rho = 1.0;
  learn_parm->xa_depth = 0;
  kernel_parm->kernel_type = 0;
  kernel_parm->poly_degree = 3;
  kernel_parm->rbf_gamma = 1.0;
  kernel_parm->coef_lin = 1;
  kernel_parm->coef_const = 1;
  strcpy(kernel_parm->custom, "empty");

  if (learn_parm->svm_iter_to_shrink == -9999) {
    learn_parm->svm_iter_to_shrink = 100;
  }

  if ((learn_parm->skip_final_opt_check)
      && (kernel_parm->kernel_type == LINEAR)) {
    printf("\nIt does not make sense to skip the final optimality check for linear kernels.\n\n");
    learn_parm->skip_final_opt_check = 0;
  }
  if ((learn_parm->skip_final_opt_check)
      && (learn_parm->remove_inconsistent)) {
    printf("\nIt is necessary to do the final optimality check when removing inconsistent \nexamples.\n");
    exit(0);
  }
  if ((learn_parm->svm_maxqpsize < 2)) {
    printf("\nMaximum size of QP-subproblems not in valid range: %ld [2..]\n", learn_parm->svm_maxqpsize);
    exit(0);
  }
  if ((learn_parm->svm_maxqpsize < learn_parm->svm_newvarsinqp)) {
    printf("\nMaximum size of QP-subproblems [%ld] must be larger than the number of\n", learn_parm->svm_maxqpsize);
    printf("new variables [%ld] entering the working set in each iteration.\n", learn_parm->svm_newvarsinqp);
    exit(0);
  }
  if (learn_parm->svm_iter_to_shrink < 1) {
    printf("\nMaximum number of iterations for shrinking not in valid range: %ld [1,..]\n", learn_parm->svm_iter_to_shrink);
    exit(0);
  }
  if (((*alg_type) < 0) || (((*alg_type) > 5) && ((*alg_type) != 9))) {
    printf("\nAlgorithm type must be either '0', '1', '2', '3', '4', or '9'!\n\n");
    exit(0);
  }
  if (learn_parm->transduction_posratio > 1) {
    printf("\nThe fraction of unlabeled examples to classify as positives must\n");
    printf("be less than 1.0 !!!\n\n");
    exit(0);
  }
  if (learn_parm->svm_costratio <= 0) {
    printf("\nThe COSTRATIO parameter must be greater than zero!\n\n");
    exit(0);
  }
  if (struct_parm->epsilon <= 0) {
    printf("\nThe epsilon parameter must be greater than zero!\n\n");
    exit(0);
  }
  if ((struct_parm->ccache_size <= 0) && ((*alg_type) == 4)) {
    printf("\nThe cache size must be at least 1!\n\n");
    exit(0);
  }
  if (((struct_parm->batch_size <= 0) || (struct_parm->batch_size > 100))
      && ((*alg_type) == 4)) {
    printf("\nThe batch size must be in the interval ]0,100]!\n\n");
    exit(0);
  }
  if ((struct_parm->slack_norm < 1) || (struct_parm->slack_norm > 2)) {
    printf("\nThe norm of the slacks must be either 1 (L1-norm) or 2 (L2-norm)!\n\n");
    exit(0);
  }
  if ((struct_parm->loss_type != SLACK_RESCALING)
      && (struct_parm->loss_type != MARGIN_RESCALING)) {
    printf("\nThe loss type must be either 1 (slack rescaling) or 2 (margin rescaling)!\n\n");
    exit(0);
  }
  if (learn_parm->rho < 0) {
    printf("\nThe parameter rho for xi/alpha-estimates and leave-one-out pruning must\n");
    printf("be greater than zero (typically 1.0 or 2.0, see T. Joachims, Estimating the\n");
    printf("Generalization Performance of an SVM Efficiently, ICML, 2000.)!\n\n");
    exit(0);
  }
  if ((learn_parm->xa_depth < 0) || (learn_parm->xa_depth > 100)) {
    printf("\nThe parameter depth for ext. xi/alpha-estimates must be in [0..100] (zero\n");
    printf("for switching to the conventional xa/estimates described in T. Joachims,\n");
    printf("Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)\n");
    exit(0);
  }

  parse_struct_parameters(struct_parm);
}

void LinkSVMDiag::learn_svm(Corpus *pC, double **phi, double *dMu) {
  boost::timer Timer;
  LEARN_PARM learn_parm;
  KERNEL_PARM kernel_parm;
  STRUCT_LEARN_PARM struct_parm;
  STRUCTMODEL structmodel;
  int alg_type;

  /* set the parameters. */
  set_init_param(&struct_parm, &learn_parm, &kernel_parm, &alg_type);
  struct_parm.C = m_dC;
  printf("set init param complete\n");
  sample_init2(pC, phi, &struct_parm);
  printf("sample init2 complete\n");
  if (m_pParam->SVM_ALGTYPE == 0)
    svm_learn_struct(m_sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, NSLACK_ALG);
    //else if(alg_type == 1)
    //	svm_learn_struct(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, NSLACK_SHRINK_ALG);
  else if (m_pParam->SVM_ALGTYPE == 2) {
    struct_parm.C = struct_parm.C * m_sample.n;   // Note: in n-slack formulation, C is not divided by N.
    svm_learn_struct_joint(m_sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, ONESLACK_PRIMAL_ALG);
  }
    //else if(alg_type == 3)
    //	svm_learn_struct_joint(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, ONESLACK_DUAL_ALG);
    //else if(alg_type == 4)
    //	svm_learn_struct_joint(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, ONESLACK_DUAL_CACHE_ALG);
    //else if(alg_type == 9)
    //	svm_learn_struct_joint_custom(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel);
  else exit(1);

  /* get the optimal lagrangian multipliers.
  *    Note: for 1-slack formulation: the "marginalization" is
  *           needed for fast computation.
  */
  int nVar = m_sample.n * m_nLabelNum;
  if (dMu != NULL) {
    for (int k = 0; k < nVar; k++) dMu[k] = 0;

    if (m_pParam->SVM_ALGTYPE == 0) {
      for (int k = 1; k < structmodel.svm_model->sv_num; k++) {
        int docnum = structmodel.svm_model->supvec[k]->orgDocNum;
        dMu[docnum] = structmodel.svm_model->alpha[k];
      }
    } else if (m_pParam->SVM_ALGTYPE == 2) {
      for (int k = 1; k < structmodel.svm_model->sv_num; k++) {
        int *vecLabel = structmodel.svm_model->supvec[k]->lvec;

        double dval = structmodel.svm_model->alpha[k] / m_sample.n;
        for (int d = 0; d < pC->num_docs; d++) {
          Document *pDoc = &(pC->docs[d]);
          for (int i = 0; i < pDoc->num_neighbors; i++) {
            if (!pDoc->bTrain[i]) continue; // train data only

            int label = vecLabel[pDoc->trainIx[i]];
            dMu[pDoc->trainIx[i] * m_nLabelNum + label] += dval;
          }
        }
      }
    } else;

#ifdef _DEBUG
		FILE *fileptr = fopen("MuSolution.txt", "a");
		for ( int i=0; i<pC->num_docs; i++ ) {
			if ( !pC->docs[i].bTrain ) break; // train data only

			for ( int k=0; k<m_nLabelNum; k++ ) {
				int muIx = i * m_nLabelNum + k;
				if ( dMu[muIx] > 0 ) fprintf(fileptr, "%d:%.5f ", k, dMu[muIx]);
			}
			fprintf(fileptr, "\n");
		}
		fprintf(fileptr, "\n\n");
		fclose(fileptr);
#endif
  }

  int rowIx, colIx;
  m_dB = structmodel.svm_model->b;
  for (int y = 0; y < m_nLabelNum; y++) {
    int nRefIx = y * m_nSVMFeature;
    double **wPtr = m_dW[y];
    for (int i = 0; i < m_nLatentFeatures; i++) {
      int wIx = nRefIx + i;
      get_index(i, rowIx, colIx);
      wPtr[rowIx][colIx] = structmodel.w[wIx + 1];
    }
    double *wOrgPtr = m_dOrgW[y];
    for (int i = 0; i < m_nOrgFeatures; i++) {
      int wIx = nRefIx + m_nLatentFeatures + i;
      wOrgPtr[i] = structmodel.w[wIx + 1];
    }
  }
  m_dsvm_primalobj = structmodel.primalobj;

  // free the memory
  free_struct_sample(m_sample);
  free_struct_model(structmodel);
  double dTrainTime = Timer.elapsed();
  running_time_for_svm += dTrainTime;
}

void LinkSVMDiag::get_index(const int &ix, int &rowIx, int &colIx) {
  rowIx = ix / 3;
  colIx = (ix % 3 - 1 + rowIx + m_nK) % m_nK;
}

void LinkSVMDiag::outputLowDimData(char *filename, Corpus *pC, double **phi) {
  Document *pDoc = NULL;
  double *phiPtr = NULL, *phiPtr2 = NULL;

  double costFactor = m_pParam->INITIAL_C2 / m_pParam->INITIAL_C1;
  FILE *fileptr = fopen(filename, "w");
  for (int d = 0; d < pC->num_docs; d++) {
    pDoc = &(pC->docs[d]);
    phiPtr = phi[d];
    for (int i = 0; i < pDoc->num_neighbors; i++) {
      int jIx = pDoc->neighbors[i];
      if (!pDoc->bTrain[i]) continue;
      phiPtr2 = phi[jIx];

      fprintf(fileptr, "%d %d", m_nSVMFeature, pDoc->linkGnd[i]);
      if (pDoc->linkGnd[i] == 1)
        fprintf(fileptr, " cost:%.3f", costFactor);
      get_fvec(pDoc, &(pC->docs[jIx]), phiPtr, phiPtr2, fvec_);
      for (int m = 0; m < m_nSVMFeature; m++) {
        fprintf(fileptr, " %d:%.10f", m, fvec_[m]);
      }

      //for ( int m=0; m<m_nK; m++ ) {// latent features.
      //	for ( int n=0; n<m_nK; n++ ) {
      //		fprintf(fileptr, " %d:%.10f", (m*m_nK + n), phiPtr[m]*phiPtr2[n]);
      //	}
      //}

      //// output input features.
      //int ix = m_nLatentFeatures;
      //for ( int m=0; m<pDoc->num_features; m++ ) {
      //	fprintf(fileptr, " %d:%.5f", ix, pDoc->feature[m]);
      //	ix ++;
      //}
      //for ( int m=0; m<pC->docs[jIx].num_features; m++ ) {
      //	fprintf(fileptr, " %d:%.5f", ix, pC->docs[jIx].feature[m]);
      //	ix ++;
      //}
      fprintf(fileptr, "\n");
    }
  }
  fclose(fileptr);
}

void LinkSVMDiag::get_fvec(Document *pDoc1, Document *pDoc2, double *phi1, double *phi2, double *fvec) {
  int ix = 0;
  for (int m = 0; m < m_nK; m++) {// latent features.
    for (int n = -1; n < 2; n++) {
      int p = (m + n + m_nK) % m_nK;
      fvec[ix] = phi1[m] * phi2[p];
      ix++;
    }
  }

  // output input features.
  for (int m = 0; m < pDoc1->num_features; m++) {
    fvec[ix] = pDoc1->feature[m];
    ix++;
  }
  for (int m = 0; m < pDoc2->num_features; m++) {
    fvec[ix] = pDoc2->feature[m];
    ix++;
  }
}

void LinkSVMDiag::save_model(char *model_root, Corpus *pC, double **phi) {
  char filename[512];
  FILE *fileptr;

  sprintf(filename, "%s.eta", model_root);
  fileptr = fopen(filename, "w");
  fprintf(fileptr, "%5.10f\n", m_dB);
  fclose(fileptr);

  sprintf(filename, "%s.param", model_root);
  m_pParam->write_settings(filename);

  sprintf(filename, "%s.phi", model_root);
  save_mat(filename, phi, m_nData, m_nK);

  sprintf(filename, "%s.ydist", model_root);
  fileptr = fopen(filename, "w");
  for (int i = 0; i < m_nData; i++) {
    double *yDistPtr = m_dYDist[i];
    Document *pDoc = &(pC->docs[i]);
    bool bEmpty = true;
    for (int j = 0; j < pDoc->num_neighbors; j++) {
      if (!pDoc->bTrain[j]) {
        fprintf(fileptr, "%d:%.10f ", pDoc->linkGnd[j], yDistPtr[j]);
        bEmpty = false;
      }
    }
    if (!bEmpty) fprintf(fileptr, "\n");
  }
  fclose(fileptr);

  if (truePos_ != NULL) {
    sprintf(filename, "%s.roc", model_root);
    fileptr = fopen(filename, "w");
    for (int i = 0; i < m_nTstLinks; i++) {
      fprintf(fileptr, "%.4f\t%.4f\n", falsePos_[i], truePos_[i]);
    }
    fclose(fileptr);
  }

  sprintf(filename, "%s.nu", model_root);
  save_mat(filename, m_var_nu, m_nK, 2);
  sprintf(filename, "%s.nuexpsum", model_root);
  save_vec(filename, m_dNuExpSum, m_nK);
  sprintf(filename, "%s.nuexpprod", model_root);
  save_vec(filename, m_dNuExpProd, m_nK);

  sprintf(filename, "%s.w0", model_root);
  save_mat(filename, m_dW[0], m_nK, m_nK);
  sprintf(filename, "%s.w1", model_root);
  save_mat(filename, m_dW[1], m_nK, m_nK);

  sprintf(filename, "%s.other", model_root);
  fileptr = fopen(filename, "w");
  fprintf(fileptr, "num_dim %d\n", m_nK);
  fprintf(fileptr, "num_labels %d\n", m_nLabelNum);
  fprintf(fileptr, "num_docs %d\n", m_nData);
  fprintf(fileptr, "alpha %f\n", m_alpha);
  fprintf(fileptr, "KL-C %5.10f\n", m_pParam->INITIAL_C2);
  fprintf(fileptr, "svm-C %5.10f\n", m_pParam->INITIAL_C1);
  fclose(fileptr);
}

void LinkSVMDiag::load_model(char *model_root) {
  char filename[512];
  FILE *fileptr;
  int /*i, j, num_terms, */num_topics, num_labels, num_docs, num_train_links;
  float C/*, learnRate, x, alpha, */;
  vector<double> vecAlpha;

  m_pParam = new Params();
  sprintf(filename, "%s.param", model_root);
  m_pParam->read_settings(filename);

  sprintf(filename, "%s.other", model_root);
  printf("loading %s\n", filename);
  fileptr = fopen(filename, "r");
  fscanf(fileptr, "num_dim %d\n", &num_topics);
  fscanf(fileptr, "num_labels %d\n", &num_labels);
  fscanf(fileptr, "num_docs %d\n", &num_docs);
  fscanf(fileptr, "num_train_links %d\n", &num_train_links);
  fscanf(fileptr, "alpha %lf\n", &m_alpha);
  fscanf(fileptr, "KL-C %f\n", &C);
  fscanf(fileptr, "svm-C %f\n", &C);

  fclose(fileptr);

  new_model(num_docs, num_train_links);

  sprintf(filename, "%s.eta", model_root);
  printf("loading %s\n", filename);
  fileptr = fopen(filename, "r");
  fscanf(fileptr, "%lf\n", &m_dB);
  fclose(fileptr);

  sprintf(filename, "%s.nu", model_root);
  load_mat(filename, m_var_nu, m_nK, 2);

  sprintf(filename, "%s.w0", model_root);
  load_mat(filename, m_dW[0], m_nK, m_nK);
  sprintf(filename, "%s.w1", model_root);
  load_mat(filename, m_dW[1], m_nK, m_nK);
}

double LinkSVMDiag::get_test_acc(Corpus *pC, double **phi) {
  for (int i = 0; i < 2; i++) {
    dAVec[i] = 0;
    dBVec[i] = 0;
    dCVec[i] = 0;
  }

  double dAcc = 0, nTstData = 0;
  for (int d = 0; d < pC->num_docs; d++) {
    Document *pDoc = &(pC->docs[d]);
    for (int i = 0; i < pDoc->num_neighbors; i++) {
      if (pDoc->bTrain[i]) continue;

      predict(pDoc, i, m_dYDist[d]);
      if (pDoc->linkGnd[i] == pDoc->linkTest[i]) {
        dAcc += 1;
        dAVec[pDoc->linkGnd[i]]++;
      } else {
        dBVec[pDoc->linkTest[i]]++;
        dCVec[pDoc->linkGnd[i]]++;
      }
      nTstData++;
    }
  }
  dAcc /= nTstData;

  m_dF1Score = 0;
  for (int i = 0; i < 2; i++) {
    m_dF1Score += (2 * dAVec[i]) / (2 * dAVec[i] + dBVec[i] + dCVec[i]);
  }
  m_dF1Score /= 2;

  return dAcc;
}

void LinkSVMDiag::predict(Document *doc, const int &i, double *yDist) {
  doc->linkTest[i] = (yDist[i] >= 0.5) ? 1 : 0;
}

// compute the AUC score
double LinkSVMDiag::compute_auc(Corpus *pC, double **yDist) {
  //vector<int> gndLabel;
  //vector<double> predScore;

  vector<std::pair<double, int> > ScrLbl;
  // retrieve the test links (true label & predict score)
  int nIx = 0, nPos = 0, nNeg = 0;
  for (int i = 0; i < pC->num_docs; i++) {
    Document *pDoc = &(pC->docs[i]);
    double *yDistPtr = yDist[i];
    for (int j = 0; j < pDoc->num_neighbors; j++) {
      //int jIx = pDoc->neighbors[j];
      if (pDoc->bTrain[j]) continue;
      ScrLbl.push_back(make_pair(yDistPtr[j], pDoc->linkGnd[j]));
      //gndLabel.push_back( pDoc->linkGnd[j] );
      //predScore.push_back( yDistPtr[j] );
      nPos += pDoc->linkGnd[j];
      nNeg += (1 - pDoc->linkGnd[j]);
      nIx++;
    }
  }

  std::stable_sort(ScrLbl.begin(), ScrLbl.end(), pair_cmp_fst_gt<double, int>);

  // compute true-positive & false-positive rates
  int nAccPos = 0;
  m_nTstLinks = nIx;
  if (truePos_ == NULL) truePos_ = (double *) malloc(sizeof(double) * nIx);
  if (falsePos_ == NULL) falsePos_ = (double *) malloc(sizeof(double) * nIx);
  memset(truePos_, 0, sizeof(double) * nIx);
  memset(falsePos_, 0, sizeof(double) * nIx);
  for (int i = 0; i < nIx; i++) {
    nAccPos += ScrLbl[i].second;
    if (nPos == 0)
      truePos_[i] = 1;
    else
      truePos_[i] = (double) nAccPos / (double) nPos;
    falsePos_[i] = (double) (i + 1 - nAccPos) / (double) nNeg;
  }

  // compute the AUC score
  vector<int> flags;
  flags.push_back(0);
  for (int i = 1; i < nIx; i++) {
    if (ScrLbl[i].first < ScrLbl[i - 1].first)
      flags.push_back(i);
  }
  flags.push_back(nIx - 1);

  double dAuc = 0;
  for (unsigned int i = 1; i < flags.size(); i++) {
    int ix1 = flags[i - 1];
    int ix2 = flags[i];
    dAuc += (falsePos_[ix2] - falsePos_[ix1])
        * ((truePos_[ix2] + truePos_[ix1]) / 2);
  }

  return dAuc;
}

// compute the AUC score
double LinkSVMDiag::compute_auc_tr(Corpus *pC, double **yDist) {
  vector<std::pair<double, int> > ScrLbl;

  // retrieve the test links (true label & predict score)
  int nIx = 0, nPos = 0, nNeg = 0;
  for (int i = 0; i < pC->num_docs; i++) {
    Document *pDoc = &(pC->docs[i]);
    double *yDistPtr = yDist[i];
    for (int j = 0; j < pDoc->num_neighbors; j++) {
      //int jIx = pDoc->neighbors[j];
      if (!pDoc->bTrain[j]) continue;

      ScrLbl.push_back(make_pair(yDistPtr[j], pDoc->linkGnd[j]));
      nPos += pDoc->linkGnd[j];
      nNeg += (1 - pDoc->linkGnd[j]);
      nIx++;
    }
  }

  // sort the score & labels in descending order.
  std::stable_sort(ScrLbl.begin(), ScrLbl.end(), pair_cmp_fst_gt<double, int>);

  int nAccPos = 0;
  double *truePos = (double *) malloc(sizeof(double) * nIx);
  double *falsePos = (double *) malloc(sizeof(double) * nIx);
  memset(truePos, 0, sizeof(double) * nIx);
  memset(falsePos, 0, sizeof(double) * nIx);
  for (int i = 0; i < nIx; i++) {
    nAccPos += ScrLbl[i].second;
    truePos[i] = (double) nAccPos / (double) nPos;
    falsePos[i] = (double) (i + 1 - nAccPos) / (double) nNeg;
  }

  // compute the AUC score
  vector<int> flags;
  flags.push_back(0);
  for (int i = 1; i < nIx; i++) {
    if (ScrLbl[i].first < ScrLbl[i - 1].first)
      flags.push_back(i);
  }
  flags.push_back(nIx - 1);

  double dAuc = 0;
  for (unsigned int i = 1; i < flags.size(); i++) {
    int ix1 = flags[i - 1];
    int ix2 = flags[i];
    dAuc += (falsePos[ix2] - falsePos[ix1])
        * ((truePos[ix2] + truePos[ix1]) / 2);
  }
  delete[]truePos;
  delete[]falsePos;
  return dAuc;
}

/*
* save the prediction results and the predictive R^2 value
*/
double LinkSVMDiag::save_prediction(char *filename, Corpus *pC, double **phi) {
  double dAcc = get_test_acc(pC, phi);
  double dAUC = compute_auc(pC, m_dYDist);
  printf("Accuracy: %.4f; AUC: %.4f\n", dAcc, dAUC);

  FILE *fileptr;
  fileptr = fopen(filename, "w");
  fprintf(fileptr, "accuracy: %.5f; F1: %.5f; AUC: %.4f\n", dAcc, m_dF1Score, dAUC);
  for (int d = 0; d < pC->num_docs; d++) {
    Document *pDoc = &(pC->docs[d]);
    bool bEmpty = true;
    for (int i = 0; i < pDoc->num_neighbors; i++) {
      if (!pDoc->bTrain[i]) {
        fprintf(fileptr, "%d:%d\t", pDoc->linkGnd[i], pDoc->linkTest[i]);
        bEmpty = false;
      }
    }
    if (!bEmpty) fprintf(fileptr, "\n");
  }
  fclose(fileptr);

  return dAcc;
}


/****
* Jiaming: Accelerated SVM functions
*/

void LinkSVMDiag::learn_svm(Corpus *pC, double **phi, double *dMu, double eps,
    double Cp, double Cn) {// CPositive, CNegative
  boost::timer Timer;
  printf("learn_svm");
  int l = pC->num_train_links;
  int w_size = this->m_nSVMFeature;
  int s;
  double C, d, G;
  double *t_alpha = new double[m_nLabelNum * l];
  double *t_w = new double[m_nLabelNum * w_size];
  double *fvec = new double[l * w_size];
  for (int t_label = 0; t_label < m_nLabelNum; t_label++) {
    if (m_nLabelNum == 2 && t_label == 1) {
      for (int i = 0; i < w_size; i++) {
        t_w[i + w_size] = -t_w[i];
      }
      for (int i = 0; i < l; i++) {
        t_alpha[i + l] = -t_alpha[i];
      }
      break;
    }
    double *QD = new double[l];
    int iter = 0;
    int max_iter = 100;//1000; // modified 
    int *index = new int[l];
    int active_size = l;
    double *alpha = t_alpha + t_label * l;
    double *w = t_w + t_label * w_size;
    double PG;
    double PGmax_old = INF;
    double PGmin_old = -INF;
    double PGmax_new, PGmin_new;
    // solver type: L2R_L1LOSS_SVC_DUAL
    double diag[3] = {0, 0, 0};

    double upper_bound[3] = {Cn, 0, Cp};
    int *y = new int[l];
    int *from = new int[l], *to = new int[l];
    int id = 0;
    extract_train_links(pC, from, to, y, t_label);
    for (int i = 0; i < l; i++) {
      alpha[i] = 0;
    }
    for (int i = 0; i < w_size; i++) {
      w[i] = 0;
    }
    id = 0;
    for (int i = 0; i < l; i++) {
      QD[i] = diag[y[i] + 1];
      get_fvec(&(pC->docs[from[i]]),
          &(pC->docs[to[i]]),
          phi[from[i]],
          phi[to[i]],
          fvec + i * w_size);
      for (int j = 0; j < w_size; j++) {
        QD[i] += fvec[i * w_size + j] * fvec[i * w_size + j];
        w[j] += y[i] * alpha[i] * fvec[i * w_size + j];
      }
      index[i] = i;
    }
    printf("max_iter: %d\n", max_iter);
    while (iter < max_iter) {
      PGmax_new = -INF;
      PGmin_new = INF;
      for (int i = 0; i < active_size / 2; i++) {
        int j = i + rand() % (active_size - i);
        swap(index[i], index[j]);
      }
      for (s = 0; s < active_size; s++) {
        int i = index[rand() % active_size];
        G = 0;
        int yi = y[i];

        for (int j = 0; j < w_size; j++) {
          G += w[j] * fvec[i * w_size + j];
        }
        G = G * yi - 1;
        C = upper_bound[y[i] + 1];
        G += alpha[i] * diag[y[i] + 1];
        PG = 0;
        if (alpha[i] == 0) {
          if (G > PGmax_old) {
            active_size--;
            swap(index[s], index[active_size]);
            s--;
            continue;
          }
          else if (G < 0) {
            PG = G;
          }
        }
        else if (alpha[i] == C) {
          if (G < PGmin_old) {
            active_size--;
            swap(index[s], index[active_size]);
            s--;
            continue;
          }
          else if (G > 0) {
            PG = G;
          }
        }
        else {
          PG = G;
        }
        PGmax_new = std::max(PGmax_new, PG);
        PGmin_new = std::min(PGmin_new, PG);
        if (fabs(PG) > 1.0e-12) {
          double alpha_old = alpha[i];
          alpha[i] = std::min(std::max(alpha[i] - G / QD[i], 0.0), C);
          d = (alpha[i] - alpha_old) * yi;

          for (int j = 0; j < w_size; j++) {
            w[j] += fvec[i * w_size + j] * d;
          }
        }
      }
      iter++;
      if (iter % 100 == 0) {
        printf("#%d: %lf %lf\n", iter, PGmax_new, PGmin_new);
      }
      if (PGmax_new - PGmin_new <= eps) {
        if (active_size == l) {
          break;
        }
        else {
          active_size = l;
          printf("*");
          PGmax_old = INF;
          PGmin_old = -INF;
          continue;
        }
      }
      PGmax_old = PGmax_new;
      PGmin_old = PGmin_new;
      if (PGmax_old <= 0) {
        PGmax_old = INF;
      }
      if (PGmin_old >= 0) {
        PGmin_old = -INF;
      }
    }
    printf("optimization finished, #iter = %d\n", iter);
    if (iter >= max_iter) {
      printf("WARNING: reaching max number of iterations.\n");
    }
  }

  // WARNING: THE FOLLOWING MIGHT NOT BE CORRECT!
  // BUT the input for dMU is NULL! HELL.....YEAH!!!!!
  if (dMu != NULL) {
    int nVar = l * m_nLabelNum;

    for (int k = 0; k < l; k++) {
      dMu[k] = t_alpha[k];
    }
  }
  int rowIx, colIx;
  // WARNING: yi cannot be larger than 1 in this setting.
  m_dB = 0;
  for (int yi = 0; yi < m_nLabelNum; yi++) {
    int nRefIx = yi * m_nSVMFeature;
    double **wPtr = m_dW[yi];
    for (int i = 0; i < m_nLatentFeatures; i++) {
      int wIx = nRefIx + i;
      get_index(i, rowIx, colIx);
      wPtr[rowIx][colIx] = t_w[wIx];
      //wPtr[i][i] = t_w[wIx];
    }
    double *wOrgPtr = m_dOrgW[yi];
    for (int i = 0; i < m_nOrgFeatures; i++) {
      int wIx = nRefIx + m_nLatentFeatures + i;
      wOrgPtr[i] = t_w[wIx];
    }
  }
  delete fvec;
  double dTrainTime = Timer.elapsed();
  printf("training time: %lf\n", dTrainTime);
  running_time_for_svm += dTrainTime;
}

void LinkSVMDiag::learn_svm_mini_batch(Corpus *pC, double **phi, double *dMu,
    double eps, double Cp, double Cn, int em_iter) {// CPositive, CNegative
  boost::timer Timer;
  printf("svm-batch: %d\n", mini_batch_svm_size);
  //int l = pC->num_train_links;
  int l = mini_batch_svm_size;
  int w_size = this->m_nSVMFeature;
  int s;
  double C, d, G;
  double *t_alpha = new double[m_nLabelNum * l];
  double *t_w = new double[m_nLabelNum * w_size];
  double *fvec = new double[l * w_size];
  for (int t_label = 0; t_label < m_nLabelNum; t_label++) {
    if (m_nLabelNum == 2 && t_label == 1) {
      for (int i = 0; i < w_size; i++) {
        t_w[i + w_size] = -t_w[i];
      }
      for (int i = 0; i < l; i++) {
        t_alpha[i + l] = -t_alpha[i];
      }
      break;
    }
    double *QD = new double[l];
    int iter = 0;
    int max_iter = 1000;
    int *index = new int[l];
    int active_size = l;
    double *alpha = t_alpha + t_label * l;
    double *w = t_w + t_label * w_size;
    double PG;
    double PGmax_old = INF;
    double PGmin_old = -INF;
    double PGmax_new, PGmin_new;
    // solver type: L2R_L1LOSS_SVC_DUAL
    double diag[3] = {0, 0, 0};

    double upper_bound[3] = {Cn, 0, Cp};
    int *y = new int[l];

    int id = 0;

    for (int i = 0; i < l; i++) {
      alpha[i] = 0;
    }
    for (int i = 0; i < w_size; i++) {
      w[i] = 0;
    }
    id = 0;
    //int *from = new int[l], *to = new int[l];
    //extract_train_links(pC, from, to, y, t_label);
    int *from = mini_batch_svm_from, *to = mini_batch_svm_to;
    for (int i = 0; i < l; i++) {
      if (mini_batch_svm_label[i] == t_label) y[i] = 1;
      else y[i] = -1;
    }
    for (int i = 0; i < l; i++) {
      QD[i] = diag[y[i] + 1];
      get_fvec(&(pC->docs[from[i]]),
          &(pC->docs[to[i]]),
          phi[from[i]],
          phi[to[i]],
          fvec + i * w_size);
      for (int j = 0; j < w_size; j++) {
        QD[i] += fvec[i * w_size + j] * fvec[i * w_size + j];
        w[j] += y[i] * alpha[i] * fvec[i * w_size + j];
      }
      index[i] = i;
    }
    printf("max_iter: %d\n", max_iter);
    while (iter < max_iter) {
      PGmax_new = -INF;
      PGmin_new = INF;
      for (int i = 0; i < active_size / 2; i++) {
        int j = i + rand() % (active_size - i);
        swap(index[i], index[j]);
      }
      for (s = 0; s < active_size; s++) {
        int i = index[rand() % active_size];
        G = 0;
        int yi = y[i];

        for (int j = 0; j < w_size; j++) {
          G += w[j] * fvec[i * w_size + j];
        }
        G = G * yi - 1;
        C = upper_bound[y[i] + 1];
        G += alpha[i] * diag[y[i] + 1];
        PG = 0;
        if (alpha[i] == 0) {
          if (G > PGmax_old) {
            active_size--;
            swap(index[s], index[active_size]);
            s--;
            continue;
          }
          else if (G < 0) {
            PG = G;
          }
        }
        else if (alpha[i] == C) {
          if (G < PGmin_old) {
            active_size--;
            swap(index[s], index[active_size]);
            s--;
            continue;
          }
          else if (G > 0) {
            PG = G;
          }
        }
        else {
          PG = G;
        }
        PGmax_new = std::max(PGmax_new, PG);
        PGmin_new = std::min(PGmin_new, PG);
        if (fabs(PG) > 1.0e-12) {
          double alpha_old = alpha[i];
          alpha[i] = std::min(std::max(alpha[i] - G / QD[i], 0.0), C);
          d = (alpha[i] - alpha_old) * yi;

          for (int j = 0; j < w_size; j++) {
            w[j] += fvec[i * w_size + j] * d;
          }
        }
      }
      iter++;
      if (iter % 100 == 0) {
        printf("#%d: %lf %lf\n", iter, PGmax_new, PGmin_new);
      }
      if (PGmax_new - PGmin_new <= eps) {
        if (active_size == l) {
          break;
        }
        else {
          active_size = l;
          printf("*");
          PGmax_old = INF;
          PGmin_old = -INF;
          continue;
        }
      }
      PGmax_old = PGmax_new;
      PGmin_old = PGmin_new;
      if (PGmax_old <= 0) {
        PGmax_old = INF;
      }
      if (PGmin_old >= 0) {
        PGmin_old = -INF;
      }
    }
    printf("optimization finished, #iter = %d\n", iter);
    if (iter >= max_iter) {
      printf("WARNING: reaching max number of iterations.\n");
    }
  }

  // WARNING: THE FOLLOWING MIGHT NOT BE CORRECT!
  // BUT the input for dMU is NULL! HELL.....YEAH!!!!!
  if (dMu != NULL) {
    int nVar = l * m_nLabelNum;

    for (int k = 0; k < l; k++) {
      dMu[k] = t_alpha[k];
    }
  }
  int rowIx, colIx;
  // WARNING: yi cannot be larger than 1 in this setting.
  m_dB = 0;
  double rou = 1.0 / (em_iter + 1);
  printf("rou = %lf\n", rou);
  for (int yi = 0; yi < m_nLabelNum; yi++) {
    int nRefIx = yi * m_nSVMFeature;
    double **wPtr = m_dW[yi];
    for (int i = 0; i < m_nLatentFeatures; i++) {
      int wIx = nRefIx + i;
      get_index(i, rowIx, colIx);
      wPtr[rowIx][colIx] = (1 - rou) * wPtr[rowIx][colIx] + rou *
          t_w[wIx];
    }
    double *wOrgPtr = m_dOrgW[yi];
    for (int i = 0; i < m_nOrgFeatures; i++) {
      int wIx = nRefIx + m_nLatentFeatures + i;
      wOrgPtr[i] = (1 - rou) * wOrgPtr[i] + rou * t_w[wIx];
    }
  }
  delete fvec;
  double dTrainTime = Timer.elapsed();
  printf("training time: %lf\n", dTrainTime);
  running_time_for_svm += dTrainTime;
}

void LinkSVMDiag::learn_svm_pegasos(Corpus *pC, double **phi, int svm_iter) {
  boost::timer Timer;

  int l = pC->num_train_links;
  int *from = new int[l], *to = new int[l];
  int w_size = this->m_nSVMFeature;
  double lambda = 1.0 / (l * m_dC);
  int max_iter = (int)(100.0 / lambda * (0.7 + 0.3 * exp(-svm_iter)));
  int exam_per_iter = 1;
  int num_iter_to_avg = 100;
  double *t_w = new double[m_nLabelNum * w_size];
  double *fvec = new double[l];
  int *label = new int[l];

  int rowIx, colIx;
  //printf("lambda = %lf\n", lambda);

  for (int yi = 0; yi < m_nLabelNum; yi++) {
    int nRefIx = yi * m_nSVMFeature;
    double **wPtr = m_dW[yi];
    for (int i = 0; i < m_nLatentFeatures; i++) {
      int wIx = nRefIx + i;
      get_index(i, rowIx, colIx);
      t_w[wIx] = wPtr[rowIx][colIx];
    }
    double *wOrgPtr = m_dOrgW[yi];
    for (int i = 0; i < m_nOrgFeatures; i++) {
      int wIx = nRefIx + m_nLatentFeatures + i;
      t_w[wIx] = wOrgPtr[i];
    }
  }
  for (int t_label = 0; t_label < m_nLabelNum; t_label++) {
    if (m_nLabelNum == 2 && t_label == 1) {
      for (int i = 0; i < w_size; i++) {
        t_w[i + w_size] = -t_w[i];
      }
      break;
    }
    double *avg_w = t_w + t_label * w_size;
    double *w = new double[w_size];


    extract_train_links(pC, from, to, label, t_label);

    for (int i = 0; i < w_size; i++) {
      w[i] = avg_w[i];
      avg_w[i] = 0;
    }

    int avg_scale = (num_iter_to_avg > max_iter)? max_iter :
        num_iter_to_avg;
    for (int i = 0; i < max_iter; i++) {
      std::vector<int> grad_index;
      std::vector<double> grad_weights;
      double eta = 1 / (lambda * (i + 2)); // learning rate?
      int r = rand() % l;
      double prediction = 0;
      get_fvec(&(pC->docs[from[r]]),
          &(pC->docs[to[r]]),
          phi[from[r]],
          phi[to[r]],
          fvec);

      for (int k = 0; k < w_size; k++) {
        prediction += w[k] * fvec[k];
      }
      double curloss = 1 - label[r] * prediction;

      if (curloss > 0.0) {
        grad_index.push_back(r);
        grad_weights.push_back((eta * label[r]));
      }

      for (int k = 0; k < w_size; k++) {
        w[k] *= (1.0 - eta * lambda);
      }
      for (int j = 0; j < grad_index.size(); j++) {

        for (int k = 0; k < w_size; k++) {
          w[k] += fvec[k] * grad_weights[j];
        }
      }
      double w_norm = 0;
      for (int k = 0; k < w_size; k++) {
        w_norm +=  w[k] * w[k];
      }
      if (w_norm > 1.0 / lambda) {
        for (int k = 0; k < w_size; k++) {
          w[k] *= sqrt(1.0 / (lambda * w_norm));
        }
      }
      w_norm = 0;
      for (int k = 0; k < w_size; k++) {
        w_norm +=  w[k] * w[k];
      }
      if (i + avg_scale >= max_iter) {
        for (int k = 0; k < w_size; k++) {
          avg_w[k] += w[k] / (double) avg_scale;
        }
      }
    }
    double loss = 0.0;
    for (int j = 0; j < l; j++) {
      double pred = 0.0;
      get_fvec(&(pC->docs[from[j]]),
          &(pC->docs[to[j]]),
          phi[from[j]],
          phi[to[j]],
          fvec);
      for (int k = 0; k < w_size; k++) {
        pred += avg_w[k] * fvec[k];
      }
      if (1 - label[j] * pred > 0.0) loss += (1 - label[j] * pred);
    }
    printf("loss: %lf\n", loss);
  }

  // WARNING: yi cannot be larger than 1 in this setting.
  m_dB = 0;
  for (int yi = 0; yi < m_nLabelNum; yi++) {
    int nRefIx = yi * m_nSVMFeature;
    double **wPtr = m_dW[yi];
    for (int i = 0; i < m_nLatentFeatures; i++) {
      int wIx = nRefIx + i;
      get_index(i, rowIx, colIx);
      wPtr[rowIx][colIx] = t_w[wIx];
    }
    double *wOrgPtr = m_dOrgW[yi];
    for (int i = 0; i < m_nOrgFeatures; i++) {
      int wIx = nRefIx + m_nLatentFeatures + i;
      wOrgPtr[i] = t_w[wIx];
    }
  }
  delete []from;
  delete []to;
  delete []t_w;
  delete []fvec;
  double dTrainTime = Timer.elapsed();
  printf("training time: %lf\n", dTrainTime);
  running_time_for_svm += dTrainTime;
}

void LinkSVMDiag::extract_train_links(Corpus *pC, int *from, int *to, int *label, int l) {
  int id = 0;
  for (int d = 0; d < pC->num_docs; d++) {
    Document *pDoc = &(pC->docs[d]);
    for (int k = 0; k < pDoc->num_neighbors; k++) {
      if (!pDoc->bTrain[k]) continue;
      from[id] = d;
      to[id] = pDoc->neighbors[k];
      if (pDoc->linkGnd[k] == l) {
        label[id] = 1;
      }
      else {
        label[id] = -1;
      }
      id++;
    }
  }
}

