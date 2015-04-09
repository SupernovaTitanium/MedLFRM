#pragma once

#ifndef LINKSVMDIAG
#define LINKSVMDIAG

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#include "../utils/utils.h"
#include "../utils/cokus.h"
#include "../utils/Params.h"
#include "../utils/Corpus.h"

#include "../SVM_Multiclass/svm_struct_api.h"
#include "../SVM_Multiclass/svm_struct_learn.h"
#include "../SVM_Multiclass/svm_struct_common.h"


class LinkSVMDiag {
 public:
  LinkSVMDiag();
  
  LinkSVMDiag(Params *pParam);
  
  ~LinkSVMDiag();
  
  double train(char *dir, Corpus *c);
  
  void random_init(Corpus *pC, double **phi);
  
  void random_wvar_nu();
  
  void random_w();
  
  void random_batch(int *target, const int batch_size, const int start, const int end);
  
  void learn_svm(Corpus *pC, double **phi, double *res);
  
  void set_init_param(STRUCT_LEARN_PARM *struct_parm, LEARN_PARM *learn_parm,
      KERNEL_PARM *kernel_parm, int *alg_type);
  
  void sample_init(Corpus *pC, double **phi, STRUCT_LEARN_PARM *struct_parm);
  
  void sample_init2(Corpus *pC, double **phi, STRUCT_LEARN_PARM *struct_parm);
  
  void outputLowDimData(char *filename, Corpus *pC, double **phi);
  
  void get_fvec(Document *pDoc1, Document *pDoc2, double *phi1, double *phi2, double *fvec);
  
  void compute_nu_stat(double **var_nu);
  
  double compute_nu_exp_prod(const int &k);
  
  double e_step(Corpus *pC, double **var_nu, double **phi);
  
  double e_step(Corpus *pC, double **var_nu, double **phi, const int batch_size);
  
  double compute_lhood(Corpus *pC, double **phi, double **var_nu);
  
  double loss_aug_predict(Document *doc, Document *doc2, double *phi, double *phi2, const int &j);
  
  double compute_disc_fscore(const int &y, Document *doc, Document *doc2, double *phi, double *phi2);
  
  double compute_mrgterm_right(Document *doc, double *phi, const int &j, const int &k);
  
  double compute_mrgterm_left(Document *doc, double *phi, const int &j, const int &k);
  
  double compute_mrgterm_self(Document *doc, double *phi, const int &j, const int &k);
  
  void update_phi(Corpus *pC, Document *pDoc, double **phi, const int &docId,
      double **var_nu);
  
  void update_phi(Corpus *pC, Document *pDoc, double **phi, const int &docId,
      double **var_nu, const int batch_size, int iter);
  
  void update_nu(Corpus *pC, double **var_nu, double **phi);
  
  void update_nu(Corpus *pC, double **var_nu, double **phi, int *sto, const int batch_size, int iter);
  
  void update_ydist(Corpus *pC, double **phi);
  
  void update_ydist_tr(Corpus *pC, double **phi);
  
  double get_2norm_w();
  
  void get_index(const int &ix, int &rowIx, int &colIx);
  
  void save_model(char *model_root, Corpus *pC, double **phi);
  
  void load_model(char *model_root);
  
  void new_model(int num_docs, int num_train_links);
  
  void free_model();
  
  double compute_auc(Corpus *pC, double **yDist);
  
  double compute_auc_tr(Corpus *pC, double **yDist);
  
  double get_test_acc(Corpus *pC, double **phi);
  
  double save_prediction(char *filename, Corpus *pC, double **phi);
  
  double loss(const int &label, const int &gnd) {
    if (label == gnd) return 0;
    else return m_dDeltaEll;
  }
  
  void predict(Document *doc, const int &i, double *yDist);
  
  /****
  * Jiaming: here are function definitions for Pegasos SVM
  **/
  // the entrance for learn_svm; here we overload the original function
  void learn_svm(Corpus *pC, double **phi, double *dMu, double eps,
      double Cp, double Cn);
  
  void learn_svm_pegasos(Corpus *pC, double **phi, int svm_iter);
  
  void learn_svm_mini_batch(Corpus *pC, double **phi, double *dMu,
      double eps, double Cp, double Cn, int em_iter);
  
  void extract_train_links(Corpus *pC, int *from, int *to, int *label, int l);
  
  //

 public:
  // hyper-parameters
  double m_dC;
  double m_alpha;
  double m_dWPriorVar;
  
  double ***m_dW;
  double **m_dOrgW;
  double *m_dMu;
  double m_dB;
  double m_dsvm_primalobj;
  int m_nSVMFeature;
  int m_nOrgFeatures;
  int m_nLatentFeatures;
  int m_nK;
  int m_nLabelNum;
  Params *m_pParam;
  SAMPLE m_sample;
  double **m_dYDist;
  
  double **m_var_nu;
  int m_nData;
  int m_nTrainLinks;
  double m_dDeltaEll;
  
  char m_dir[512];
  // expectation statistics of nu.
  double *m_dNuExpSum;
  double *m_dNuExpProd;
  
  double *m_digamma_v1_;
  double *m_digamma_v2_;
  double *m_digamma_vsum_;
  double **m_dQ_dist_;
  double *m_exp_feat_num_;
  double *m_exp_feat_n_minus_num_;
  double *m_phi_old_;
  
  double *dAVec;
  double *dBVec;
  double *dCVec;
  double m_dF1Score;
  
  double *truePos_;
  double *falsePos_;
  double *m_dFeature_;
  double *m_sum_phi_;
  int m_nTstLinks;
  double *fvec_;
  
  int m_stochastic_nu;
  int m_stochastic_phi;
  double _delay_nu;
  double _forgetting_rate_nu;
  double _delay_phi;
  double _forgetting_rate_phi;
//	char m_sample;
  
  double running_time_for_update_phi;
  double running_time_for_update_nu;
  double running_time_for_svm;
  double running_time_for_vi;
  int phi_iter;
  
  int *mini_batch_svm_to;
  int *mini_batch_svm_from;
  int *mini_batch_svm_label;
  int mini_batch_svm_size;
};

#endif
