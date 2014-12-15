#include <cstdio>
#include <cstring>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include "LinkSVM/LinkSVM.h"
#include "LinkSVM/LinkSVMSym.h"
#include "utils/Corpus.h"
#include "utils/Params.h"

namespace fs = boost::filesystem;
namespace po = boost::program_options;

int main(int argc, char **argv) {
  // Read Settings
  seedMT(time(NULL));
  po::options_description desc("Options");
  desc.add_options()
      ("help,h", "print help messages")
      ("settings,s", po::value<std::string>(), "settings file")
      ("truncate,t", po::value<int>(), "truncation level")
      ("alpha,a", po::value<float>(), "intitial alpha")
      ("c1", po::value<float>(), "initial c1")
      ("c2", po::value<float>(), "initial c2")
      ("ell", po::value<float>(), "delta ell")
      ("nu", po::value<int>(), "stochastic number for nu")
      ("phi", po::value<int>(), "stochastic level for phi")
      ("var_iter", po::value<int>(), "number of variational iterations")
      ("iter", po::value<int>(), "total number of iterations")
      ("kappa_nu", po::value<float>(), "forgetting rate for nu")
      ("kappa_phi", po::value<float>(), "forgetting rate for phi")
      ("phi_iter", po::value<int>(), "number of iterations in one stochastic update_phi()");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  if (argc > 1) {
    Corpus* c = NULL;
    Params param;
    fs::path Path = fs::current_path().parent_path();
    Path += fs::path("/Settings/");
    char settings_file[512];
    strcpy(settings_file, Path.string().c_str());
    if (vm.count("settings")) {
      strcat(settings_file, vm["settings"].as<std::string>().c_str());
    } else {
      std::cout << desc << std::endl;
      return 0;
    }
    //printf("read settings from %s\n", settings_file);
    param.read_settings(settings_file);
    printf("label: %s\n", param.label);
    if (vm.count("truncate")) {
      param.T = vm["truncate"].as<int>();
    }
    if (vm.count("nu")) {
      param.STOCHASTIC_NU = vm["nu"].as<int>();
    }
    if (vm.count("phi")) {
      param.STOCHASTIC_PHI = vm["phi"].as<int>();
    }
    if (vm.count("alpha")) {
     param.INITIAL_ALPHA = vm["alpha"].as<float>();
    }
    if (vm.count("c1")) {
     param.INITIAL_C1 = vm["c1"].as<float>();
    }
    if (vm.count("c2")) {
     param.INITIAL_C2 = vm["c2"].as<float>();
    }
    if (vm.count("ell")) {
     param.DELTA_ELL = vm["ell"].as<float>();
    }
    if (vm.count("var_iter")) {
      param.VAR_MAX_ITER = vm["var_iter"].as<int>();
    }
    if (vm.count("iter")) {
      param.EM_MAX_ITER = vm["iter"].as<int>();
    }
    if (vm.count("kappa_nu")) {
      param.FORGETRATE_NU = vm["kappa_nu"].as<float>();
    }
    if (vm.count("kappa_phi")) {
      param.FORGETRATE_PHI = vm["kappa_phi"].as<float>();
    }
    if (vm.count("phi_iter")) {
      param.PHI_ITER = vm["phi_iter"].as<int>();
    }

    char fileName[512];
    char cur[512];
    char dir[512], directory[512];
    fs::path curPath;
    curPath = fs::current_path().parent_path();
    strcpy(cur, curPath.string().c_str());
    // Start Inference
    /*
     * Inference for Kinship Dataset.
     */
    if (strcmp(param.label, "kinship") == 0) {
       double *dTstAUC = new double[param.NUM_RELATION];
       double dMeanAUC = 0;
#ifdef MULTI_KINSHIP_RELATION
       for (int i = 0; i < param.NUM_RELATION; i++) {
#else
         int i = 11;
#endif

         param.NFOLDS = i + 1;
         c = new Corpus();



         sprintf(fileName, "%s_R%d.dat", param.file_root, i + 1);
         strcat(cur, fileName);
         c->read(cur, param.NDOCS);
         sprintf(fileName, "%s_R%d_trSample.dat", param.file_root, i + 1);
         strcpy(cur, curPath.string().c_str());
         strcat(cur, fileName);

         c->set_train_tag(80, cur);

         sprintf(dir, "/K%d_nu%d_phi%d_c1_%.2f_c2_%.2f", param.T, param.STOCHASTIC_NU,
                 param.STOCHASTIC_PHI, param.INITIAL_C1, param.INITIAL_C2);
         curPath = fs::current_path().parent_path();
         curPath += fs::path("/Results/Kinship");
         strcpy(directory, curPath.string().c_str());
         strcat(directory, dir);
         fs::create_directory(fs::path(directory));

         LinkSVM model(&param);
         dTstAUC[i] = model.train(directory, c);
         dMeanAUC += dTstAUC[i] / param.NUM_RELATION;
         delete c;
#ifdef MULTI_KINSHIP_RELATION
       }

       double dStd = 0;
       for (int i = 0; i < param.NUM_RELATION; i++) {
         dStd += (dTstAUC[i] - dMeanAUC) * (dTstAUC[i] - dMeanAUC) / param.NUM_RELATION;
       }
       dStd = sqrt(dStd);
#else
       double dStd = 0;
#endif
       printf("dMeanAUC = %.5f\tdStd = %.5f\n", dMeanAUC, dStd);
    } else if (strcmp(param.label, "symnips") == 0) {
      c = new Corpus();

      sprintf(fileName, "%s_R1.dat", param.file_root);
      strcat(cur, fileName);
      c->read(cur, param.NDOCS);

      sprintf(fileName, "%s_R1_trSample_nonSym.dat", param.file_root);
      strcpy(cur, curPath.string().c_str());
      strcat(cur, fileName);
      c->set_train_tag(80, cur);
      sprintf(dir, "/K%d_nu%d_phi%d_knu%.1f_kphi%.1f_c1%.1f_c2%.1f_%s", param.T, param.STOCHASTIC_NU,
                       param.STOCHASTIC_PHI, param.FORGETRATE_NU, param.FORGETRATE_PHI, param.INITIAL_C1, param.INITIAL_C2, param.model);
      curPath = fs::current_path().parent_path();
      curPath += fs::path("/Results/Nips_Coauthorship");
      strcpy(directory, curPath.string().c_str());
      strcat(directory, dir);
      fs::create_directory(fs::path(directory));
      if (strcmp(param.model, "LinkSVMSym") == 0) {
        LinkSVMSym model(&param);
        model.train(directory, c);
      } 
    } else if (strcmp(param.label, "wiki") == 0) {
      c = new Corpus();
      sprintf(fileName, "%s.dat", param.file_root);
      strcat(cur, fileName);
      c->read2(cur, param.NDOCS);
      sprintf(fileName, "%s_trSample.dat", param.file_root);
      strcpy(cur, curPath.string().c_str());
      strcat(cur, fileName);
      c->set_train_tag_rand(80, cur);
      sprintf(dir, "/K%d_nu%d_phi%d_a%.1f_ell%.1f_c1%.1f_c2%.1f_%s", param.T, param.STOCHASTIC_NU,
                       param.STOCHASTIC_PHI, param.INITIAL_ALPHA, param.DELTA_ELL, param.INITIAL_C1, param.INITIAL_C2, param.model);
      curPath = fs::current_path().parent_path();
      curPath += fs::path("/Results/Wiki");
      strcpy(directory, curPath.string().c_str());
      strcat(directory, dir);
      fs::create_directory(fs::path(directory));
      if (strcmp(param.model, "LinkSVM") == 0) {
        LinkSVM model(&param);
        model.train(directory, c);
      }
    }

  } else {
    std::cout << desc << std::endl;
  }
  return 0;
}
