#include <bits/stdc++.h>
#include <pcl/io/pcd_io.h>
#include "common/common.h"
#include "pclmatch/wrapper.hpp"

using namespace std;

void ShowXYT(const Eigen::Matrix3d &mtx, string str = "") {
  printf("%s: (%f, %f)(%f), %f\n", str.c_str(), mtx(0, 2), mtx(1, 2),
         mtx.block<2, 1>(0, 2).norm(), acos(mtx(0, 0)));
}

vector<double> Error(const Eigen::Matrix3d &t1, const Eigen::Matrix3d &t2) {
  double tl = (t1.block<2, 1>(0, 2) - t2.block<2, 1>(0, 2)).norm();
  double rot = abs(acos(t1(0, 0)) - acos(t2(0, 0))) * 180. / M_PI;
  return {tl, rot};
}

vector<vector<double>> ConstructConfigs(const vector<vector<double>> &input) {
  Expects(input.size() == 5);
  vector<vector<double>> ret;
  for (const auto &c1 : input.at(0)) {
    for (const auto &c2 : input.at(1)) {
      for (const auto &c3 : input.at(2)) {
        for (const auto &c4 : input.at(3)) {
          for (const auto &c5 : input.at(4)) {
            ret.push_back({c1, c2, c3, c4, c5});
          }
        }
      }
    }
  }
  return ret;
}

vector<int> StateNumbers(vector<ConvergeState> states) {
  int nc, it, tr, ab, re, no, fa;
  nc = it = tr = ab = re = no = fa = 0;
  for (const auto &state : states) {
    if (state == ConvergeState::NOT_CONVERGED) {
      ++nc;
    } else if (state == ConvergeState::ITERATIONS) {
      ++it;
    } else if (state == ConvergeState::TRANSFORM) {
      ++tr;
    } else if (state == ConvergeState::ABS_MSE) {
      ++ab;
    } else if (state == ConvergeState::REL_MSE) {
      ++re;
    } else if (state == ConvergeState::NO_CORRESPONDENCES) {
      ++no;
    } else if (state == ConvergeState::FAILURE_AFTER_MAX_ITERATIONS) {
      ++fa;
    }
  }
  return {nc, it, tr, ab, re, no, fa};
}

int main() {
  MatchPackage mp;
  pcl::io::loadPCDFile(APATH(20210128/cases/spc00.pcd), *mp.source);
  pcl::io::loadPCDFile(APATH(20210128/cases/tpc00.pcd), *mp.target);
  mp.actual = common::Matrix3dFromXYTRadian(common::ReadFromFile(APATH(20210128/cases/tgt00.txt)));
  auto config = common::ReadFromFile(APATH(20210128/cases/config.txt));

  // ICP config: max_iter, max_corr, reci, rms_err, tf_err
  vector<double> iters = {100};
  vector<double> corrs = {5, 10, 15};
  vector<double> recis = {0};
  // vector<double> rmses = {0.005, 0.001, 0.0005, 0.0001, 0.00005};
  // vector<double> tfses = {0.005, 0.001, 0.0005, 0.0001, 0.00005};
  vector<double> rmses = {0.0005, 0.0001, 0.00005};
  vector<double> tfses = {0.0005, 0.0001, 0.00005};

  auto configs = ConstructConfigs({iters, corrs, recis, rmses, tfses});
  dprintf("config size: %ld\n", configs.size());

  // ICP initial guess: tranlation, rotation
  vector<common::RandomTransformGenerator2D> rtgs;
  common::RandomTransformGenerator2D rtg;
  rtg.SetCenterXYRadian(mp.actual(0, 2), mp.actual(1, 2), acos(mp.actual(0, 0)));

  rtg.SetTranslationRadiusBound(0, 2);
  rtg.SetRotationDegreeBound(0, 15);
  rtgs.push_back(rtg);

  rtg.SetTranslationRadiusBound(0, 2);
  rtg.SetRotationDegreeBound(15, 30);
  rtgs.push_back(rtg);

  rtg.SetTranslationRadiusBound(0, 2);
  rtg.SetRotationDegreeBound(30, 45);
  rtgs.push_back(rtg);

  rtg.SetTranslationRadiusBound(0, 2);
  rtg.SetRotationDegreeBound(45, 60);
  rtgs.push_back(rtg);

  rtg.SetTranslationRadiusBound(5, 7);
  rtg.SetRotationDegreeBound(0, 15);
  rtgs.push_back(rtg);

  rtg.SetTranslationRadiusBound(5, 7);
  rtg.SetRotationDegreeBound(15, 30);
  rtgs.push_back(rtg);

  // Do ICP
  PrintVersion("ICP");
  for (auto &rtg : rtgs) {
    dprintf("(%.2f, %.2f) x (%.2f, %.2f)\n",
            rtg.GetTranslationRadiusBound().at(0),
            rtg.GetTranslationRadiusBound().at(1),
            rtg.GetRotationDegreeBound().at(0),
            rtg.GetRotationDegreeBound().at(1));
    size_t samples = 100;
    auto mtxs = rtg.Generate(samples);
    for (const auto &config : configs) {
      vector<double> tlerrs, roterrs;
      vector<ConvergeState> states;
      dprintf("  cfg: (%d, %f, %d, %f, %f) ", static_cast<int>(config.at(0)),
              config.at(1), static_cast<int>(config.at(2)), config.at(3),
              config.at(4));
      for (const auto &mtx : mtxs) {
        mp.guess = mtx;
        DoSICP(mp, config);
        auto err = Error(mp.result, mp.actual);
        tlerrs.push_back(err.at(0));
        roterrs.push_back(err.at(1));
        states.push_back(mp.state);
      }
      sort(tlerrs.begin(), tlerrs.end());
      sort(roterrs.begin(), roterrs.end());
      dprintf("err median: %.2f, %.2f, ", tlerrs.at(samples / 2), roterrs.at(samples / 2));
      auto statenum = StateNumbers(states);
      // NOT_CON, ITER, TF, ABS_MSE, REL_MSE, NO_CORR, FAIL_AFTER_MAXITER
      for (int i = 0; i < 7; ++i) {
        dprintf("%02d%s", statenum.at(i), (i == 6) ? "" : "/");
      }
      dprintf("\n");
    }
  }

  // mp.guess = rtg.Generate(1).at(0);
  // ShowXYT(mp.guess, "guess");
  // ShowXYT(mp.actual, "actual");
  // DoICP(mp, {100, 10, 0, 0.0001, 0.0001});
  // DoSICP(mp, {100, 10, 0, 0.0001, 0.0001});
  // ShowXYT(mp.result, "result");
  // common::WriteToFile(APATH(20210128/cases/res00.txt), {mp.result(0, 2), mp.result(1, 2), acos(mp.result(0, 0))});
}
