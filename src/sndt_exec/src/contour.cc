/**
 * @file contour.cc
 * @brief Output to /tmp/contour.txt, and Input of contour.py
 * @version 0.1
 * @date 2021-10-25
 * @copyright Copyright (c) 2021
 */
#include <bits/stdc++.h>
#include <common/common.h>
#include <pcl/search/kdtree.h>
#include <sndt/cost_functors.h>
#include <sndt_exec/wrapper.h>

#include <Eigen/Dense>
#include <boost/program_options.hpp>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

template <typename T>
vector<pair<int, int>> Corres(const T &target_map,
                              const T &source_map,
                              const Affine2d &guess_tf = Affine2d::Identity()) {
  vector<pair<int, int>> ret;
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());
  auto t1 = GetTime();
  auto next_map = source_map.PseudoTransformCells(guess_tf, true);
  for (size_t i = 0; i < next_map.size(); ++i) {
    auto cellp = next_map[i];
    if (!cellp->HasGaussian()) continue;
    auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
    if (idx == -1) continue;
    auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
        kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
    if (!cellq || !cellq->HasGaussian()) continue;
    ret.push_back({i, target_map.GetCellIndex(cellq)});
    // ret.push_back({i, i});
  }
  return ret;
}

void SF(string str, Affine2d T) {
  cout << str << ((T.translation().isZero(1)) ? ": success" : ": fail") << endl;
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  int n, m;
  double cell_size, voxel, r, radius = 1.5;
  string data;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("n,n", po::value<int>(&n)->default_value(0), "n")
      ("m,m", po::value<int>(&m)->default_value(0), "m")
      ("r,r", po::value<double>(&r)->default_value(15), "r");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  po::notify(vm);

  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath(data), "lidar.ser"), vpc);

  auto tgt = PCMsgTo2D(vpc[n], voxel);
  TransformPointsInPlace(tgt, aff2);

  VectorXd xs = VectorXd::LinSpaced(16, 0, 15);
  VectorXd ys = VectorXd::LinSpaced(16, 0, 15);
  constexpr int sz = 81;
  VectorXd x = VectorXd::LinSpaced(sz, -40, 40);
  VectorXd y = VectorXd::LinSpaced(sz, -40, 40);
  MatrixXd zz(sz, sz);

  for (int i = 15; i < xs.size(); ++i) {
    cout << "aff: " << xs[i] << ", " << ys[i] << endl;
    // auto aff = Translation2d(xs[i], ys[i]) * Rotation2Dd(0);
    auto aff = Translation2d(14.4966, -3.85344) * Rotation2Dd(0);
    auto src = TransformPoints(tgt, aff);
    vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, Affine2d::Identity()}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{{src, Affine2d::Identity()}};

    D2DNDTParameters params7;
    params7.cell_size = cell_size;
    params7.r_variance = params7.t_variance = 0;
    auto tgt7 = MakeNDTMap(datat, params7);
    auto src7 = MakeNDTMap(datas, params7);
    auto corres7 = Corres(tgt7, src7);
    auto T7 = SNDTMatch2(tgt7, src7, params7);
    SF("m7", aff * T7);

    SNDTParameters params6;
    params6.cell_size = cell_size;
    params6.radius = radius;
    params6.r_variance = params6.t_variance = 0;
    auto tgt6 = MakeSNDTMap(datat, params6);
    auto src6 = MakeSNDTMap(datas, params6);
    auto corres6 = Corres(tgt6, src6);
    auto T6 = SNDTMatch(tgt6, src6, params6);
    SF("m6", aff * T6);

    double ini = 0, fin = 0;
    for (auto corr : corres6) {
      auto up = src6[corr.first]->GetPointMean();
      auto cp = src6[corr.first]->GetPointCov();
      auto unp = src6[corr.first]->GetNormalMean();
      auto cnp = src6[corr.first]->GetNormalCov();
      auto uq = tgt6[corr.second]->GetPointMean();
      auto cq = tgt6[corr.second]->GetPointCov();
      auto unq = tgt6[corr.second]->GetNormalMean();
      auto cnq = tgt6[corr.second]->GetNormalCov();
      ini += SNDTCostFunctor2::Cost2(up, cp, unp, cnp, uq, cq, unq, cnq);
      fin += SNDTCostFunctor2::Cost2(up, cp, unp, cnp, uq, cq, unq, cnq, aff.inverse());
    }
    cout << "m6: " << ini << " -> " << fin << endl;
    ini = fin = 0;
    for (auto corr : corres7) {
      auto up = src7[corr.first]->GetPointMean();
      auto cp = src7[corr.first]->GetPointCov();
      Vector2d unp = src7[corr.first]->GetPointEvecs().col(0);
      auto uq = tgt7[corr.second]->GetPointMean();
      auto cq = tgt7[corr.second]->GetPointCov();
      Vector2d unq = tgt7[corr.second]->GetPointEvecs().col(0);
      ini += SNDTCostFunctor3::Cost2(up, cp, unp, uq, cq, unq);
      fin += SNDTCostFunctor3::Cost2(up, cp, unp, uq, cq, unq, aff.inverse());
    }
    cout << "m7: " << ini << " -> " << fin << endl;
    ini = fin = 0;
    for (auto corr : corres6) {
      auto up = src6[corr.first]->GetPointMean();
      auto cp = src6[corr.first]->GetPointCov();
      Vector2d unp = src6[corr.first]->GetPointEvecs().col(0);
      auto uq = tgt6[corr.second]->GetPointMean();
      auto cq = tgt6[corr.second]->GetPointCov();
      Vector2d unq = tgt6[corr.second]->GetPointEvecs().col(0);
      ini += SNDTCostFunctor3::Cost2(up, cp, unp, uq, cq, unq);
      fin += SNDTCostFunctor3::Cost2(up, cp, unp, uq, cq, unq, aff.inverse());
    }
    cout << "m8: " << ini << " -> " << fin << endl;

    for (int j = 0; j < sz; ++j) {
      for (int k = 0; k < sz; ++k) {
        auto ges = Translation2d(x(k), y(j)) * Rotation2Dd(0);
        zz(j, k) = 0;
        if (m == 6) {
          for (auto corr : corres6) {
            auto up = src6[corr.first]->GetPointMean();
            auto cp = src6[corr.first]->GetPointCov();
            auto unp = src6[corr.first]->GetNormalMean();
            auto cnp = src6[corr.first]->GetNormalCov();
            auto uq = tgt6[corr.second]->GetPointMean();
            auto cq = tgt6[corr.second]->GetPointCov();
            auto unq = tgt6[corr.second]->GetNormalMean();
            auto cnq = tgt6[corr.second]->GetNormalCov();
            zz(j, k) += SNDTCostFunctor2::Cost2(up, cp, unp, cnp, uq, cq, unq, cnq, ges);
          }
        } else if (m == 7) {
          for (auto corr : corres7) {
            auto up = src7[corr.first]->GetPointMean();
            auto cp = src7[corr.first]->GetPointCov();
            Vector2d unp = src7[corr.first]->GetPointEvecs().col(0);
            auto uq = tgt7[corr.second]->GetPointMean();
            auto cq = tgt7[corr.second]->GetPointCov();
            Vector2d unq = tgt7[corr.second]->GetPointEvecs().col(0);
            zz(j, k) += SNDTCostFunctor3::Cost2(up, cp, unp, uq, cq, unq, ges);
          }
        } else if (m == 8) {
          for (auto corr : corres6) {
            auto up = src6[corr.first]->GetPointMean();
            auto cp = src6[corr.first]->GetPointCov();
            Vector2d unp = src6[corr.first]->GetPointEvecs().col(0);
            auto uq = tgt6[corr.second]->GetPointMean();
            auto cq = tgt6[corr.second]->GetPointCov();
            Vector2d unq = tgt6[corr.second]->GetPointEvecs().col(0);
            zz(j, k) += SNDTCostFunctor3::Cost2(up, cp, unp, uq, cq, unq, ges);
          }
        }
      }
    }
  }

  string filepath = "/tmp/contour.txt";
  ofstream fout(filepath);
  for (int i = 0; i < sz; ++i) fout << x[i] << ((i + 1 == sz) ? "\n" : ", ");
  for (int i = 0; i < sz; ++i) fout << y[i] << ((i + 1 == sz) ? "\n" : ", ");
  for (int i = 0; i < sz; ++i) {
    for (int j = 0; j < sz; ++j) {
      fout << zz(i, j) << ((j + 1 == sz) ? "\n" : ", ");
    }
  }
  fout.close();

  /*int samples = 15;
  auto affs = RandomTransformGenerator2D(r).Generate(samples);

  for (auto aff : affs) {
    auto src = TransformPoints(tgt, aff);
    vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, Affine2d::Identity()}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{{src, Affine2d::Identity()}};

    SNDTParameters params;
    params.cell_size = cell_size;
    params.radius = radius;
    params.r_variance = params.t_variance = 0;
    auto tgt6 = MakeSNDTMap(datat, params);
    auto src6 = MakeSNDTMap(datas, params);
    auto T = SNDTMatch(tgt6, src6, params);

    D2DNDTParameters params;
    params.cell_size = cell_size;
    params.r_variance = params.t_variance = 0;
    auto tgt7 = MakeNDTMap(datat, params);
    auto src7 = MakeNDTMap(datas, params);
    auto T = SNDTMatch2(tgt7, src7, params);

    if ((aff * T).translation().isZero(1)) {
      cout << "s: " << TransNormRotDegAbsFromAffine2d(aff * T).transpose()
           << ", ";
    } else {
      cout << "f: " << TransNormRotDegAbsFromAffine2d(aff * T).transpose()
           << ", ";
    }
    cout << "Iter: " << params._iteration << " & " << params._ceres_iteration;
    cout << ", " << int(params._converge) << endl;
  }*/
}
