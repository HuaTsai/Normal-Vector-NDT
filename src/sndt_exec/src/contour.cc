/**
 * @file contour.cc
 * @brief Output to /tmp/contour.txt
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

double M3Cost(vector<Vector2d> &tgt,
              vector<Vector2d> &tgtn,
              vector<Vector2d> &src,
              vector<Vector2d> &srcn,
              vector<pair<int, int>> corres,
              Affine2d T) {
  double ret = 0;
  for (auto corr : corres) {
    auto p = src[corr.first];
    auto np = srcn[corr.first];
    auto q = tgt[corr.second];
    auto nq = tgtn[corr.second];
    ret += SICPCostFunctor::Cost(p, np, q, nq, T);
  }
  return ret;
}

double M5Cost(NDTMap &tgt,
              NDTMap &src,
              vector<pair<int, int>> corres,
              Affine2d T) {
  double ret = 0;
  for (auto corr : corres) {
    auto cellp = src[corr.first];
    auto cellq = tgt[corr.second];
    auto up = cellp->GetPointMean();
    auto cp = cellq->GetPointCov();
    auto uq = cellq->GetPointMean();
    auto cq = cellq->GetPointCov();
    ret += D2DNDTMDCostFunctor::Cost(up, cp, uq, cq, T);
  }
  return ret;
}

double M6Cost(SNDTMap &tgt,
              SNDTMap &src,
              vector<pair<int, int>> corres,
              Affine2d T) {
  double ret = 0;
  for (auto corr : corres) {
    auto up = src[corr.first]->GetPointMean();
    auto cp = src[corr.first]->GetPointCov();
    auto unp = src[corr.first]->GetNormalMean();
    auto cnp = src[corr.first]->GetNormalCov();
    auto uq = tgt[corr.second]->GetPointMean();
    auto cq = tgt[corr.second]->GetPointCov();
    auto unq = tgt[corr.second]->GetNormalMean();
    auto cnq = tgt[corr.second]->GetNormalCov();
    ret += SNDTMDCostFunctor::Cost(up, cp, unp, cnp, uq, cq, unq, cnq, T);
  }
  return ret;
}

double M7Cost(NDTMap &tgt,
              NDTMap &src,
              vector<pair<int, int>> corres,
              Affine2d T) {
  double ret = 0;
  for (auto corr : corres) {
    auto up = src[corr.first]->GetPointMean();
    auto cp = src[corr.first]->GetPointCov();
    Vector2d unp = src[corr.first]->GetPointEvecs().col(0);
    auto uq = tgt[corr.second]->GetPointMean();
    auto cq = tgt[corr.second]->GetPointCov();
    Vector2d unq = tgt[corr.second]->GetPointEvecs().col(0);
    ret += SNDTMDCostFunctor2::Cost(up, cp, unp, uq, cq, unq, T);
  }
  return ret;
}

double M8Cost(SNDTMap &tgt,
              SNDTMap &src,
              vector<pair<int, int>> corres,
              Affine2d T) {
  double ret = 0;
  for (auto corr : corres) {
    auto up = src[corr.first]->GetPointMean();
    auto cp = src[corr.first]->GetPointCov();
    Vector2d unp = src[corr.first]->GetPointEvecs().col(0);
    auto uq = tgt[corr.second]->GetPointMean();
    auto cq = tgt[corr.second]->GetPointCov();
    Vector2d unq = tgt[corr.second]->GetPointEvecs().col(0);
    ret += SNDTMDCostFunctor2::Cost(up, cp, unp, uq, cq, unq, T);
  }
  return ret;
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

  constexpr int sz = 41;
  Vector3d gt;
  VectorXd x = VectorXd::LinSpaced(sz, -20, 20);
  VectorXd y = VectorXd::LinSpaced(sz, -20, 20);
  MatrixXd zz(sz, sz);

  double xs = 14.4966, ys = -3.85344;
  auto aff = Translation2d(xs, ys) * Rotation2Dd(0);
  gt << -xs, -ys, 0;
  auto src = TransformPoints(tgt, aff);
  vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, Affine2d::Identity()}};
  vector<pair<vector<Vector2d>, Affine2d>> datas{{src, Affine2d::Identity()}};

  SICPParameters params3;
  params3.radius = radius;
  params3.fixstrategy = FixStrategy::kFixTheta;
  auto tgt3 = MakePoints(datat, params3);
  auto src3 = MakePoints(datas, params3);
  auto T3 = SICPMatch(tgt3, src3, params3);

  D2DNDTParameters params5;
  params5.cell_size = cell_size;
  params5.r_variance = params5.t_variance = 0;
  params5.fixstrategy = FixStrategy::kFixTheta;
  auto tgt5 = MakeNDTMap(datat, params5);
  auto src5 = MakeNDTMap(datas, params5);
  auto T5 = D2DNDTMDMatch(tgt5, src5, params5);

  SNDTParameters params6;
  params6.cell_size = cell_size;
  params6.radius = radius;
  params6.r_variance = params6.t_variance = 0;
  params6.fixstrategy = FixStrategy::kFixTheta;
  auto tgt6 = MakeSNDTMap(datat, params6);
  auto src6 = MakeSNDTMap(datas, params6);
  auto T6 = SNDTMDMatch(tgt6, src6, params6);

  D2DNDTParameters params7;
  params7.cell_size = cell_size;
  params7.r_variance = params7.t_variance = 0;
  params7.fixstrategy = FixStrategy::kFixTheta;
  auto tgt7 = MakeNDTMap(datat, params7);
  auto src7 = MakeNDTMap(datas, params7);
  auto T7 = SNDTMDMatch2(tgt7, src7, params7);

  string filepath = "/tmp/contour.txt";
  ofstream fout(filepath);
  for (int i = 0; i < 3; ++i) fout << gt[i] << ((i + 1 == 3) ? "\n" : ", ");
  for (int i = 0; i < sz; ++i) fout << x[i] << ((i + 1 == sz) ? "\n" : ", ");
  for (int i = 0; i < sz; ++i) fout << y[i] << ((i + 1 == sz) ? "\n" : ", ");
  if (m == 3) {
    auto tnms = ComputeNormals(tgt, radius);
    auto snms = ComputeNormals(src, radius);
    for (size_t iter = 0; iter < params3._corres.size(); ++iter) {
      Vector2d xy = params3._sols[iter][0].translation();
      fout << xy(0) << ", " << xy(1) << ", 0\n";
      for (int j = 0; j < sz; ++j) {
        for (int k = 0; k < sz; ++k) {
          auto ges = Translation2d(x(k), y(j)) * Rotation2Dd(0);
          zz(j, k) = M3Cost(tgt3, tnms, src3, snms, params3._corres[iter], ges);
          fout << zz(j, k) << ((k + 1 == sz) ? "\n" : ", ");
        }
      }
    }
  } else if (m == 5) {
    for (size_t iter = 0; iter < params5._corres.size(); ++iter) {
      Vector2d xy = params5._sols[iter][0].translation();
      fout << xy(0) << ", " << xy(1) << ", 0\n";
      for (int j = 0; j < sz; ++j) {
        for (int k = 0; k < sz; ++k) {
          auto ges = Translation2d(x(k), y(j)) * Rotation2Dd(0);
          zz(j, k) = M5Cost(tgt5, src5, params5._corres[iter], ges);
          fout << zz(j, k) << ((k + 1 == sz) ? "\n" : ", ");
        }
      }
    }
  } else if (m == 6) {
    for (size_t iter = 0; iter < params6._corres.size(); ++iter) {
      Vector2d xy = params6._sols[iter][0].translation();
      fout << xy(0) << ", " << xy(1) << ", 0\n";
      for (int j = 0; j < sz; ++j) {
        for (int k = 0; k < sz; ++k) {
          auto ges = Translation2d(x(k), y(j)) * Rotation2Dd(0);
          zz(j, k) = M6Cost(tgt6, src6, params6._corres[iter], ges);
          fout << zz(j, k) << ((k + 1 == sz) ? "\n" : ", ");
        }
      }
    }
  } else if (m == 7) {
    for (size_t iter = 0; iter < params7._corres.size(); ++iter) {
      Vector2d xy = params7._sols[iter][0].translation();
      fout << xy(0) << ", " << xy(1) << ", 0\n";
      for (int j = 0; j < sz; ++j) {
        for (int k = 0; k < sz; ++k) {
          auto ges = Translation2d(x(k), y(j)) * Rotation2Dd(0);
          zz(j, k) = M7Cost(tgt7, src7, params7._corres[iter], ges);
          fout << zz(j, k) << ((k + 1 == sz) ? "\n" : ", ");
        }
      }
    }
  }
  fout.close();
}
