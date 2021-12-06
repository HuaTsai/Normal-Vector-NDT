/**
 * @file contourxt.cc
 * @brief Output to /tmp/contourxt.txt
 * @version 0.1
 * @date 2021-11-09
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

bool SF(Affine2d T) {
  auto rt = TransNormRotDegAbsFromAffine2d(T);
  return rt(0) < 1 && rt(1) < 3;
}

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

  // vector<double> xs{4, 8, 12, 16, 20, 24, 28};
  // vector<double> ts{5, 10, 15, 20, 25, 30, 35, 40};
  // for (auto x : xs) {
  // for (auto t : ts) {
  auto aff = Translation2d(4, 0) * Rotation2Dd(20 * M_PI / 180.);
  Vector3d gt(-4, 0, -20);
  cout << "aff: " << XYTDegreeFromAffine2d(aff).transpose() << " -> ";
  auto src = TransformPoints(tgt, aff);
  vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, Affine2d::Identity()}};
  vector<pair<vector<Vector2d>, Affine2d>> datas{{src, Affine2d::Identity()}};

  SICPParameters params3;
  params3.radius = radius;
  // params3.fixstrategy = FixStrategy::kFixY;
  auto tgt3 = MakePoints(datat, params3);
  auto src3 = MakePoints(datas, params3);
  auto T3 = SICPMatch(tgt3, src3, params3);
  if (SF(aff * T3)) cout << "M3, ";

  D2DNDTParameters params5;
  params5.cell_size = cell_size;
  params5.r_variance = params5.t_variance = 0;
  // params5.fixstrategy = FixStrategy::kFixY;
  auto tgt5 = MakeNDTMap(datat, params5);
  auto src5 = MakeNDTMap(datas, params5);
  auto T5 = D2DNDTMDMatch(tgt5, src5, params5);
  if (SF(aff * T5)) cout << "M5, ";

  SNDTParameters params6;
  params6.cell_size = cell_size;
  params6.radius = radius;
  params6.r_variance = params6.t_variance = 0;
  // params6.fixstrategy = FixStrategy::kFixY;
  auto tgt6 = MakeSNDTMap(datat, params6);
  auto src6 = MakeSNDTMap(datas, params6);
  auto T6 = SNDTMDMatch(tgt6, src6, params6);
  if (SF(aff * T6)) cout << "M6, ";

  D2DNDTParameters params7;
  params7.cell_size = cell_size;
  params7.r_variance = params7.t_variance = 0;
  // params7.fixstrategy = FixStrategy::kFixY;
  auto tgt7 = MakeNDTMap(datat, params7);
  auto src7 = MakeNDTMap(datas, params7);
  auto T7 = SNDTMDMatch2(tgt7, src7, params7);
  if (SF(aff * T7)) cout << "M7, ";
  cout << endl;
  // }
  // }
  cout << XYTDegreeFromAffine2d(T3).transpose() << endl;
  cout << XYTDegreeFromAffine2d(T5).transpose() << endl;
  cout << XYTDegreeFromAffine2d(T6).transpose() << endl;
  cout << XYTDegreeFromAffine2d(T7).transpose() << endl;

  constexpr int sz = 41;
  VectorXd x = VectorXd::LinSpaced(sz, -10, 10);
  VectorXd t = VectorXd::LinSpaced(sz, -30, 30);
  MatrixXd yy(sz, sz);

  string filepath = "/tmp/contourxt.txt";
  ofstream fout(filepath);
  for (int i = 0; i < 3; ++i) fout << gt[i] << ((i + 1 == 3) ? "\n" : ", ");
  for (int i = 0; i < sz; ++i) fout << x[i] << ((i + 1 == sz) ? "\n" : ", ");
  for (int i = 0; i < sz; ++i) fout << t[i] << ((i + 1 == sz) ? "\n" : ", ");
  if (m == 3) {
    auto tnms = ComputeNormals(tgt, radius);
    auto snms = ComputeNormals(src, radius);
    for (size_t iter = 0; iter < params3._corres.size(); ++iter) {
      auto xyt = XYTDegreeFromAffine2d(params3._sols[iter][0]);
      fout << xyt(0) << ", " << xyt(1) << ", " << xyt(2) << endl;
      for (int j = 0; j < sz; ++j) {
        for (int k = 0; k < sz; ++k) {
          auto ges = Translation2d(x(k), 0) * Rotation2Dd(t(j) * M_PI / 180.);
          yy(j, k) = M3Cost(tgt3, tnms, src3, snms, params3._corres[iter], ges);
          fout << yy(j, k) << ((k + 1 == sz) ? "\n" : ", ");
        }
      }
    }
  } else if (m == 5) {
    for (size_t iter = 0; iter < params5._corres.size(); ++iter) {
      auto xyt = XYTDegreeFromAffine2d(params5._sols[iter][0]);
      fout << xyt(0) << ", " << xyt(1) << ", " << xyt(2) << endl;
      for (int j = 0; j < sz; ++j) {
        for (int k = 0; k < sz; ++k) {
          auto ges = Translation2d(x(k), 0) * Rotation2Dd(t(j) * M_PI / 180.);
          yy(j, k) = M5Cost(tgt5, src5, params5._corres[iter], ges);
          fout << yy(j, k) << ((k + 1 == sz) ? "\n" : ", ");
        }
      }
    }
  } else if (m == 6) {
    for (size_t iter = 0; iter < params6._corres.size(); ++iter) {
      auto xyt = XYTDegreeFromAffine2d(params6._sols[iter][0]);
      fout << xyt(0) << ", " << xyt(1) << ", " << xyt(2) << endl;
      for (int j = 0; j < sz; ++j) {
        for (int k = 0; k < sz; ++k) {
          auto ges = Translation2d(x(k), 0) * Rotation2Dd(t(j) * M_PI / 180.);
          yy(j, k) = M6Cost(tgt6, src6, params6._corres[iter], ges);
          fout << yy(j, k) << ((k + 1 == sz) ? "\n" : ", ");
        }
      }
    }
  } else if (m == 7) {
    for (size_t iter = 0; iter < params7._corres.size(); ++iter) {
      auto xyt = XYTDegreeFromAffine2d(params7._sols[iter][0]);
      fout << xyt(0) << ", " << xyt(1) << ", " << xyt(2) << endl;
      for (int j = 0; j < sz; ++j) {
        for (int k = 0; k < sz; ++k) {
          auto ges = Translation2d(x(k), 0) * Rotation2Dd(t(j) * M_PI / 180.);
          yy(j, k) = M7Cost(tgt7, src7, params7._corres[iter], ges);
          fout << yy(j, k) << ((k + 1 == sz) ? "\n" : ", ");
        }
      }
    }
  }
  fout.close();
}
