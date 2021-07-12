#pragma once

#include <bits/stdc++.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include "common/common.h"
#include "sndt/ndt_matcher.h"
#include "normal2d/Normal2dEstimation.h"

using namespace std;
using namespace Eigen;
using pcl::search::KdTree;
using pcl::PointXYZ;
typedef pcl::PointCloud<pcl::PointXYZ> PCXYZ;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PCXYZPtr;

MatrixXd MatrixXdFromPCL(PCXYZ pc) {
  MatrixXd ret;
  for (const auto &pt : pc.points) {
    ret.conservativeResize(3, ret.cols() + 1);
    ret.col(ret.cols() - 1) = Vector3d(pt.x, pt.y, pt.z);
  }
  return ret;
}

MatrixXd MatrixXdFromPCL(PCXYZPtr pc) {
  return MatrixXdFromPCL(*pc);
}

PCXYZPtr PCLFromMatrixXd(const MatrixXd &mtx) {
  PCXYZPtr ret(new PCXYZ);
  for (int i = 0; i < mtx.cols(); ++i)
    ret->push_back(pcl::PointXYZ(mtx(0, i), mtx(1, i), mtx(2, i)));
  return ret;
}

/**
 * @brief Wrapper of SNDT
 * 
 * @param mp 
 * @param mit 
 * @param params 
 */
void DoSNDT(common::MatchPackage &mp, common::MatchInternal &mit, const vector<double> &params) {
  double cell_size = params.at(1);
  double radius = params.at(2);
  double tf_eps = params.at(3);

  auto source = PCLFromMatrixXd(mp.source);
  auto sourcen = ComputeNormals(source, radius);
  NDTMap source_map(cell_size);
  source_map.LoadPointCloud(*source, *sourcen);

  auto target = PCLFromMatrixXd(mp.target);
  auto targetn = ComputeNormals(target, radius);
  NDTMap target_map(cell_size);
  target_map.LoadPointCloud(*target, *targetn);

  NDTMatcher matcher;
  matcher.SetStrategy(NDTMatcher::Strategy::kNEAREST_1_POINTS);
  matcher.SetThreshold(tf_eps);
  mp.result = matcher.CeresMatch(target_map, source_map, mp.guess);
  mp.iters = matcher.GetIteration();
}

/**
 * @param data MatrixXd 2xN & Affine2d
 * @param point_intrinsic variance (r_sig^2, theta_sig^2)
 * @param params (cell_size, radius)
 */
NDTMap MakeMap(const vector<pair<MatrixXd, Affine2d>> &data,
               const Vector2d &intrinsic,
               const vector<double> &params) {
  double cell_size = params.at(0);
  double radius = params.at(1);

  auto n = accumulate(data.begin(), data.end(), 0,
                      [](auto a, auto b) { return a + (int)b.first.cols(); });
  MatrixXd points(2, n);
  MatrixXd point_covs(2, 2 * n);

  int j = 0;
  for (const auto &elem : data) {
    auto pts = elem.first;
    auto T = elem.second;
    for (int i = 0; i < pts.cols(); ++i) {
      Vector2d pt = pts.col(i);
      double r2 = pt.squaredNorm();
      double theta = atan2(pt(1), pt(0));
      Matrix2d J = Rotation2Dd(theta).matrix();
      Matrix2d S = Vector2d(intrinsic(0), r2 * intrinsic(1)).asDiagonal();
      points.col(j) = T * pt;
      point_covs.block<2, 2>(0, 2 * j) = T.rotation() * J * S * J.transpose() * T.rotation().transpose();
      ++j;
    }
  }
  auto normals = ComputeNormals2(points, radius);
  NDTMap ret(cell_size);
  ret.LoadPointCloudWithCovariances(points, normals, point_covs);
  return ret;
}
