#pragma once

#include <bits/stdc++.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include "common/common.h"
#include "sndt/ndt_matcher_2d.hpp"
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

  NDTMatcherD2D2D matcher;
  matcher.set_strategy(NDTMatcherD2D2D::Strategy::kNEAREST_1_POINTS);
  matcher.set_threshold(tf_eps);
  mp.result = matcher.CeresMatch(target_map, source_map, mp.guess);
  mp.iters = matcher.iteration();

  mit.set_source(mp.source);
  mit.set_target(mp.target);
  mit.set_cell_size(cell_size);
  mit.set_corrs(matcher.mit().corrs());
  mit.set_tfs(matcher.mit().tfs());
  mit.set_has_data(true);
}
