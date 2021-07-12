#pragma once

#include <bits/stdc++.h>
#include <ceres/ceres.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_ros/point_cloud.h>
#include <sndt/ndt_map.h>

using namespace std;
using namespace Eigen;

pcl::KdTreeFLANN<pcl::PointXYZ> MakeKDTree(const NDTMap &map);

class NDTMatcher {
 public:
  enum Strategy { kDEFAULT, kNEAREST_1_POINTS };
  NDTMatcher();

  Matrix3d CeresMatch(NDTMap &target_map, NDTMap &source_map, const Matrix3d &guess_tf = Matrix3d::Identity());

  int GetIteration() { return iteration_; }
  void SetStrategy(Strategy strategy) { strategy_ = strategy; }
  void SetMaxIterations(int max_iterations) { max_iterations_ = max_iterations; }
  void SetThreshold(double threshold) { threshold_ = threshold; }

 private:
  Strategy strategy_;
  int max_iterations_;
  double threshold_;
  int iteration_;
  // Strategy kDEFAULT
  int maxdist_of_cells_;
  // Strategy kNEAREST_1_POINTS
  double inlier_ratio_;
};
