#pragma once

#include <bits/stdc++.h>
#include <ceres/ceres.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_ros/point_cloud.h>
#include "sndt/ndt_visualizations.h"

using namespace std;
using namespace Eigen;
using visualization_msgs::MarkerArray;

pcl::KdTreeFLANN<pcl::PointXYZ> MakeKDTree(const NDTMap &map);

class NDTMatcher {
 public:
  enum Strategy { kDEFAULT, kNEAREST_1_POINTS, kUSE_CELLS_GREATER_THAN_TWO_POINTS };
  NDTMatcher();

  Matrix3d CeresMatch(NDTMap &target_map, NDTMap &source_map, const Affine2d &guess_tf = Affine2d::Identity()) {
    return CeresMatch(target_map, source_map, guess_tf.matrix());
  }

  Matrix3d CeresMatch(NDTMap &target_map, NDTMap &source_map, const Matrix3d &guess_tf = Matrix3d::Identity());

  pair<MarkerArray, MarkerArray> MarkerArrayOfMapCorrespondences(
      NDTMap &target_map, NDTMap &source_map, const Affine2d &T1,
      const Affine2d &T2) {
    return MarkerArrayOfMapCorrespondences(target_map, source_map, T1.matrix(),
                                           T2.matrix());
  }

  /**
   * @brief This function takes two transforms of start and end of matching, and
   * the first transform determines the correspondences.
   */
  pair<MarkerArray, MarkerArray> MarkerArrayOfMapCorrespondences(
      NDTMap &target_map, NDTMap &source_map, const Matrix3d &T1,
      const Matrix3d &T2);

  int GetIteration() { return iteration_; }
  void SetStrategy(Strategy strategy) { strategy_ = strategy; }
  void SetMaxIterations(int max_iterations) { max_iterations_ = max_iterations; }
  void SetThreshold(double threshold) { threshold_ = threshold; }

  vector<MarkerArray> vmas;
  vector<Vector2d> costs;
  bool usehuber;
  bool verbose;
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
