#pragma once
#include <bits/stdc++.h>
#include <common/other_utils.h>
#include <nav_msgs/Path.h>

class TrajectoryEvaluation {
 public:
  enum EvalType { kDummy, kAbsolute, kRelativeBySingle, kRelativeByLength };

  TrajectoryEvaluation();

  std::pair<Stat, Stat> ComputeRMSError2D();

  void set_evaltype(const EvalType &evaltype) { evaltype_ = evaltype; }
  void set_estpath(const nav_msgs::Path &estpath) { estpath_ = estpath; }
  void set_gtpath(const nav_msgs::Path &gtpath) { gtpath_ = gtpath; }
  void set_length(double length) { length_ = length; }
  nav_msgs::Path estpath() const { return estpath_; }
  nav_msgs::Path gtpath() const { return gtpath_; }
  nav_msgs::Path align_estpath() const { return align_estpath_; }
  double gtlength() const { return gtlength_; }

 private:
  /**
   * @brief Absolute Trajectory Error
   * @details Implement from Sturm, Jürgen, et al. "A Benchmark for the
   * Evaluation of RGB-D SLAM Systems", 2012.
   */
  std::pair<Stat, Stat> AbsoluteTrajectoryError();

  /**
   * @brief Relative Pose Error by a consecutive matching
   * @details Implement from Sturm, Jürgen, et al. "A Benchmark for the
   * Evaluation of RGB-D SLAM Systems", 2012.
   */
  std::pair<Stat, Stat> RelativePoseErrorBySingle();

  /**
   * @brief Relative Pose Error by a fixed length
   * @details Implement from Sturm, Jürgen, et al. "A Benchmark for the
   * Evaluation of RGB-D SLAM Systems", 2012.
   */
  std::pair<Stat, Stat> RelativePoseErrorByLength();

  EvalType evaltype_;
  nav_msgs::Path estpath_;
  nav_msgs::Path gtpath_;
  nav_msgs::Path align_estpath_;
  nav_msgs::Path gtsync_;
  double length_;
  double gtlength_;
};
