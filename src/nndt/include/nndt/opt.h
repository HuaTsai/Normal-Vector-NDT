#pragma once
#include <ceres/ceres.h>

class Optimizer {
 public:
  Optimizer();

  ~Optimizer();

  void BuildProblem(ceres::FirstOrderFunction *func);

  void Optimize();

  bool CheckConverge();

  void set_cur_tf(const Eigen::Affine3d &cur_tf) { cur_tf_ = cur_tf; }
  Eigen::Affine3d cur_tf() const { return cur_tf_; }

 private:
  double xyzxyzw_[7];
  ceres::GradientProblem *problem_;
  ceres::ProductParameterization *param_;
  Eigen::Affine3d cur_tf_;
  double threshold_;
  int iteration_;
  int max_iterations_;
};
