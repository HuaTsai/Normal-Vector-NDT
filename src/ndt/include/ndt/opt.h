#pragma once
#include <ceres/ceres.h>
#include <ndt/costs.h>
#include <ndt/options.h>

class Optimizer {
 public:
  Optimizer() = delete;

  explicit Optimizer(Options type);

  ~Optimizer();

  void BuildProblem(ceres::FirstOrderFunction *func);

  void Optimize();

  bool CheckConverge(const std::vector<Eigen::Affine3d> &tfs);

  bool CheckConverge(const std::vector<Eigen::Affine2d> &tfs);

  void set_cur_tf3(const Eigen::Affine3d &cur_tf3) { cur_tf3_ = cur_tf3; }
  Eigen::Affine3d cur_tf3() const { return cur_tf3_; }

  void set_cur_tf2(const Eigen::Affine2d &cur_tf2) { cur_tf2_ = cur_tf2; }
  Eigen::Affine2d cur_tf2() const { return cur_tf2_; }

 private:
  Options type_;
  double xyzrpy_[6];
  double xyzxyzw_[7];
  double xyt_[3];
  ceres::GradientProblem *problem_;
  ceres::LocalParameterization *param_;
  Eigen::Affine3d cur_tf3_;
  Eigen::Affine2d cur_tf2_;
  double threshold_;
  double threshold_ang_;
  Eigen::Vector2d tlang_;
};
