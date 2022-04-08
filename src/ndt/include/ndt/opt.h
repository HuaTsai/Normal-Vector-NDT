#pragma once
#include <ceres/ceres.h>
#include <ndt/options.h>

class Optimizer {
 public:
  Optimizer() = delete;

  explicit Optimizer(Options type);

  ~Optimizer();

  void AddResidualBlock(ceres::CostFunction *func);

  void BuildProblem(ceres::FirstOrderFunction *func);

  void Optimize();

  bool CheckConverge(const std::vector<Eigen::Affine3d> &tfs);

  void set_cur_tf(const Eigen::Affine3d &cur_tf) { cur_tf_ = cur_tf; }
  Eigen::Affine3d cur_tf() const { return cur_tf_; }

 private:
  Options type_;
  double xyzxyzw_[7];
  ceres::Problem problem_;
  ceres::GradientProblem *gproblem_;
  ceres::ProductParameterization *param_;
  Eigen::Affine3d cur_tf_;
  double threshold_;
  double threshold_ang_;
};

class Optimizer2D {
 public:
  Optimizer2D() = delete;

  explicit Optimizer2D(Options type);

  ~Optimizer2D();

  void AddResidualBlock(ceres::CostFunction *func);

  void BuildProblem(ceres::FirstOrderFunction *func);

  void Optimize();

  bool CheckConverge(const std::vector<Eigen::Affine2d> &tfs);

  void set_cur_tf(const Eigen::Affine2d &cur_tf) { cur_tf_ = cur_tf; }
  Eigen::Affine2d cur_tf() const { return cur_tf_; }

 private:
  Options type_;
  double xyt_[3];
  ceres::Problem problem_;
  ceres::GradientProblem *gproblem_;
  Eigen::Affine2d cur_tf_;
  double threshold_;
  double threshold_ang_;
};
