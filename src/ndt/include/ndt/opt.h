#pragma once
#include <ceres/ceres.h>

class Optimizer {
 public:
  enum class OptType { kLS, kTR };

  Optimizer() = delete;

  explicit Optimizer(OptType type);

  ~Optimizer();

  void AddResidualBlock(ceres::CostFunction *func);

  void BuildProblem(ceres::FirstOrderFunction *func);

  void Optimize();

  bool CheckConverge(const std::vector<Eigen::Affine3d> &tfs);

  void set_cur_tf(const Eigen::Affine3d &cur_tf) { cur_tf_ = cur_tf; }
  Eigen::Affine3d cur_tf() const { return cur_tf_; }

 private:
  OptType type_;
  double xyzxyzw_[7];
  ceres::Problem problem_;
  ceres::GradientProblem *gproblem_;
  ceres::ProductParameterization *param_;
  Eigen::Affine3d cur_tf_;
  double threshold_;
  double threshold_ang_;
};
