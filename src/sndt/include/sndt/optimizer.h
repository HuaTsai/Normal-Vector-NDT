/**
 * @file optimizer.h
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Declaration of Optimizers
 * @version 0.1
 * @date 2022-02-26
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once
#include <sndt/parameters.h>

class InspectCallback : public ceres::IterationCallback {
 public:
  InspectCallback(CommonParameters &params,
                  double &x,
                  double &y,
                  double &t,
                  const Eigen::Affine2d &cur_tf);

  ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary);

 private:
  bool needinit_;
  CommonParameters &params_;
  double &x_;
  double &y_;
  double &t_;
  const Eigen::Affine2d cur_tf_;
};

class LeastSquareOptimize {
 public:
  LeastSquareOptimize();

  void AddResidualBlock(ceres::CostFunction *cost_function);

  void Optimize(CommonParameters &params);

  void CheckConverge(CommonParameters &params,
                     const std::vector<Eigen::Affine2d> &tfs);

  void set_cur_tf(const Eigen::Affine2d &cur_tf) { cur_tf_ = cur_tf; }
  Eigen::Affine2d cur_tf() const { return cur_tf_; }

 private:
  double xyt_[3];
  ceres::LossFunctionWrapper *loss_;
  Eigen::Affine2d cur_tf_;
  ceres::Problem problem_;
};

class GeneralOptimize {
 public:
  GeneralOptimize();

  ~GeneralOptimize();

  void BuildProblem(ceres::FirstOrderFunction *func);

  void Optimize(CommonParameters &params);

  void CheckConverge(CommonParameters &params,
                     const std::vector<Eigen::Affine2d> &tfs);

  void set_cur_tf(const Eigen::Affine2d &cur_tf) { cur_tf_ = cur_tf; }
  Eigen::Affine2d cur_tf() const { return cur_tf_; }

 private:
  double xyt_[3];
  Eigen::Affine2d cur_tf_;
  ceres::GradientProblem *problem_;
};
