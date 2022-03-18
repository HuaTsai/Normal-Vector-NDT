/**
 * @file optimizer.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Definition of Optimizers
 * @version 0.1
 * @date 2022-02-26
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <common/angle_utils.h>
#include <common/eigen_utils.h>
#include <sndt/optimizer.h>

InspectCallback::InspectCallback(CommonParameters &params,
                                 double &x,
                                 double &y,
                                 double &t,
                                 const Eigen::Affine2d &cur_tf)
    : needinit_(true), params_(params), x_(x), y_(y), t_(t), cur_tf_(cur_tf) {}

ceres::CallbackReturnType InspectCallback::operator()(
    const ceres::IterationSummary &summary) {
  if (needinit_) {
    params_._sols.push_back({cur_tf_});
    needinit_ = false;
  } else {
    auto next_tf =
        Eigen::Translation2d(x_, y_) * Eigen::Rotation2Dd(t_) * cur_tf_;
    params_._sols.back().push_back(next_tf);
  }
  return ceres::SOLVER_CONTINUE;
}

LeastSquareOptimize::LeastSquareOptimize()
    : loss_(new ceres::LossFunctionWrapper(nullptr, ceres::TAKE_OWNERSHIP)),
      cur_tf_(Eigen::Affine2d::Identity()) {
  memset(xyt_, 0, sizeof(xyt_));
}

void LeastSquareOptimize::AddResidualBlock(ceres::CostFunction *cost_function) {
  problem_.AddResidualBlock(cost_function, loss_, xyt_);
}

void LeastSquareOptimize::Optimize(CommonParameters &params) {
  ceres::Solver::Options options;
  options.minimizer_type = params.minimize;
  options.minimizer_progress_to_stdout = params.verbose;
  options.linear_solver_type = params.solver;
  options.max_num_iterations = params.ceres_max_iterations;
  // FIXME
  // options.use_nonmonotonic_steps = true;

  options.update_state_every_iteration = params.inspect;
  if (params.inspect) {
    auto cb = std::make_shared<InspectCallback>(params, xyt_[0], xyt_[1],
                                                xyt_[2], cur_tf_);
    options.callbacks.push_back(cb.get());
  }

  std::vector<ceres::ResidualBlockId> rbids;
  if (params.save_costs) {
    problem_.GetResidualBlocks(&rbids);
    params._costs.push_back(
        std::vector<std::pair<double, double>>(rbids.size()));
    for (size_t i = 0; i < rbids.size(); ++i)
      problem_.EvaluateResidualBlock(
          rbids[i], false, &params._costs.back()[i].first, nullptr, nullptr);
  }

  ceres::Solver::Summary summary;

  params._usedtime.ProcedureStart(UsedTime::Procedure::kOptimize);
  ceres::Solve(options, &problem_, &summary);
  params._usedtime.ProcedureFinish();

  params._initial_cost = summary.initial_cost;
  params._final_cost = summary.final_cost;
  ++params._iteration;
  params._ceres_iteration += summary.num_linear_solves;
  // for (const auto &sit : summary.iterations) {
    // TODO: There are only 0 or 1
    // params._search_iteration += sit.linear_solver_iterations;
  // }
  params._opt_time += summary.total_time_in_seconds * 1000.;

  if (params.save_costs) {
    params._all_costs.push_back({summary.initial_cost, summary.final_cost});
    for (size_t i = 0; i < rbids.size(); ++i)
      problem_.EvaluateResidualBlock(
          rbids[i], false, &params._costs.back()[i].second, nullptr, nullptr);
  }

  // FIXME
  cur_tf_ = Eigen::Translation2d(xyt_[0], xyt_[1]) *
            Eigen::Rotation2Dd(xyt_[2]) * cur_tf_;
}

void LeastSquareOptimize::CheckConverge(
    CommonParameters &params, const std::vector<Eigen::Affine2d> &tfs) {
  Eigen::Vector2d xy = Eigen::Vector2d(xyt_[0], xyt_[1]);
  if (xy.norm() < params.threshold && Rad2Deg(xyt_[2]) < params.threshold_t) {
    params._converge = Converge::kThreshold;
    return;
  }
  if (params._iteration > params.max_iterations) {
    params._converge = Converge::kMaxIterations;
    return;
  }
  for (auto tf : tfs) {
    Eigen::Vector2d diff =
        TransNormRotDegAbsFromAffine2d(tf.inverse() * cur_tf_);
    if (diff(0) < params.threshold && diff(1) < params.threshold_t) {
      params._converge = Converge::kBounce;
      return;
    }
  }
}

GeneralOptimize::GeneralOptimize() : cur_tf_(Eigen::Affine2d::Identity()) {
  memset(xyt_, 0, sizeof(xyt_));
}

GeneralOptimize::~GeneralOptimize() { delete problem_; }

void GeneralOptimize::BuildProblem(ceres::FirstOrderFunction *func) {
  problem_ = new ceres::GradientProblem(func);
}

void GeneralOptimize::Optimize(CommonParameters &params) {
  ceres::GradientProblemSolver::Options options;
  ceres::GradientProblemSolver::Summary summary;
  options.minimizer_progress_to_stdout = params.verbose;
  options.max_num_iterations = params.ceres_max_iterations;
  options.logging_type = ceres::SILENT;

  // BUG: Inspect not work now.
  // options.update_state_every_iteration = params.inspect;
  // if (params.inspect) {
  //   auto cb = std::make_shared<InspectCallback>(params, xyt_[0], xyt_[1],
  //   xyt_[2], cur_tf_); options.callbacks.push_back(cb.get());
  // }
  if (params.inspect) {
    params._sols.push_back({cur_tf_});
  }

  params._usedtime.ProcedureStart(UsedTime::Procedure::kOptimize);
  ceres::Solve(options, *problem_, xyt_, &summary);
  params._usedtime.ProcedureFinish();

  params._initial_cost = summary.initial_cost;
  params._final_cost = summary.final_cost;
  params._opt_time += summary.total_time_in_seconds * 1000.;

  // FIXME: iteration count api???
  // std::cout << summary.message << std::endl;
  // printf("r: %d, %.4f, g: %d, %.4f (%.4f), it:%ld\n",
  //        summary.num_cost_evaluations,
  //        summary.cost_evaluation_time_in_seconds * 1000.,
  //        summary.num_gradient_evaluations,
  //        summary.gradient_evaluation_time_in_seconds * 1000.,
  //        summary.gradient_evaluation_time_in_seconds * 1000. /
  //            summary.num_gradient_evaluations,
  //        summary.iterations.size());
  // printf("There are %ld iterations: ", summary.iterations.size());
  for (const auto &sit : summary.iterations) {
    // printf("(%d, %d), ", sit.line_search_function_evaluations,
    //        sit.line_search_iterations);
    params._search_iteration += sit.line_search_iterations;
  }
  // printf("\nTotal: %d\n", params._search_iteration);
  // std::cout << summary.cost_evaluation_time_in_seconds << ", ";
  // std::cout << summary.gradient_evaluation_time_in_seconds << ", ";
  // std::cout << summary.line_search_polynomial_minimization_time_in_seconds << " -> ";
  // std::cout << summary.total_time_in_seconds << std::endl;

  params._ceres_iteration += summary.iterations.size();
  ++params._iteration;
  if (params.verbose) {
    std::cout << summary.FullReport() << std::endl;
  }

  cur_tf_ = Eigen::Translation2d(xyt_[0], xyt_[1]) *
            Eigen::Rotation2Dd(xyt_[2]) * cur_tf_;
  if (params.inspect) {
    params._sols.back().push_back({cur_tf_});
  }
}

void GeneralOptimize::CheckConverge(CommonParameters &params,
                                    const std::vector<Eigen::Affine2d> &tfs) {
  Eigen::Vector2d xy = Eigen::Vector2d(xyt_[0], xyt_[1]);
  if (xy.norm() < params.threshold && Rad2Deg(xyt_[2]) < params.threshold_t) {
    params._converge = Converge::kThreshold;
    return;
  }
  if (params._iteration > params.max_iterations) {
    params._converge = Converge::kMaxIterations;
    return;
  }
  for (auto tf : tfs) {
    Eigen::Vector2d diff =
        TransNormRotDegAbsFromAffine2d(tf.inverse() * cur_tf_);
    if (diff(0) < params.threshold && diff(1) < params.threshold_t) {
      params._converge = Converge::kBounce;
      return;
    }
  }
}
