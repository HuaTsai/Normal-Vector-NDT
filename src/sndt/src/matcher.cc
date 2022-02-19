/**
 * @file matcher.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Definition of Matching Algorithm
 * @version 0.1
 * @date 2021-07-29
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <ceres/ceres.h>
#include <common/eigen_utils.h>
#include <common/other_utils.h>
#include <normal2d/normal2d.h>
#include <sndt/cost_functors.h>
#include <sndt/matcher.h>
#include <sndt/visuals.h>
#include <sndt/outlier_reject.h>

class InspectCallback : public ceres::IterationCallback {
 public:
  InspectCallback(CommonParameters &params,
                  double &x,
                  double &y,
                  double &t,
                  const Eigen::Affine2d &cur_tf)
      : needinit_(true),
        params_(params),
        x_(x),
        y_(y),
        t_(t),
        cur_tf_(cur_tf) {}
  ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) {
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
  LeastSquareOptimize()
      : loss_(new ceres::LossFunctionWrapper(nullptr, ceres::TAKE_OWNERSHIP)),
        cur_tf_(Eigen::Affine2d::Identity()) {
    memset(xyt_, 0, sizeof(xyt_));
  }

  void AddResidualBlock(ceres::CostFunction *cost_function) {
    problem_.AddResidualBlock(cost_function, loss_, xyt_);
  }

  void Optimize(CommonParameters &params) {
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = params.verbose;
    options.linear_solver_type = params.solver;
    options.max_num_iterations = params.ceres_max_iterations;

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

    if (params.save_costs) {
      params._all_costs.push_back({summary.initial_cost, summary.final_cost});
      for (size_t i = 0; i < rbids.size(); ++i)
        problem_.EvaluateResidualBlock(
            rbids[i], false, &params._costs.back()[i].second, nullptr, nullptr);
    }

    cur_tf_ = Eigen::Translation2d(xyt_[0], xyt_[1]) *
              Eigen::Rotation2Dd(xyt_[2]) * cur_tf_;
  }

  void CheckConverge(CommonParameters &params) {
    Eigen::Vector2d xy = Eigen::Vector2d(xyt_[0], xyt_[1]);
    if (xy.norm() < params.threshold)
      params._converge = Converge::kThreshold;
    else if (params._iteration > params.max_iterations)
      params._converge = Converge::kMaxIterations;
  }

  void CheckConverge2(CommonParameters &params,
                      const std::vector<Eigen::Affine2d> &tfs) {
    Eigen::Vector2d xy = Eigen::Vector2d(xyt_[0], xyt_[1]);
    if (xy.norm() < params.threshold && xyt_[2] * 180. / M_PI < params.threshold_t) {
      params._converge = Converge::kThreshold;
      return;
    }
    if (params._iteration > params.max_iterations) {
      params._converge = Converge::kMaxIterations;
      return;
    }
    for (auto tf : tfs) {
      Eigen::Vector2d diff = TransNormRotDegAbsFromAffine2d(tf.inverse() * cur_tf_);
      if (diff(0) < params.threshold && diff(1) * M_PI / 180. < params.threshold_t) {
        params._converge = Converge::kThreshold;
        return;
      }
    }
  }

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
  GeneralOptimize() : cur_tf_(Eigen::Affine2d::Identity()) {
    memset(xyt_, 0, sizeof(xyt_));
  }

  ~GeneralOptimize() { delete problem_; }

  void BuildProblem(ceres::FirstOrderFunction *func) {
    problem_ = new ceres::GradientProblem(func);
  }

  void Optimize(CommonParameters &params) {
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

    // FIXME: iteration count api???
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

  void CheckConverge(CommonParameters &params) {
    Eigen::Vector2d xy = Eigen::Vector2d(xyt_[0], xyt_[1]);
    if (xy.norm() < params.threshold)
      params._converge = Converge::kThreshold;
    else if (params._iteration > params.max_iterations)
      params._converge = Converge::kMaxIterations;
  }

  void CheckConverge2(CommonParameters &params,
                      const std::vector<Eigen::Affine2d> &tfs) {
    Eigen::Vector2d xy = Eigen::Vector2d(xyt_[0], xyt_[1]);
    if (xy.norm() < params.threshold && xyt_[2] * 180. / M_PI < params.threshold_t) {
      params._converge = Converge::kThreshold;
      return;
    }
    if (params._iteration > params.max_iterations) {
      params._converge = Converge::kMaxIterations;
      return;
    }
    for (auto tf : tfs) {
      Eigen::Vector2d diff = TransNormRotDegAbsFromAffine2d(tf.inverse() * cur_tf_);
      if (diff(0) < params.threshold && diff(1) * M_PI / 180. < params.threshold_t) {
        params._converge = Converge::kThreshold;
        return;
      }
    }
  }

  void set_cur_tf(const Eigen::Affine2d &cur_tf) { cur_tf_ = cur_tf; }
  Eigen::Affine2d cur_tf() const { return cur_tf_; }

 private:
  double xyt_[3];
  Eigen::Affine2d cur_tf_;
  ceres::GradientProblem *problem_;
};

Eigen::Affine2d ICPMatch(const std::vector<Eigen::Vector2d> &tpts,
                         const std::vector<Eigen::Vector2d> &spts,
                         ICPParameters &params,
                         const Eigen::Affine2d &guess_tf) {
  auto kd = MakeKDTree(tpts);

  auto cur_tf = guess_tf;
  std::vector<Eigen::Affine2d> tfs;
  tfs.push_back(cur_tf);

  while (params._converge == Converge::kNotConverge) {
    params._usedtime.ProcedureStart(UsedTime::Procedure::kBuild);
    LeastSquareOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto next_pts = TransformPoints(spts, opt.cur_tf());

    RangeOutlierRejection orj;
    orj.set_reject(params.reject);
    std::vector<ceres::CostFunction *> residuals;
    std::vector<std::pair<int, int>> corres;
    for (size_t i = 0; i < next_pts.size(); ++i) {
      Eigen::Vector2d p = next_pts[i];
      if (!p.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      Eigen::Vector2d q = tpts[idx];
      if (!q.allFinite()) continue;
      corres.push_back({i, idx});
      orj.AddCorrespondence(p, q);
      residuals.push_back(ICPCostFunctor::Create(p, q));
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i) {
      opt.AddResidualBlock(residuals[indices[i]]);
      params._corres.back().push_back(corres[i]);
    }
    params._usedtime.ProcedureFinish();

    opt.Optimize(params);
    opt.CheckConverge2(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

Eigen::Affine2d SICPMatch(const std::vector<Eigen::Vector2d> &target_points,
                          const std::vector<Eigen::Vector2d> &source_points,
                          SICPParameters &params,
                          const Eigen::Affine2d &guess_tf) {
  params._usedtime.ProcedureStart(UsedTime::Procedure::kNormal);
  auto target_normals = ComputeNormals(target_points, params.radius);
  auto source_normals = ComputeNormals(source_points, params.radius);
  params._usedtime.ProcedureFinish();

  std::vector<Eigen::Vector2d> tpts, spts, tnms, snms;
  ExcludeInfinite(target_points, target_normals, tpts, tnms);
  ExcludeInfinite(source_points, source_normals, spts, snms);

  auto kd = MakeKDTree(tpts);

  auto cur_tf = guess_tf;
  std::vector<Eigen::Affine2d> tfs;
  tfs.push_back(cur_tf);
  while (params._converge == Converge::kNotConverge) {
    params._usedtime.ProcedureStart(UsedTime::Procedure::kBuild);
    LeastSquareOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto next_pts = TransformPoints(spts, opt.cur_tf());
    auto next_nms = TransformNormals(snms, opt.cur_tf());

    RangeOutlierRejection orj;
    orj.set_reject(params.reject);
    std::vector<ceres::CostFunction *> residuals;
    std::vector<std::pair<int, int>> corres;
    for (size_t i = 0; i < next_pts.size(); ++i) {
      Eigen::Vector2d p = next_pts[i], np = next_nms[i];
      if (!p.allFinite() || !np.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      Eigen::Vector2d q = tpts[idx], nq = tnms[idx];
      if (!q.allFinite() || !nq.allFinite()) continue;
      corres.push_back({i, idx});
      orj.AddCorrespondence(p, q);
      if (p.dot(np) < 0) np = -np;
      if (q.dot(nq) < 0) nq = -nq;
      residuals.push_back(SICPCostFunctor::Create(p, np, q, nq));
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i) {
      opt.AddResidualBlock(residuals[indices[i]]);
      params._corres.back().push_back(corres[i]);
    }
    params._usedtime.ProcedureFinish();

    opt.Optimize(params);
    opt.CheckConverge2(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

Eigen::Affine2d D2DNDTMatch(const NDTMap &target_map,
                            const NDTMap &source_map,
                            D2DNDTParameters &params,
                            const Eigen::Affine2d &guess_tf) {
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  std::vector<Eigen::Affine2d> tfs;
  tfs.push_back(cur_tf);
  while (params._converge == Converge::kNotConverge) {
    params._usedtime.ProcedureStart(UsedTime::Procedure::kBuild);
    GeneralOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto next_map = source_map.PseudoTransformCells(cur_tf);

    // AngleOutlierRejection orj;
    RangeOutlierRejection orj;
    // orj.set_reject(false);
    // orj.set_reject(true);
    orj.set_reject(params.reject);
    std::vector<Eigen::Vector2d> ups, uqs;
    std::vector<Eigen::Matrix2d> cps, cqs;
    std::vector<std::pair<int, int>> corres;
    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      corres.push_back({i, target_map.GetCellIndex(cellq)});
      ups.push_back(cellp->GetPointMean());
      cps.push_back(cellp->GetPointCov());
      uqs.push_back(cellq->GetPointMean());
      cqs.push_back(cellq->GetPointCov());
      orj.AddCorrespondence(cellp->GetPointMean(), cellq->GetPointMean());
      auto unp = Eigen::Vector2d(cellp->GetPointEvecs().col(0));
      auto unq = Eigen::Vector2d(cellq->GetPointEvecs().col(0));
      if (ups.back().dot(unp) < 0) unp = -unp;
      if (uqs.back().dot(unq) < 0) unq = -unq;
      // if (unp.dot(unq) < 0) unp = -unp;
      // orj.AddCorrespondence(unp, unq);
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    // XXX: we should change in place...
    std::vector<Eigen::Vector2d> ups2, uqs2;
    std::vector<Eigen::Matrix2d> cps2, cqs2;
    for (size_t i = 0; i < indices.size(); ++i) {
      ups2.push_back(ups[indices[i]]);
      cps2.push_back(cps[indices[i]]);
      uqs2.push_back(uqs[indices[i]]);
      cqs2.push_back(cqs[indices[i]]);
      params._corres.back().push_back(corres[indices[i]]);
    }

    if (params.d2 == -1)
      opt.BuildProblem(
          D2DNDTCostFunctor2::Create(ups2, cps2, uqs2, cqs2, params.cell_size, 0.2));
    else
      opt.BuildProblem(
          D2DNDTCostFunctor::Create(params.d2, ups2, cps2, uqs2, cqs2));
    params._usedtime.ProcedureFinish();

    opt.Optimize(params);
    opt.CheckConverge2(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

Eigen::Affine2d SNDTMatch2(const NDTMap &target_map,
                           const NDTMap &source_map,
                           D2DNDTParameters &params,
                           const Eigen::Affine2d &guess_tf) {
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  std::vector<Eigen::Affine2d> tfs;
  tfs.push_back(cur_tf);
  while (params._converge == Converge::kNotConverge) {
    params._usedtime.ProcedureStart(UsedTime::Procedure::kBuild);
    GeneralOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto next_map = source_map.PseudoTransformCells(opt.cur_tf(), true);

    RangeOutlierRejection orj;
    // orj.set_reject(params.reject);
    // AngleOutlierRejection orj;
    // orj.set_reject(true);
    orj.set_reject(params.reject);
    std::vector<Eigen::Vector2d> ups, unps, uqs, unqs;
    std::vector<Eigen::Matrix2d> cps, cqs;
    std::vector<std::pair<int, int>> corres;
    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      corres.push_back({i, target_map.GetCellIndex(cellq)});
      ups.push_back(cellp->GetPointMean());
      cps.push_back(cellp->GetPointCov());
      unps.push_back(Eigen::Vector2d(cellp->GetPointEvecs().col(0)));
      uqs.push_back(cellq->GetPointMean());
      cqs.push_back(cellq->GetPointCov());
      unqs.push_back(Eigen::Vector2d(cellq->GetPointEvecs().col(0)));
      if (ups.back().dot(unps.back()) < 0) unps.back() *= -1.;
      if (uqs.back().dot(unqs.back()) < 0) unqs.back() *= -1.;
      if (unps.back().dot(unqs.back()) < 0) unps.back() *= -1.;
      // orj.AddCorrespondence(unps.back(), unqs.back());
      orj.AddCorrespondence(cellp->GetPointMean(), cellq->GetPointMean());
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    // XXX: we should change in place...
    std::vector<Eigen::Vector2d> ups2, unps2, uqs2, unqs2;
    std::vector<Eigen::Matrix2d> cps2, cqs2;
    for (size_t i = 0; i < indices.size(); ++i) {
      ups2.push_back(ups[indices[i]]);
      cps2.push_back(cps[indices[i]]);
      unps2.push_back(unps[indices[i]]);
      uqs2.push_back(uqs[indices[i]]);
      cqs2.push_back(cqs[indices[i]]);
      unqs2.push_back(unqs[indices[i]]);
      params._corres.back().push_back(corres[indices[i]]);
    }

    opt.BuildProblem(SNDTCostFunctor2::Create(params.d2, ups2, cps2, unps2,
                                              uqs2, cqs2, unqs2));
    params._usedtime.ProcedureFinish();

    opt.Optimize(params);
    opt.CheckConverge2(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

#include "matcher_arc.cc"
