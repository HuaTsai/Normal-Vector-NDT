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

// class InspectCallback2 : public ceres::IterationCallback {
//  public:
//   InspectCallback2(CommonParameters &params,
//                    double *xyt,
//                    const Eigen::Affine2d &cur_tf)
//       : needinit_(true), params_(params), xyt_(xyt), cur_tf_(cur_tf) {}
//   ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) {
//     if (needinit_) {
//       params_._sols.push_back({cur_tf_});
//       needinit_ = false;
//     } else {
//       auto next_tf = Eigen::Translation2d(xyt_[0], xyt_[1]) *
//                      Eigen::Rotation2Dd(xyt_[2]) * cur_tf_;
//       params_._sols.back().push_back(next_tf);
//     }
//     return ceres::SOLVER_CONTINUE;
//   }

//  private:
//   bool needinit_;
//   CommonParameters &params_;
//   double *xyt_;
//   const Eigen::Affine2d cur_tf_;
// };

class LeastSquareOptimize {
 public:
  // Note that loss_ need not be deleted since ceres takes the ownership
  LeastSquareOptimize()
      : x_(0),
        y_(0),
        t_(0),
        loss_(new ceres::LossFunctionWrapper(nullptr, ceres::TAKE_OWNERSHIP)),
        cur_tf_(Eigen::Affine2d::Identity()) {}

  void AddResidualBlock(ceres::CostFunction *cost_function) {
    problem_.AddResidualBlock(cost_function, loss_, &x_, &y_, &t_);
  }

  void Optimize(CommonParameters &params) {
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = params.verbose;
    options.linear_solver_type = params.solver;
    options.max_num_iterations = params.ceres_max_iterations;

    // BUG: Due to transform multiplications, fix x or y will not work
    if (params.fixstrategy == FixStrategy::kFixX) {
      problem_.SetParameterBlockConstant(&x_);
    } else if (params.fixstrategy == FixStrategy::kFixY) {
      problem_.SetParameterBlockConstant(&y_);
    } else if (params.fixstrategy == FixStrategy::kFixTheta) {
      problem_.SetParameterBlockConstant(&t_);
    }

    options.update_state_every_iteration = params.inspect;
    if (params.inspect) {
      auto cb = std::make_shared<InspectCallback>(params, x_, y_, t_, cur_tf_);
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
    auto t1 = GetTime();
    ceres::Solve(options, &problem_, &summary);
    auto t2 = GetTime();
    params._usedtime.optimize += GetDiffTime(t1, t2);
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

    cur_tf_ = Eigen::Translation2d(x_, y_) * Eigen::Rotation2Dd(t_) * cur_tf_;
  }

  void CheckConverge(CommonParameters &params) {
    Eigen::Vector2d xy = Eigen::Vector2d(x_, y_);
    if (xy.norm() < params.threshold)
      params._converge = Converge::kThreshold;
    else if (params._iteration > params.max_iterations)
      params._converge = Converge::kMaxIterations;
  }

  void set_cur_tf(const Eigen::Affine2d &cur_tf) { cur_tf_ = cur_tf; }
  Eigen::Affine2d cur_tf() const { return cur_tf_; }

 private:
  double x_, y_, t_;
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

    // BUG: Inspect not work now.
    // options.update_state_every_iteration = params.inspect;
    // if (params.inspect) {
    //   auto cb = std::make_shared<InspectCallback>(params, xyt_[0], xyt_[1], xyt_[2], cur_tf_);
    //   options.callbacks.push_back(cb.get());
    // }
    if (params.inspect) {
      params._sols.push_back({cur_tf_});
    }

    auto t1 = GetTime();
    ceres::Solve(options, *problem_, xyt_, &summary);
    auto t2 = GetTime();
    params._usedtime.optimize += GetDiffTime(t1, t2);
    params._initial_cost = summary.initial_cost;
    params._final_cost = summary.final_cost;
    // FIXME: iteration count api???
    // params._ceres_iteration += summary.iterations;
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

  void set_cur_tf(const Eigen::Affine2d &cur_tf) { cur_tf_ = cur_tf; }
  Eigen::Affine2d cur_tf() const { return cur_tf_; }

 private:
  double xyt_[3];
  Eigen::Affine2d cur_tf_;
  ceres::GradientProblem *problem_;
};

class OutlierRejection {
 public:
  OutlierRejection() { multiplier_ = 1.5; }

  void AddCorrespondence(const Eigen::Vector2d &p, const Eigen::Vector2d &q) {
    distances_.push_back((p - q).norm());
  }
  std::vector<int> GetIndices() {
    int n = distances_.size();
    std::vector<int> ret;
    if (reject_) {
      auto mean = std::accumulate(distances_.begin(), distances_.end(), 0.) / n;
      double stdev = 0;
      for (int i = 0; i < n; ++i)
        stdev += (distances_[i] - mean) * (distances_[i] - mean);
      stdev = sqrt(stdev / (n - 1));
      for (int i = 0; i < n; ++i)
        if (distances_[i] < mean + multiplier_ * stdev) ret.push_back(i);
    } else {
      for (int i = 0; i < n; ++i) ret.push_back(i);
    }
    return ret;
  }

  void set_reject(bool reject) { reject_ = reject; }
  void set_multiplier(double multiplier) { multiplier_ = multiplier; }

 private:
  std::vector<double> distances_;
  double multiplier_;
  bool reject_;
};

Eigen::Affine2d ICPMatch(const std::vector<Eigen::Vector2d> &tpts,
                         const std::vector<Eigen::Vector2d> &spts,
                         ICPParameters &params,
                         const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto kd = MakeKDTree(tpts);

  auto cur_tf = guess_tf;
  while (params._converge == Converge::kNotConverge) {
    LeastSquareOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto t1 = GetTime();
    auto next_pts = TransformPoints(spts, opt.cur_tf());

    OutlierRejection orj;
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

    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf();
  }
  auto t2 = GetTime();
  params._usedtime.others =
      GetDiffTime(t1, t2) - params._usedtime.optimize - params._usedtime.build;
  return cur_tf;
}

Eigen::Affine2d Pt2plICPMatch(const std::vector<Eigen::Vector2d> &tpts,
                              const std::vector<Eigen::Vector2d> &spts,
                              Pt2plICPParameters &params,
                              const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto tnms = ComputeNormals(tpts, params.radius);
  auto t2 = GetTime();
  params._usedtime.normal = GetDiffTime(t1, t2);
  auto kd = MakeKDTree(tpts);

  auto cur_tf = guess_tf;
  while (params._converge == Converge::kNotConverge) {
    LeastSquareOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto t1 = GetTime();
    auto next_pts = TransformPoints(spts, opt.cur_tf());

    OutlierRejection orj;
    orj.set_reject(params.reject);
    std::vector<ceres::CostFunction *> residuals;
    std::vector<std::pair<int, int>> corres;
    for (size_t i = 0; i < next_pts.size(); ++i) {
      Eigen::Vector2d p = next_pts[i];
      if (!p.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      Eigen::Vector2d q = tpts[idx], nq = tnms[idx];
      if (!q.allFinite() || !nq.allFinite()) continue;
      corres.push_back({i, idx});
      orj.AddCorrespondence(p, q);
      residuals.push_back(Pt2plICPCostFunctor::Create(p, q, nq));
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i) {
      opt.AddResidualBlock(residuals[indices[i]]);
      params._corres.back().push_back(corres[i]);
    }

    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf();
  }
  auto t3 = GetTime();
  params._usedtime.others =
      GetDiffTime(t2, t3) - params._usedtime.optimize - params._usedtime.build;
  return cur_tf;
}

Eigen::Affine2d SICPMatch(const std::vector<Eigen::Vector2d> &tpts,
                          const std::vector<Eigen::Vector2d> &spts,
                          SICPParameters &params,
                          const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto tnms = ComputeNormals(tpts, params.radius);
  auto snms = ComputeNormals(spts, params.radius);
  auto t2 = GetTime();
  params._usedtime.normal = GetDiffTime(t1, t2);
  auto kd = MakeKDTree(tpts);

  auto cur_tf = guess_tf;
  while (params._converge == Converge::kNotConverge) {
    LeastSquareOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto t1 = GetTime();
    auto next_pts = TransformPoints(spts, opt.cur_tf());
    auto next_nms = TransformNormals(snms, opt.cur_tf());

    OutlierRejection orj;
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
      // if (np.dot(nq) < 0) nq = -nq;
      corres.push_back({i, idx});
      orj.AddCorrespondence(p, q);
      residuals.push_back(SICPCostFunctor::Create(p, np, q, nq));
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i) {
      opt.AddResidualBlock(residuals[indices[i]]);
      params._corres.back().push_back(corres[i]);
    }

    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf();
  }
  auto t3 = GetTime();
  params._usedtime.others =
      GetDiffTime(t2, t3) - params._usedtime.optimize - params._usedtime.build;
  return cur_tf;
}

Eigen::Affine2d P2DNDTMDMatch(const NDTMap &target_map,
                              const std::vector<Eigen::Vector2d> &spts,
                              P2DNDTParameters &params,
                              const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  while (params._converge == Converge::kNotConverge) {
    LeastSquareOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto t1 = GetTime();
    auto next_pts = TransformPoints(spts, opt.cur_tf());

    OutlierRejection orj;
    orj.set_reject(params.reject);
    std::vector<ceres::CostFunction *> residuals;
    std::vector<std::pair<int, int>> corres;
    for (size_t i = 0; i < next_pts.size(); ++i) {
      auto p = next_pts[i];
      if (!p.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      corres.push_back({i, target_map.GetCellIndex(cellq)});
      orj.AddCorrespondence(p, cellq->GetPointMean());
      auto q = cellq->GetPointMean();
      auto cq = cellq->GetPointCov();
      residuals.push_back(P2DNDTMDCostFunctor::Create(p, q, cq));
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i) {
      opt.AddResidualBlock(residuals[indices[i]]);
      params._corres.back().push_back(corres[i]);
    }

    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf();
  }
  auto t2 = GetTime();
  params._usedtime.others =
      GetDiffTime(t1, t2) - params._usedtime.optimize - params._usedtime.build;
  return cur_tf;
}

Eigen::Affine2d P2DNDTMatch(const NDTMap &target_map,
                            const std::vector<Eigen::Vector2d> &spts,
                            P2DNDTParameters &params,
                            const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  while (params._converge == Converge::kNotConverge) {
    GeneralOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto t1 = GetTime();
    auto next_pts = TransformPoints(spts, opt.cur_tf());

    OutlierRejection orj;
    orj.set_reject(params.reject);
    std::vector<Eigen::Vector2d> ps, uqs;
    std::vector<Eigen::Matrix2d> cqs;
    std::vector<std::pair<int, int>> corres;
    for (size_t i = 0; i < next_pts.size(); ++i) {
      auto p = next_pts[i];
      if (!p.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      corres.push_back({i, target_map.GetCellIndex(cellq)});
      ps.push_back(p);
      uqs.push_back(cellq->GetPointMean());
      cqs.push_back(cellq->GetPointCov());
      orj.AddCorrespondence(p, cellq->GetPointMean());
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    std::vector<Eigen::Vector2d> ps2, uqs2;
    std::vector<Eigen::Matrix2d> cqs2;
    for (size_t i = 0; i < indices.size(); ++i) {
      ps2.push_back(ps[i]);
      uqs2.push_back(uqs[i]);
      cqs2.push_back(cqs[i]);
      params._corres.back().push_back(corres[i]);
    }

    opt.BuildProblem(P2DNDTCostFunctor::Create(params.d2, ps2, uqs2, cqs2));
    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf();
  }
  auto t2 = GetTime();
  params._usedtime.others =
      GetDiffTime(t1, t2) - params._usedtime.optimize - params._usedtime.build;
  return cur_tf;
}

Eigen::Affine2d D2DNDTMDMatch(const NDTMap &target_map,
                              const NDTMap &source_map,
                              D2DNDTParameters &params,
                              const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());
  // auto kd = MakeKDTree(target_map.GetPoints());

  auto cur_tf = guess_tf;
  while (params._converge == Converge::kNotConverge) {
    LeastSquareOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto t1 = GetTime();
    auto next_map = source_map.PseudoTransformCells(opt.cur_tf());

    OutlierRejection orj;
    orj.set_reject(params.reject);
    std::vector<ceres::CostFunction *> residuals;
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
      auto up = cellp->GetPointMean();
      auto cp = cellp->GetPointCov();
      auto uq = cellq->GetPointMean();
      auto cq = cellq->GetPointCov();
      orj.AddCorrespondence(up, uq);
      residuals.push_back(D2DNDTMDCostFunctor::Create(up, cp, uq, cq));
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i) {
      opt.AddResidualBlock(residuals[indices[i]]);
      params._corres.back().push_back(corres[i]);
    }

    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf();
  }
  auto t2 = GetTime();
  params._usedtime.others =
      GetDiffTime(t1, t2) - params._usedtime.optimize - params._usedtime.build;
  return cur_tf;
}

Eigen::Affine2d D2DNDTMatch(const NDTMap &target_map,
                            const NDTMap &source_map,
                            D2DNDTParameters &params,
                            const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  while (params._converge == Converge::kNotConverge) {
    GeneralOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto t1 = GetTime();
    auto next_map = source_map.PseudoTransformCells(cur_tf);

    OutlierRejection orj;
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
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    // XXX: we should change in place...
    std::vector<Eigen::Vector2d> ups2, uqs2;
    std::vector<Eigen::Matrix2d> cps2, cqs2;
    for (size_t i = 0; i < indices.size(); ++i) {
      ups2.push_back(ups[i]);
      cps2.push_back(cps[i]);
      uqs2.push_back(uqs[i]);
      cqs2.push_back(cqs[i]);
      params._corres.back().push_back(corres[i]);
    }

    opt.BuildProblem(D2DNDTCostFunctor::Create(params.d2, ups2, cps2, uqs2, cqs2));
    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf();
  }
  auto t2 = GetTime();
  params._usedtime.others =
      GetDiffTime(t1, t2) - params._usedtime.optimize - params._usedtime.build;
  return cur_tf;
}

Eigen::Affine2d SNDTMDMatch(const SNDTMap &target_map,
                            const SNDTMap &source_map,
                            SNDTParameters &params,
                            const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());
  // auto kd = MakeKDTree(target_map.GetPoints());

  auto cur_tf = guess_tf;
  while (params._converge == Converge::kNotConverge) {
    LeastSquareOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto t1 = GetTime();
    auto next_map = source_map.PseudoTransformCells(opt.cur_tf(), true);

    OutlierRejection orj;
    orj.set_reject(params.reject);
    std::vector<ceres::CostFunction *> residuals;
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
      auto up = cellp->GetPointMean();
      auto cp = cellp->GetPointCov();
      auto unp = cellp->GetNormalMean();
      auto cnp = cellp->GetNormalCov();
      auto uq = cellq->GetPointMean();
      auto cq = cellq->GetPointCov();
      auto unq = cellq->GetNormalMean();
      auto cnq = cellq->GetNormalCov();
      orj.AddCorrespondence(up, uq);
      residuals.push_back(
          SNDTMDCostFunctor::Create(up, cp, unp, cnp, uq, cq, unq, cnq));
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i) {
      opt.AddResidualBlock(residuals[indices[i]]);
      params._corres.back().push_back(corres[i]);
    }

    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf();
  }
  auto t2 = GetTime();
  params._usedtime.others =
      GetDiffTime(t1, t2) - params._usedtime.optimize - params._usedtime.build;
  return cur_tf;
}

// TODO: finish it
Eigen::Affine2d SNDTMatch(const SNDTMap &target_map,
                          const SNDTMap &source_map,
                          SNDTParameters &params,
                          const Eigen::Affine2d &guess_tf) {
  // auto t1 = GetTime();
  // auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  // auto cur_tf = guess_tf;
  // while (params._converge == Converge::kNotConverge) {
  //   LeastSquareOptimize opt;
  //   opt.set_cur_tf(cur_tf);
  //   auto t1 = GetTime();
  //   auto next_map = source_map.PseudoTransformCells(opt.cur_tf(), true);

  //   OutlierRejection orj;
  //   orj.set_reject(params.reject);
  //   std::vector<ceres::CostFunction *> residuals;
  //   std::vector<std::pair<int, int>> corres;
  //   for (size_t i = 0; i < next_map.size(); ++i) {
  //     auto cellp = next_map[i];
  //     if (!cellp->HasGaussian()) continue;
  //     auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
  //     if (idx == -1) continue;
  //     auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
  //         kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
  //     if (!cellq || !cellq->HasGaussian()) continue;
  //     corres.push_back({i, target_map.GetCellIndex(cellq)});
  //     auto up = cellp->GetPointMean();
  //     auto cp = cellp->GetPointCov();
  //     auto unp = cellp->GetNormalMean();
  //     auto cnp = cellp->GetNormalCov();
  //     auto uq = cellq->GetPointMean();
  //     auto cq = cellq->GetPointCov();
  //     auto unq = cellq->GetNormalMean();
  //     auto cnq = cellq->GetNormalCov();
  //     orj.AddCorrespondence(up, uq);
  //     residuals.push_back(
  //         SNDTMDCostFunctor::Create(up, cp, unp, cnp, uq, cq, unq, cnq));
  //   }
  //   params._corres.push_back({});
  //   auto indices = orj.GetIndices();
  //   for (size_t i = 0; i < indices.size(); ++i) {
  //     opt.AddResidualBlock(residuals[indices[i]]);
  //     params._corres.back().push_back(corres[i]);
  //   }

  //   auto t2 = GetTime();
  //   params._usedtime.build += GetDiffTime(t1, t2);

  //   opt.Optimize(params);
  //   opt.CheckConverge(params);
  //   cur_tf = opt.cur_tf();
  // }
  // auto t2 = GetTime();
  // params._usedtime.others =
  //     GetDiffTime(t1, t2) - params._usedtime.optimize - params._usedtime.build;
  // return cur_tf;
  return Eigen::Affine2d::Identity();
}

Eigen::Affine2d SNDTMDMatch2(const NDTMap &target_map,
                             const NDTMap &source_map,
                             D2DNDTParameters &params,
                             const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  while (params._converge == Converge::kNotConverge) {
    LeastSquareOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto t1 = GetTime();
    auto next_map = source_map.PseudoTransformCells(opt.cur_tf(), true);

    OutlierRejection orj;
    orj.set_reject(params.reject);
    std::vector<ceres::CostFunction *> residuals;
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
      auto up = cellp->GetPointMean();
      auto cp = cellp->GetPointCov();
      Eigen::Vector2d unp = cellp->GetPointEvecs().col(0);
      auto uq = cellq->GetPointMean();
      auto cq = cellq->GetPointCov();
      Eigen::Vector2d unq = cellq->GetPointEvecs().col(0);
      orj.AddCorrespondence(up, uq);
      residuals.push_back(SNDTMDCostFunctor2::Create(up, cp, unp, uq, cq, unq));
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i) {
      opt.AddResidualBlock(residuals[indices[i]]);
      params._corres.back().push_back(corres[i]);
    }

    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf();
  }
  auto t2 = GetTime();
  params._usedtime.others =
      GetDiffTime(t1, t2) - params._usedtime.optimize - params._usedtime.build;
  return cur_tf;
}

Eigen::Affine2d SNDTMatch2(const NDTMap &target_map,
                           const NDTMap &source_map,
                           D2DNDTParameters &params,
                           const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  while (params._converge == Converge::kNotConverge) {
    GeneralOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto t1 = GetTime();
    auto next_map = source_map.PseudoTransformCells(opt.cur_tf(), true);

    OutlierRejection orj;
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
      orj.AddCorrespondence(cellp->GetPointMean(), cellq->GetPointMean());
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    // XXX: we should change in place...
    std::vector<Eigen::Vector2d> ups2, unps2, uqs2, unqs2;
    std::vector<Eigen::Matrix2d> cps2, cqs2;
    for (size_t i = 0; i < indices.size(); ++i) {
      ups2.push_back(ups[i]);
      cps2.push_back(cps[i]);
      unps2.push_back(unps[i]);
      uqs2.push_back(uqs[i]);
      cqs2.push_back(cqs[i]);
      unqs2.push_back(unqs[i]);
      params._corres.back().push_back(corres[i]);
    }

    opt.BuildProblem(
        SNDTCostFunctor2::Create(params.d2, ups2, cps2, unps2, uqs2, cqs2, unqs2));
    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf();
  }
  auto t2 = GetTime();
  params._usedtime.others =
      GetDiffTime(t1, t2) - params._usedtime.optimize - params._usedtime.build;
  return cur_tf;
}

Eigen::Affine2d SNDTCellMatch(const SNDTCell *target_cell,
                              const SNDTCell *source_cell,
                              SNDTParameters &params,
                              const Eigen::Affine2d &guess_tf,
                              int method) {
  auto cellq = target_cell;
  auto cur_tf = guess_tf;
  while (params._converge == Converge::kNotConverge) {
    Eigen::Matrix2d R = cur_tf.rotation();
    Eigen::Vector2d t = cur_tf.translation();
    double skew_rad = Eigen::Rotation2Dd(R).angle();
    SNDTCell *cellp = new SNDTCell();
    cellp->SetN(source_cell->GetN());
    cellp->SetPHasGaussian(source_cell->GetPHasGaussian());
    cellp->SetNHasGaussian(source_cell->GetNHasGaussian());
    cellp->SetSkewRad(skew_rad);
    cellp->SetCenter(R * source_cell->GetCenter() + t);
    cellp->SetSize(source_cell->GetSize());
    cellp->SetPointMean(R * source_cell->GetPointMean() + t);
    if (source_cell->GetPHasGaussian()) {
      cellp->SetPointCov(R * source_cell->GetPointCov() * R.transpose());
      cellp->SetPointEvals(source_cell->GetPointEvals());
      cellp->SetPointEvecs(R * source_cell->GetPointEvecs());
    }
    cellp->SetNormalMean(R * source_cell->GetNormalMean());
    if (source_cell->GetNHasGaussian()) {
      cellp->SetNormalCov(R * source_cell->GetNormalCov() * R.transpose());
      cellp->SetNormalEvals(source_cell->GetNormalEvals());
      cellp->SetNormalEvecs(R * source_cell->GetNormalEvecs());
    }
    for (auto pt : source_cell->GetPoints()) {
      if (pt.allFinite()) pt = R * pt + t;
      cellp->AddPoint(pt);
    }
    for (auto nm : source_cell->GetNormals()) {
      if (nm.allFinite()) nm = R * nm;
      cellp->AddNormal(nm);
    }

    LeastSquareOptimize opt;
    opt.set_cur_tf(cur_tf);
    if (!cellp->HasGaussian() || !cellq->HasGaussian()) continue;
    auto up = cellp->GetPointMean();
    auto cp = cellp->GetPointCov();
    auto unp = cellp->GetNormalMean();
    auto cnp = cellp->GetNormalCov();
    auto uq = cellq->GetPointMean();
    auto cq = cellq->GetPointCov();
    auto unq = cellq->GetNormalMean();
    auto cnq = cellq->GetNormalCov();
    if (method == 1)
      opt.AddResidualBlock(ICPCostFunctor::Create(up, uq));
    else if (method == 2)
      opt.AddResidualBlock(Pt2plICPCostFunctor::Create(up, uq, unq));
    else if (method == 3)
      opt.AddResidualBlock(SICPCostFunctor::Create(up, unp, uq, unq));
    else if (method == 4)
      opt.AddResidualBlock(P2DNDTMDCostFunctor::Create(up, uq, cq));
    else if (method == 5)
      opt.AddResidualBlock(D2DNDTMDCostFunctor::Create(up, cp, uq, cq));
    else if (method == 6)
      opt.AddResidualBlock(
          SNDTMDCostFunctor::Create(up, cp, unp, cnp, uq, cq, unq, cnq));
    else if (method == 7)
      opt.AddResidualBlock(
          SNDTMDCostFunctor2::Create(up, cp, unp, uq, cq, unq));
    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf();
    delete cellp;
  }
  return cur_tf;
}