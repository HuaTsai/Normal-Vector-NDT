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
#include <sndt/pcl_utils.h>
#include <sndt/visuals.h>

class InspectCallback : public ceres::IterationCallback {
 public:
  InspectCallback(CommonParameters &params, double &x, double &y, double &t,
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

struct OptimizeObjects {
  OptimizeObjects()
      : x(0),
        y(0),
        t(0),
        loss(new ceres::LossFunctionWrapper(nullptr, ceres::TAKE_OWNERSHIP)),
        cur_tf(Eigen::Affine2d::Identity()) {}

  void AddResidualBlock(ceres::CostFunction *cost_function) {
    problem.AddResidualBlock(cost_function, loss, &x, &y, &t);
  }

  void Optimize(CommonParameters &params) {
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = params.verbose;
    options.linear_solver_type = params.solver;
    options.max_num_iterations = params.ceres_max_iterations;
    // options.min_lm_diagonal = 5e-3;
    // options.use_inner_iterations = true;
    // options.minimizer_type = ceres::LINE_SEARCH;
    // problem.SetParameterBlockConstant(&t);

    // Sense Solution
    options.update_state_every_iteration = true;
    auto inspectcb = std::make_shared<InspectCallback>(params, x, y, t, cur_tf);
    options.callbacks.push_back(inspectcb.get());

    std::vector<ceres::ResidualBlockId> vrbds;
    problem.GetResidualBlocks(&vrbds);
    params._costs.push_back(std::vector<std::pair<double, double>>(vrbds.size()));

    for (size_t i = 0; i < vrbds.size(); ++i)
      problem.EvaluateResidualBlock(vrbds[i], false, &params._costs.back()[i].first,
                                    nullptr, nullptr);
    // XXX: change huber threshold
    // if (params.method == CommonParameters::Method::kICP) {
    //   loss->Reset(new ceres::HuberLoss(1), ceres::TAKE_OWNERSHIP);
    // }
    // if (params.method == CommonParameters::Method::kSNDT) {
    //   loss->Reset(new ceres::CauchyLoss(5), ceres::TAKE_OWNERSHIP);
    // }
    // loss->Reset(new ceres::HuberLoss(std::max(costs[3 * costs.size() /
    // 4], 1.)), ceres::TAKE_OWNERSHIP); loss->Reset(new
    // ceres::HuberLoss(params.huber), ceres::TAKE_OWNERSHIP); loss->Reset(new
    // ceres::HuberLoss(1.0), ceres::TAKE_OWNERSHIP);

    ceres::Solver::Summary summary;
    auto t1 = GetTime();
    ceres::Solve(options, &problem, &summary);
    auto t2 = GetTime();

    for (size_t i = 0; i < vrbds.size(); ++i)
      problem.EvaluateResidualBlock(vrbds[i], false, &params._costs.back()[i].second,
                                    nullptr, nullptr);

    params._usedtime.optimize += GetDiffTime(t1, t2);
    ++params._iteration;
    params._ceres_iteration += summary.num_linear_solves;

    cur_tf = Eigen::Translation2d(x, y) * Eigen::Rotation2Dd(t) * cur_tf;
  }

  void CheckConverge(CommonParameters &params) {
    Eigen::Vector2d xy = Eigen::Vector2d(x, y);
    if (xy.norm() < params.threshold)
      params._converge = Converge::kThreshold;
    else if (params._iteration > params.max_iterations)
      params._converge = Converge::kMaxIterations;
  }

  double x, y, t;
  ceres::LossFunctionWrapper *loss;
  Eigen::Affine2d cur_tf;
  ceres::Problem problem;
};

class OutlierRejection {
 public:
  OutlierRejection() { mul_ = 1; }
  void AddCorrespondence(const Eigen::Vector2d &p, const Eigen::Vector2d &q) {
    distances_.push_back((p - q).norm());
  }
  std::vector<int> GetIndices() {
    int n = distances_.size();
    std::vector<int> ret;
#if 0
    auto mean = std::accumulate(distances_.begin(), distances_.end(), 0.) / n;
    double stdev = 0;
    for (int i = 0; i < n; ++i)
      stdev += (distances_[i] - mean) * (distances_[i] - mean);
    stdev = sqrt(stdev / (n - 1));
    for (int i = 0; i < n; ++i)
      if (distances_[i] < mean + mul_ * stdev) ret.push_back(i);
#else
    for (int i = 0; i < n; ++i) ret.push_back(i);
#endif
    return ret;
  }

 private:
  std::vector<double> distances_;
  double mul_;
};

Eigen::Vector2d FindNearestNeighbor(const Eigen::Vector2d &query,
                                    const pcl::KdTreeFLANN<pcl::PointXY> &kd) {
  pcl::PointXY pt;
  pt.x = query(0), pt.y = query(1);
  std::vector<int> idx{0};
  std::vector<float> dist2{0};
  int found = kd.nearestKSearch(pt, 1, idx, dist2);
  if (!found) {
    Eigen::Vector2d ret;
    ret.fill(std::numeric_limits<double>::quiet_NaN());
    return ret;
  }
  return Eigen::Vector2d(kd.getInputCloud()->at(idx[0]).x,
                         kd.getInputCloud()->at(idx[0]).y);
}

int FindNearestNeighborIndex(const Eigen::Vector2d &query,
                             const pcl::KdTreeFLANN<pcl::PointXY> &kd) {
  pcl::PointXY pt;
  pt.x = query(0), pt.y = query(1);
  std::vector<int> idx{0};
  std::vector<float> dist2{0};
  int found = kd.nearestKSearch(pt, 1, idx, dist2);
  if (!found) return -1;
  return idx[0];
}

Eigen::Affine2d ICPMatch(const std::vector<Eigen::Vector2d> &tpts,
                         const std::vector<Eigen::Vector2d> &spts,
                         ICPParameters &params,
                         const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto kd = MakeKDTree(tpts);

  auto cur_tf = guess_tf;
  while (params._converge == Converge::kNotConverge) {
    OptimizeObjects opt;
    opt.cur_tf = cur_tf;
    auto t1 = GetTime();
    auto next_pts = TransformPoints(spts, opt.cur_tf);

    OutlierRejection orj;
    std::vector<ceres::CostFunction *> residuals;
    for (size_t i = 0; i < next_pts.size(); ++i) {
      Eigen::Vector2d p = next_pts[i];
      if (!p.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      Eigen::Vector2d q = tpts[idx];
      if (!q.allFinite()) continue;
      orj.AddCorrespondence(p, q);
      residuals.push_back(ICPCostFunctor::Create(p, q));
    }
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i)
      opt.AddResidualBlock(residuals[indices[i]]);

    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf;
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
    OptimizeObjects opt;
    opt.cur_tf = cur_tf;
    auto t1 = GetTime();
    auto next_pts = TransformPoints(spts, opt.cur_tf);

    OutlierRejection orj;
    std::vector<ceres::CostFunction *> residuals;
    for (size_t i = 0; i < next_pts.size(); ++i) {
      Eigen::Vector2d p = next_pts[i];
      if (!p.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      Eigen::Vector2d q = tpts[idx], nq = tnms[idx];
      if (!q.allFinite() || !nq.allFinite()) continue;
      orj.AddCorrespondence(p, q);
      residuals.push_back(Pt2plICPCostFunctor::Create(p, q, nq));
    }
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i)
      opt.AddResidualBlock(residuals[indices[i]]);

    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf;
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
    OptimizeObjects opt;
    opt.cur_tf = cur_tf;
    auto t1 = GetTime();
    auto next_pts = TransformPoints(spts, opt.cur_tf);
    auto next_nms = TransformNormals(snms, opt.cur_tf);

    OutlierRejection orj;
    std::vector<ceres::CostFunction *> residuals;
    for (size_t i = 0; i < next_pts.size(); ++i) {
      Eigen::Vector2d p = next_pts[i], np = next_nms[i];
      if (!p.allFinite() || !np.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      Eigen::Vector2d q = tpts[idx], nq = tnms[idx];
      if (!q.allFinite() || !nq.allFinite()) continue;
      if (np.dot(nq) < 0) nq = -nq;
      orj.AddCorrespondence(p, q);
      residuals.push_back(SICPCostFunctor::Create(p, np, q, nq));
    }
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i)
      opt.AddResidualBlock(residuals[indices[i]]);

    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf;
  }
  auto t3 = GetTime();
  params._usedtime.others =
      GetDiffTime(t2, t3) - params._usedtime.optimize - params._usedtime.build;
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
    OptimizeObjects opt;
    opt.cur_tf = cur_tf;
    auto t1 = GetTime();
    auto next_pts = TransformPoints(spts, opt.cur_tf);

    OutlierRejection orj;
    std::vector<ceres::CostFunction *> residuals;
    for (size_t i = 0; i < next_pts.size(); ++i) {
      auto p = next_pts[i];
      if (!p.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      orj.AddCorrespondence(p, cellq->GetPointMean());
      residuals.push_back(P2DNDTCostFunctor::Create(p, cellq));
    }
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i)
      opt.AddResidualBlock(residuals[indices[i]]);

    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf;
  }
  auto t2 = GetTime();
  params._usedtime.others =
      GetDiffTime(t1, t2) - params._usedtime.optimize - params._usedtime.build;
  return cur_tf;
}

Eigen::Affine2d D2DNDTMatch(const NDTMap &target_map, const NDTMap &source_map,
                            D2DNDTParameters &params,
                            const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  while (params._converge == Converge::kNotConverge) {
    OptimizeObjects opt;
    opt.cur_tf = cur_tf;
    auto t1 = GetTime();
    auto next_map = source_map.PseudoTransformCells(opt.cur_tf);

    OutlierRejection orj;
    std::vector<ceres::CostFunction *> residuals;
    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      orj.AddCorrespondence(cellp->GetPointMean(), cellq->GetPointMean());
      residuals.push_back(D2DNDTCostFunctor::Create(cellp.get(), cellq));
    }
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i)
      opt.AddResidualBlock(residuals[indices[i]]);

    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf;
  }
  auto t2 = GetTime();
  params._usedtime.others =
      GetDiffTime(t1, t2) - params._usedtime.optimize - params._usedtime.build;
  return cur_tf;
}

Eigen::Affine2d SNDTMatch(const SNDTMap &target_map, const SNDTMap &source_map,
                          SNDTParameters &params,
                          const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  while (params._converge == Converge::kNotConverge) {
    OptimizeObjects opt;
    opt.cur_tf = cur_tf;
    auto t1 = GetTime();
    auto next_map = source_map.PseudoTransformCells(opt.cur_tf, true);

    params._corres.push_back({});
    OutlierRejection orj;
    std::vector<ceres::CostFunction *> residuals;
    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      auto up = cellp->GetPointMean();
      auto cp = cellp->GetPointCov();
      auto unp = cellp->GetNormalMean();
      auto cnp = cellp->GetNormalCov();
      auto uq = cellq->GetPointMean();
      auto cq = cellq->GetPointCov();
      auto unq = cellq->GetNormalMean();
      auto cnq = cellq->GetNormalCov();
      params._corres.back().push_back({i, target_map.GetCellIndex(cellq)});
      orj.AddCorrespondence(up, uq);
      residuals.push_back(
          SNDTCostFunctor2::Create(up, cp, unp, cnp, uq, cq, unq, cnq));
    }
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i)
      opt.AddResidualBlock(residuals[indices[i]]);

    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf;
  }
  auto t2 = GetTime();
  params._usedtime.others =
      GetDiffTime(t1, t2) - params._usedtime.optimize - params._usedtime.build;
  return cur_tf;
}

Eigen::Affine2d SNDTMatch2(const NDTMap &target_map, const NDTMap &source_map,
                           D2DNDTParameters &params,
                           const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  while (params._converge == Converge::kNotConverge) {
    OptimizeObjects opt;
    opt.cur_tf = cur_tf;
    auto t1 = GetTime();
    auto next_map = source_map.PseudoTransformCells(opt.cur_tf, true);

    OutlierRejection orj;
    std::vector<ceres::CostFunction *> residuals;
    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      auto up = cellp->GetPointMean();
      auto cp = cellp->GetPointCov();
      Eigen::Vector2d unp = cellp->GetPointEvecs().col(0);
      auto uq = cellq->GetPointMean();
      auto cq = cellq->GetPointCov();
      Eigen::Vector2d unq = cellq->GetPointEvecs().col(0);
      orj.AddCorrespondence(up, uq);
      residuals.push_back(
          SNDTCostFunctor3::Create(up, cp, unp, uq, cq, unq));
    }
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i)
      opt.AddResidualBlock(residuals[indices[i]]);

    auto t2 = GetTime();
    params._usedtime.build += GetDiffTime(t1, t2);

    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf;
  }
  auto t2 = GetTime();
  params._usedtime.others =
      GetDiffTime(t1, t2) - params._usedtime.optimize - params._usedtime.build;
  return cur_tf;
}

Eigen::Affine2d SNDTCellMatch(
    const SNDTCell *target_cell, const SNDTCell *source_cell,
    SNDTParameters &params,
    const Eigen::Affine2d &guess_tf, int method) {
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

    OptimizeObjects opt;
    opt.cur_tf = cur_tf;
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
      opt.AddResidualBlock(P2DNDTCostFunctor2::Create(up, uq, cq));
    else if (method == 5)
      opt.AddResidualBlock(D2DNDTCostFunctor2::Create(up, cp, uq, cq));
    else if (method == 6)
      opt.AddResidualBlock(SNDTCostFunctor2::Create(up, cp, unp, cnp, uq, cq, unq, cnq));
    else if (method == 7)
      opt.AddResidualBlock(SNDTCostFunctor3::Create(up, cp, unp, uq, cq, unq));
    opt.Optimize(params);
    opt.CheckConverge(params);
    cur_tf = opt.cur_tf;
    delete cellp;
  }
  return cur_tf;
}