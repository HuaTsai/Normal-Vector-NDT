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
#include <sndt/matcher.h>
#include <sndt/pcl_utils.h>
#include <normal2d/normal2d.h>
#include <sndt/visuals.h>
#include <common/other_utils.h>
#include <sndt/cost_functors.h>

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
      auto next_tf = Eigen::Translation2d(x_, y_) *
                     Eigen::Rotation2Dd(t_) * cur_tf_;
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

    // Sense Solution
    options.update_state_every_iteration = true;
    auto inspectcb = std::make_shared<InspectCallback>(params, x, y, t, cur_tf);
    options.callbacks.push_back(inspectcb.get());

    std::vector<ceres::ResidualBlockId> vrbds;
    problem.GetResidualBlocks(&vrbds);
    params._costs.push_back(std::vector<double>(vrbds.size()));

    // XXX: change huber threshold
    std::vector<double> costs(vrbds.size());
    for (size_t i = 0; i < vrbds.size(); ++i) {
      problem.EvaluateResidualBlock(vrbds[i], false, &params._costs.back()[i], nullptr, nullptr);
      costs[i] = params._costs.back()[i];
    }
    sort(costs.begin(), costs.end());
    // loss->Reset(new ceres::HuberLoss(std::max(costs[3 * costs.size() / 4], 1.)), ceres::TAKE_OWNERSHIP);
    // loss->Reset(new ceres::HuberLoss(params.huber), ceres::TAKE_OWNERSHIP);
    // loss->Reset(new ceres::HuberLoss(1.0), ceres::TAKE_OWNERSHIP);

    ceres::Solver::Summary summary;
    auto t1 = GetTime();
    ceres::Solve(options, &problem, &summary);
    auto t2 = GetTime();
    params._usedtime.optimize += GetDiffTime(t1, t2);
    ++params._iteration;

    cur_tf = Eigen::Translation2d(x, y) * Eigen::Rotation2Dd(t) * cur_tf;
  }

  void CheckConverge(CommonParameters &params) {
    if (Eigen::Vector2d(x, y).norm() < params.threshold)
        params._converge = Converge::kThreshold;
    if (params._iteration > params.max_iterations &&
        params._converge != Converge::kThreshold)
      params._converge = Converge::kMaxIterations;
  }

  double x, y, t;
  ceres::LossFunctionWrapper *loss;
  Eigen::Affine2d cur_tf;
  ceres::Problem problem;
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

Eigen::Affine2d ICPMatch(
    const std::vector<Eigen::Vector2d> &tpts,
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
    std::vector<Eigen::Vector2d> next_pts(spts.size());
    std::transform(spts.begin(), spts.end(), next_pts.begin(),
                   [&opt](auto p) { return opt.cur_tf * p; });

    for (size_t i = 0; i < next_pts.size(); ++i) {
      Eigen::Vector2d p = next_pts[i];
      if (!p.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      Eigen::Vector2d q = tpts[idx];
      if (!q.allFinite()) continue;
      opt.AddResidualBlock(ICPCostFunctor::Create(p, q));
    }
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

Eigen::Affine2d Pt2plICPMatch(
    const std::vector<Eigen::Vector2d> &tpts,
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
    std::vector<Eigen::Vector2d> next_pts(spts.size());
    std::transform(spts.begin(), spts.end(), next_pts.begin(),
                   [&opt](auto p) { return opt.cur_tf * p; });

    for (size_t i = 0; i < next_pts.size(); ++i) {
      Eigen::Vector2d p = next_pts[i];
      if (!p.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      Eigen::Vector2d q = tpts[idx], nq = tnms[idx];
      if (!q.allFinite() || !nq.allFinite()) continue;
      opt.AddResidualBlock(Pt2plICPCostFunctor::Create(p, q, nq));
    }
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

Eigen::Affine2d SICPMatch(
    const std::vector<Eigen::Vector2d> &tpts,
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
    std::vector<Eigen::Vector2d> next_pts(spts.size());
    std::vector<Eigen::Vector2d> next_nms(snms.size());
    std::transform(spts.begin(), spts.end(), next_pts.begin(),
                   [&opt](auto p) { return opt.cur_tf * p; });
    // The return type of cur_tf * p is const Eigen::Matrix<double, 2, 1>.
    // However, The return type of cur_tf.rotation() * p is const
    // Eigen::Product<Eigen::Matrix<double, 2, 2>, Eigen::Matrix<double, 2, 1>,
    // 0>, so that we need to use ctor of Eigen::Vector2d to avoid bug.
    std::transform(
        snms.begin(), snms.end(), next_nms.begin(),
        [&opt](auto p) { return Eigen::Vector2d(opt.cur_tf.rotation() * p); });

    for (size_t i = 0; i < next_pts.size(); ++i) {
      Eigen::Vector2d p = next_pts[i], np = next_nms[i];
      if (!p.allFinite() || !np.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      Eigen::Vector2d q = tpts[idx], nq = tnms[idx];
      if (!q.allFinite() || !nq.allFinite()) continue;
      if (np.dot(nq) < 0) nq = -nq;
      opt.AddResidualBlock(SICPCostFunctor::Create(p, np, q, nq));
    }
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

Eigen::Affine2d P2DNDTMatch(
    const NDTMap &target_map, const std::vector<Eigen::Vector2d> &spts,
    P2DNDTParameters &params,
    const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  while (params._converge == Converge::kNotConverge) {
    OptimizeObjects opt;
    opt.cur_tf = cur_tf;
    auto t1 = GetTime();
    std::vector<Eigen::Vector2d> next_pts(spts.size());
    std::transform(spts.begin(), spts.end(), next_pts.begin(),
                   [&opt](auto p) { return opt.cur_tf * p; });

    for (size_t i = 0; i < next_pts.size(); ++i) {
      auto p = next_pts[i];
      if (!p.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      opt.AddResidualBlock(P2DNDTCostFunctor::Create(p, cellq));
      if (params._iteration == 0 && P2DNDTCostFunctor::Cost(p, cellq) > 500) {
        std::cout << "cost = " << P2DNDTCostFunctor::Cost(p, cellq) << std::endl;
        std::cout << "p = " << p.transpose() << std::endl;
        std::cout << cellq->ToString() << std::endl;
      }
    }
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

Eigen::Affine2d D2DNDTMatch(
    const NDTMap &target_map, const NDTMap &source_map,
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

    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      opt.AddResidualBlock(D2DNDTCostFunctor::Create(cellp.get(), cellq));
    }
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
      if (unp.dot(unq) < 0) { unq = -unq; }
      opt.AddResidualBlock(
          SNDTCostFunctor2::Create(up, cp, unp, cnp, uq, cq, unq, cnq));
    }
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
