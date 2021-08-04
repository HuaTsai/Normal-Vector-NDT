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
#include <sndt/visuals.h>
#include <common/common_utils.hpp>
#include <sndt/cost_functors.h>

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

Eigen::Affine2d SICPMatch(
    const std::vector<Eigen::Vector2d> &target_points,
    const std::vector<Eigen::Vector2d> &source_points,
    SICPParameters &params,
    const Eigen::Affine2d &guess_tf) {
  params.InitializeOutput();
  auto tpts = ExcludeNaNInf(target_points);
  auto spts = ExcludeNaNInf(source_points);
  auto kd = MakeKDTree(tpts);
  auto tnms = ComputeNormals(tpts, params.radius);
  auto snms = ComputeNormals(spts, params.radius);
  auto cur_tf = guess_tf;
  Eigen::Affine2d best = cur_tf;

  while (params._converge == Converge::kNotConverge) {
    ++params._iteration;
    double x = 0, y = 0, t = 0;
    std::vector<Eigen::Vector2d> next_pts;
    std::vector<Eigen::Vector2d> next_nms;
    std::transform(spts.begin(), spts.end(), std::back_inserter(next_pts),
                   [&cur_tf](auto p) { return cur_tf * p; });
    std::transform(snms.begin(), snms.end(), std::back_inserter(next_nms),
                   [&cur_tf](auto p) { return cur_tf.rotation() * p; });

    ceres::Problem problem;
    for (size_t i = 0; i < source_points.size(); ++i) {
      Eigen::Vector2d p = next_pts[i], np = next_nms[i];
      if (!p.allFinite() || !np.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      Eigen::Vector2d q = tpts[idx], nq = tnms[idx];
      if (!q.allFinite() || !nq.allFinite()) continue;
      ceres::LossFunction *loss = nullptr;
      if (params.huber != 0)
        loss = new ceres::HuberLoss(params.huber);
      problem.AddResidualBlock(SICPCostFunctor::Create(p, np, q, nq), loss, &x, &y, &t);
    }

    ceres::Solver::Options options;
    if (params.verbose) options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = params.solver;
    options.max_num_iterations = params.max_iterations;

    ceres::Solver::Summary summary;
    auto t1 = GetTime();
    ceres::Solve(options, &problem, &summary);
    auto t2 = GetTime();
    params._usedtime.optimize += GetDiffTime(t1, t2);

    cur_tf = Eigen::Translation2d(x, y) * Eigen::Rotation2Dd(t) * cur_tf;

    // XXX: minimum cost?
    if (summary.final_cost > params._min_cost) {
      ++params._min_ctr;
      if (params._min_ctr == 5)
        params._converge = Converge::kConsecutiveMinimum;
    } else {
      params._min_cost = summary.final_cost;
      params._min_ctr = 0;
      best = cur_tf;
    }

    // if (Eigen::Vector2d(x, y).norm() < params.threshold)
    //   params._converge = Converge::kThreshold;
    if (params._iteration > params.max_iterations &&
        params._converge != Converge::kMaxIterations)
      params._converge = Converge::kMaxIterations;
  }
  return cur_tf;
}


Eigen::Affine2d NDTD2DMatch(
    const NDTMap &target_map, const NDTMap &source_map,
    NDTD2DParameters &params,
    const Eigen::Affine2d &guess_tf) {
  params.InitializeOutput();
  auto kd = MakeKDTree(target_map.GetPoints());
  auto cur_tf = guess_tf;
  Eigen::Affine2d best = cur_tf;

  while (params._converge == Converge::kNotConverge) {
    ++params._iteration;
    double x = 0, y = 0, t = 0;
    auto next_map = source_map.PseudoTransformCells(cur_tf);
    ceres::Problem problem;

    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      // XXX: direct get?
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      ceres::LossFunction *loss = nullptr;
      if (params.huber != 0)
        loss = new ceres::HuberLoss(params.huber);
      problem.AddResidualBlock(NDTD2DCostFunctor::Create(cellp.get(), cellq), loss, &x, &y, &t);
    }

    ceres::Solver::Options options;
    if (params.verbose) options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = params.solver;
    options.max_num_iterations = params.max_iterations;

    ceres::Solver::Summary summary;
    auto t1 = GetTime();
    ceres::Solve(options, &problem, &summary);
    auto t2 = GetTime();
    params._usedtime.optimize += GetDiffTime(t1, t2);

    cur_tf = Eigen::Translation2d(x, y) * Eigen::Rotation2Dd(t) * cur_tf;

    // XXX: minimum cost?
    if (summary.final_cost > params._min_cost) {
      ++params._min_ctr;
      if (params._min_ctr == 5)
        params._converge = Converge::kConsecutiveMinimum;
    } else {
      params._min_cost = summary.final_cost;
      params._min_ctr = 0;
      best = cur_tf;
    }

    // if (Eigen::Vector2d(x, y).norm() < params.threshold)
    //   params._converge = Converge::kThreshold;
    if (params._iteration > params.max_iterations &&
        params._converge != Converge::kMaxIterations)
      params._converge = Converge::kMaxIterations;
  }
  return best;
  // return cur_tf;
}

Eigen::Affine2d SNDTMatch(const SNDTMap &target_map, const SNDTMap &source_map,
                          SNDTParameters &params,
                          const Eigen::Affine2d &guess_tf) {
  params.InitializeOutput();
  auto kd = MakeKDTree(target_map.GetPoints());
  auto cur_tf = guess_tf;
  Eigen::Affine2d best = cur_tf;

  while (params._converge == Converge::kNotConverge) {
    ++params._iteration;
    double x = 0, y = 0, t = 0;
    auto next_map = source_map.PseudoTransformCells(cur_tf);
    ceres::Problem problem;

    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      // XXX: direct get?
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      ceres::LossFunction *loss = nullptr;
      if (params.huber != 0)
        loss = new ceres::HuberLoss(params.huber);
      problem.AddResidualBlock(SNDTCostFunctor::Create(cellp.get(), cellq), loss, &x, &y, &t);
    }

    ceres::Solver::Options options;
    if (params.verbose) options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = params.solver;
    options.max_num_iterations = params.max_iterations;

    ceres::Solver::Summary summary;
    auto t1 = GetTime();
    ceres::Solve(options, &problem, &summary);
    auto t2 = GetTime();
    params._usedtime.optimize += GetDiffTime(t1, t2);

    cur_tf = Eigen::Translation2d(x, y) * Eigen::Rotation2Dd(t) * cur_tf;

    // XXX: minimum cost?
    if (summary.final_cost > params._min_cost) {
      ++params._min_ctr;
      if (params._min_ctr == 5)
        params._converge = Converge::kConsecutiveMinimum;
    } else {
      params._min_cost = summary.final_cost;
      params._min_ctr = 0;
      best = cur_tf;
    }

    // if (Eigen::Vector2d(x, y).norm() < params.threshold)
    //   params._converge = Converge::kThreshold;
    if (params._iteration > params.max_iterations &&
        params._converge != Converge::kMaxIterations)
      params._converge = Converge::kMaxIterations;
  }
  return best;
  // return cur_tf;
}
