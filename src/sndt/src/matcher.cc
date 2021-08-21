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
#include <common/other_utils.h>
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

void Optimize(ceres::Problem &problem, CommonParameters &params) {
  ceres::Solver::Options options;
  if (params.verbose) options.minimizer_progress_to_stdout = true;
  options.linear_solver_type = params.solver;
  options.max_num_iterations = params.ceres_max_iterations;
  options.num_threads = params.threads;

  ceres::Solver::Summary summary;
  auto t1 = GetTime();
  ceres::Solve(options, &problem, &summary);
  auto t2 = GetTime();
  if (params.verbose) std::cout << summary.FullReport() << std::endl;
  params._usedtime.optimize += GetDiffTime(t1, t2);
  params._ceres_iteration += summary.num_linear_solves;
  ++params._iteration;
}

void CheckConverge(CommonParameters &params, double x, double y) {
  if (Eigen::Vector2d(x, y).norm() < params.threshold)
    params._converge = Converge::kThreshold;
  if (params._iteration > params.max_iterations &&
      params._converge != Converge::kThreshold)
    params._converge = Converge::kMaxIterations;
}

Eigen::Affine2d SNDTMatch(const SNDTMap &target_map, const SNDTMap &source_map,
                          SNDTParameters &params,
                          const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());
  auto cur_tf = guess_tf;

  while (params._converge == Converge::kNotConverge) {
    double x = 0, y = 0, t = 0;
    auto next_map = source_map.PseudoTransformCells(cur_tf);

    ceres::Problem problem;
    int blks = 0;
    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      ceres::LossFunction *loss = nullptr;
      if (params.huber != 0)
        loss = new ceres::HuberLoss(params.huber);
      problem.AddResidualBlock(SNDTCostFunctor::Create(cellp.get(), cellq), loss, &x, &y, &t);
      ++blks;
    }
    params._corres.push_back(blks);

    Optimize(problem, params);
    cur_tf = Eigen::Translation2d(x, y) * Eigen::Rotation2Dd(t) * cur_tf;
    CheckConverge(params, x, y);
  }
  auto t2 = GetTime();
  params._usedtime.others = GetDiffTime(t1, t2) - params._usedtime.optimize;
  return cur_tf;
}

Eigen::Affine2d NDTD2DMatch(
    const NDTMap &target_map, const NDTMap &source_map,
    NDTD2DParameters &params,
    const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());
  auto cur_tf = guess_tf;

  while (params._converge == Converge::kNotConverge) {
    ++params._iteration;
    double x = 0, y = 0, t = 0;
    auto next_map = source_map.PseudoTransformCells(cur_tf);

    ceres::Problem problem;
    int blks = 0;
    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      ceres::LossFunction *loss = nullptr;
      if (params.huber != 0)
        loss = new ceres::HuberLoss(params.huber);
      problem.AddResidualBlock(NDTD2DCostFunctor::Create(cellp.get(), cellq), loss, &x, &y, &t);
      ++blks;
    }
    params._corres.push_back(blks);

    Optimize(problem, params);
    cur_tf = Eigen::Translation2d(x, y) * Eigen::Rotation2Dd(t) * cur_tf;
    CheckConverge(params, x, y);
  }
  auto t2 = GetTime();
  params._usedtime.others = GetDiffTime(t1, t2) - params._usedtime.optimize;
  return cur_tf;
}

Eigen::Affine2d SICPMatch(
    const std::vector<Eigen::Vector2d> &target_points,
    const std::vector<Eigen::Vector2d> &source_points,
    SICPParameters &params,
    const Eigen::Affine2d &guess_tf) {
  auto t1 = GetTime();
  auto tpts = ExcludeNaNInf(target_points);
  auto spts = ExcludeNaNInf(source_points);
  auto kd = MakeKDTree(tpts);
  auto t2 = GetTime();
  auto tnms = ComputeNormals(tpts, params.radius);
  auto snms = ComputeNormals(spts, params.radius);
  auto t3 = GetTime();
  auto cur_tf = guess_tf;

  while (params._converge == Converge::kNotConverge) {
    double x = 0, y = 0, t = 0;
    std::vector<Eigen::Vector2d> next_pts;
    std::vector<Eigen::Vector2d> next_nms;
    std::transform(spts.begin(), spts.end(), std::back_inserter(next_pts),
                   [&cur_tf](auto p) { return cur_tf * p; });
    // The return type of cur_tf * p is const Eigen::Matrix<double, 2, 1>.
    // However, The return type of cur_tf.rotation() * p is const
    // Eigen::Product<Eigen::Matrix<double, 2, 2>, Eigen::Matrix<double, 2, 1>,
    // 0>, so that we need to use ctor of Eigen::Vector2d to avoid bug.
    std::transform(
        snms.begin(), snms.end(), std::back_inserter(next_nms),
        [&cur_tf](auto p) { return Eigen::Vector2d(cur_tf.rotation() * p); });

    ceres::Problem problem;
    int blks = 0;
    for (size_t i = 0; i < next_pts.size(); ++i) {
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
      ++blks;
    }
    params._corres.push_back(blks);

    Optimize(problem, params);
    cur_tf = Eigen::Translation2d(x, y) * Eigen::Rotation2Dd(t) * cur_tf;
    CheckConverge(params, x, y);
  }
  auto t4 = GetTime();
  params._usedtime.normal = GetDiffTime(t2, t3);
  params._usedtime.others =
      GetDiffTime(t1, t4) - params._usedtime.optimize - params._usedtime.normal;
  return cur_tf;
}
