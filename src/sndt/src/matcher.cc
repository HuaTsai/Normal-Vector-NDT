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
  std::cout << "SICP: ";
  auto tpts = ExcludeNaNInf(target_points);
  auto spts = ExcludeNaNInf(source_points);
  auto kd = MakeKDTree(tpts);
  auto tnms = ComputeNormals(tpts, params.radius);
  auto snms = ComputeNormals(spts, params.radius);
  bool converge = false;
  int iteration = 0;
  auto cur_tf = guess_tf;
  int n = source_points.size();
  int min_ctr = 0;
  double min_cost = std::numeric_limits<double>::max();

  while (!converge) {
    double x = 0, y = 0, t = 0;
    std::vector<Eigen::Vector2d> next_pts(n);
    std::vector<Eigen::Vector2d> next_nms(n);
    std::transform(spts.begin(), spts.end(), next_pts.begin(), [&cur_tf](auto p) { return cur_tf * p; });
    std::transform(snms.begin(), snms.end(), next_nms.begin(), [&cur_tf](auto p) { return cur_tf.rotation() * p; });

    ceres::Problem problem;
    for (int i = 0; i < n; ++i) {
      auto p = next_pts[i];
      auto np = next_nms[i];
      if (!p.allFinite() || !np.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      auto q = tpts[idx];
      auto nq = tnms[idx];
      if (!nq.allFinite()) continue;
      if (params.huber != 0) {
        ceres::LossFunction *loss(new ceres::HuberLoss(params.huber));
        problem.AddResidualBlock(SICPCostFunctor::Create(p, np, q, nq), loss,
                                 &x, &y, &t);
      } else {
        problem.AddResidualBlock(SICPCostFunctor::Create(p, np, q, nq), nullptr,
                                 &x, &y, &t);
      }
    }

    ceres::Solver::Options options;
    if (params.verbose) options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = params.max_iterations;
    ceres::Solver::Summary summary;
    auto t3 = GetTime();
    ceres::Solve(options, &problem, &summary);
    auto t4 = GetTime();
    params.usedtime.optimize += GetDiffTime(t3, t4);
    cur_tf = Eigen::Rotation2Dd(t) * Eigen::Translation2d(x, y) * cur_tf;

    if (params.verbose) {
      std::cout << "Iteration " << iteration << ": " << std::endl;
      std::cout << summary.BriefReport() << std::endl;
    }

    if (min_cost < summary.final_cost) {
      ++min_ctr;
    } else {
      min_ctr = 0;
      min_cost = summary.final_cost;
    }

    converge = (min_ctr == 5) ||
               (Eigen::Vector2d(x, y).norm() < params.threshold) ||
               (iteration > params.max_iterations);
    ++iteration;
  }
  return cur_tf;
}


Eigen::Affine2d NDTD2DMatch(
    const NDTMap &target_map, const NDTMap &source_map,
    NDTD2DParameters &params,
    const Eigen::Affine2d &guess_tf) {
  std::cout << "NDTD2D: ";
  auto kd = MakeKDTree(target_map.GetPoints());
  bool converge = false;
  int iteration = 0;
  auto cur_tf = guess_tf;
  int min_ctr = 0;
  double min_cost = std::numeric_limits<double>::max();

  while (!converge) {
    double x = 0, y = 0, t = 0;
    auto next_map = source_map.PseudoTransformCells(cur_tf);
    ceres::Problem problem;

    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      if (params.huber != 0) {
        ceres::LossFunction *loss(new ceres::HuberLoss(params.huber));
        problem.AddResidualBlock(NDTD2DCostFunctor::Create(cellp.get(), cellq),
                                 loss, &x, &y, &t);
      } else {
        problem.AddResidualBlock(NDTD2DCostFunctor::Create(cellp.get(), cellq),
                                 nullptr, &x, &y, &t);
      }
    }

    ceres::Solver::Options options;
    if (params.verbose) options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = params.max_iterations;
    ceres::Solver::Summary summary;
    auto t1 = GetTime();
    ceres::Solve(options, &problem, &summary);
    auto t2 = GetTime();
    params.usedtime.optimize += GetDiffTime(t1, t2);
    cur_tf = Eigen::Rotation2Dd(t) * Eigen::Translation2d(x, y) * cur_tf;

    if (params.verbose) {
      std::cout << "Iteration " << iteration << ": " << std::endl;
      std::cout << summary.BriefReport() << std::endl;
    }

    if (min_cost < summary.final_cost) {
      ++min_ctr;
    } else {
      min_ctr = 0;
      min_cost = summary.final_cost;
    }
    converge = (min_ctr == 5) ||
               (Eigen::Vector2d(x, y).norm() < params.threshold) ||
               (iteration > params.max_iterations);

    // converge = true;
    ++iteration;
  }
  return cur_tf;
}

Eigen::Affine2d SNDTMatch(const SNDTMap &target_map, const SNDTMap &source_map,
                          SNDTParameters &params,
                          const Eigen::Affine2d &guess_tf) {
  auto kd = MakeKDTree(target_map.GetPoints());
  bool converge = false;
  int iteration = 0;
  auto cur_tf = guess_tf;
  int min_ctr = 0;
  double min_cost = std::numeric_limits<double>::max();

  while (!converge) {
    double x = 0, y = 0, t = 0;
    auto next_map = source_map.PseudoTransformCells(cur_tf);
    ceres::Problem problem;

    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      if (params.huber != 0) {
        ceres::LossFunction *loss(new ceres::HuberLoss(params.huber));
        problem.AddResidualBlock(SNDTCostFunctor::Create(cellp.get(), cellq),
                                 loss, &x, &y, &t);
      } else {
        problem.AddResidualBlock(SNDTCostFunctor::Create(cellp.get(), cellq),
                                 nullptr, &x, &y, &t);
      }
    }

    ceres::Solver::Options options;
    if (params.verbose) options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = params.max_iterations;
    ceres::Solver::Summary summary;
    auto t1 = GetTime();
    ceres::Solve(options, &problem, &summary);
    auto t2 = GetTime();
    params.usedtime.optimize += GetDiffTime(t1, t2);
    cur_tf = Eigen::Rotation2Dd(t) * Eigen::Translation2d(x, y) * cur_tf;

    if (params.verbose) {
      std::cout << "Iteration " << iteration << ": " << std::endl;
      std::cout << summary.BriefReport() << std::endl;
    }

    // XXX: converge strategy
    if (min_cost < summary.final_cost) {
      ++min_ctr;
    } else {
      min_ctr = 0;
      min_cost = summary.final_cost;
    }
    converge = (min_ctr == 5) ||
               (Eigen::Vector2d(x, y).norm() < params.threshold) ||
               (iteration > params.max_iterations);

    // converge = true;
    ++iteration;
  }
  return cur_tf;
}
