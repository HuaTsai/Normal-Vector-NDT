/**
 * @file wrapper.hpp
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief 
 * @version 0.1
 * @date 2021-07-30
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#pragma once

#include <sndt/ndt_map.h>
#include <sndt/pcl_utils.h>

NDTMap MakeNDTMap(
    const std::vector<std::pair<Eigen::MatrixXd, Eigen::Affine2d>> &data,
    const Eigen::Vector2d &intrinsic, const std::vector<double> &params) {
  double cell_size = params.at(0);

  auto n = std::accumulate(data.begin(), data.end(), 0, [](auto a, auto b) {
    return a + (int)b.first.cols();
  });
  std::vector<Eigen::Vector2d> points(n);
  std::vector<Eigen::Matrix2d> point_covs(n);

  int j = 0;
  for (const auto &elem : data) {
    auto pts = elem.first;
    auto T = elem.second;
    for (int i = 0; i < pts.cols(); ++i) {
      Eigen::Vector2d pt = pts.col(i);
      double r2 = pt.squaredNorm();
      double theta = atan2(pt(1), pt(0));
      Eigen::Matrix2d J = Eigen::Rotation2Dd(theta).matrix();
      Eigen::Matrix2d S = Eigen::Vector2d(intrinsic(0), r2 * intrinsic(1)).asDiagonal();
      points[j] = T * pt;
      point_covs[j] = T.rotation() * J * S * J.transpose() * T.rotation().transpose();
      ++j;
    }
  }
  NDTMap ret(cell_size);
  ret.LoadPointsWithCovariances(points, point_covs);
  return ret;
}

/**
 * @param data MatrixXd 2xN & Affine2d
 * @param point_intrinsic variance (r_sig^2, theta_sig^2)
 * @param params (cell_size, radius)
 */
SNDTMap MakeSNDTMap(
    const std::vector<std::pair<Eigen::MatrixXd, Eigen::Affine2d>> &data,
    const Eigen::Vector2d &intrinsic, const std::vector<double> &params) {
  double cell_size = params.at(0);
  double radius = params.at(1);

  auto n = std::accumulate(data.begin(), data.end(), 0, [](auto a, auto b) {
    return a + (int)b.first.cols();
  });
  std::vector<Eigen::Vector2d> points(n);
  std::vector<Eigen::Matrix2d> point_covs(n);

  int j = 0;
  for (const auto &elem : data) {
    auto pts = elem.first;
    auto T = elem.second;
    for (int i = 0; i < pts.cols(); ++i) {
      Eigen::Vector2d pt = pts.col(i);
      double r2 = pt.squaredNorm();
      double theta = atan2(pt(1), pt(0));
      Eigen::Matrix2d J = Eigen::Rotation2Dd(theta).matrix();
      Eigen::Matrix2d S = Eigen::Vector2d(intrinsic(0), r2 * intrinsic(1)).asDiagonal();
      points[j] = T * pt;
      point_covs[j] = T.rotation() * J * S * J.transpose() * T.rotation().transpose();
      ++j;
    }
  }
  auto normals = ComputeNormals(points, radius);
  SNDTMap ret(cell_size);
  ret.LoadPointsWithCovariancesAndNormals(points, point_covs, normals);
  return ret;
}
