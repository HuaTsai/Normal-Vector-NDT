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
#include <sndt/pcl_utils.h>
#include <sndt/matcher.h>

// start, ..., end, end+1
// <<------- T -------->>
std::vector<std::pair<std::vector<Eigen::Vector2d>, Eigen::Affine2d>> Augment(
    const std::vector<common::EgoPointClouds> &vepcs, int start, int end,
    Eigen::Affine2d &T, std::vector<Eigen::Affine2d> &allT) {
  std::vector<std::pair<std::vector<Eigen::Vector2d>, Eigen::Affine2d>> ret;
  double dx = 0, dy = 0, dth = 0;
  for (int i = start; i <= end; ++i) {
    Eigen::Affine2d T0i = Eigen::Rotation2Dd(dth) * Eigen::Translation2d(dx, dy);
    allT.push_back(T0i);
    for (const auto &pc : vepcs[i].pcs) {
      std::vector<Eigen::Vector2d> pts;
      for (size_t i = 0; i < pc.points.size(); ++i)
        pts.push_back(Eigen::Vector2d(pc.points[i].x, pc.points[i].y));
      Eigen::Affine3d aff;
      tf2::fromMsg(pc.origin, aff);
      Eigen::Matrix3d mtx = Eigen::Matrix3d::Identity();
      mtx.block<2, 2>(0, 0) = aff.matrix().block<2, 2>(0, 0);
      mtx.block<2, 1>(0, 2) = aff.matrix().block<2, 1>(0, 3);
      ret.push_back({pts, T0i * Eigen::Affine2d(mtx)});
    }
    double dt = (vepcs[i + 1].stamp - vepcs[i].stamp).toSec();
    dx += vepcs[i].vxyt[0] * dt;
    dy += vepcs[i].vxyt[1] * dt;
    dth += vepcs[i].vxyt[2] * dt;
  }
  T = Eigen::Rotation2Dd(dth) * Eigen::Translation2d(dx, dy);
  return ret;
}

// start, ..., end, end+1
// <<------- T -------->>
std::vector<Eigen::Vector2d> AugmentPoints(
    const std::vector<common::EgoPointClouds> &vepcs, int start, int end,
    Eigen::Affine2d &T, std::vector<Eigen::Affine2d> &allT) {
  std::vector<Eigen::Vector2d> ret;
  double dx = 0, dy = 0, dth = 0;
  for (int i = start; i <= end; ++i) {
    Eigen::Affine2d T0i = Eigen::Rotation2Dd(dth) * Eigen::Translation2d(dx, dy);
    allT.push_back(T0i);
    for (const auto &pc : vepcs[i].pcs) {
      Eigen::Affine3d aff3;
      tf2::fromMsg(pc.origin, aff3);
      Eigen::Matrix3d mtx = Eigen::Matrix3d::Identity();
      mtx.block<2, 2>(0, 0) = aff3.matrix().block<2, 2>(0, 0);
      mtx.block<2, 1>(0, 2) = aff3.matrix().block<2, 1>(0, 3);
      Eigen::Affine2d aff2(mtx);
      for (size_t i = 0; i < pc.points.size(); ++i) {
        auto pt = Eigen::Vector2d(pc.points[i].x, pc.points[i].y);
        ret.push_back(T0i * aff2 * pt);
      }
    }
    double dt = (vepcs[i + 1].stamp - vepcs[i].stamp).toSec();
    dx += vepcs[i].vxyt[0] * dt;
    dy += vepcs[i].vxyt[1] * dt;
    dth += vepcs[i].vxyt[2] * dt;
  }
  T = Eigen::Rotation2Dd(dth) * Eigen::Translation2d(dx, dy);
  return ret;
}

NDTMap MakeNDTMap(
    const std::vector<std::pair<std::vector<Eigen::Vector2d>, Eigen::Affine2d>> &data,
    const NDTD2DParameters &params) {
  double cell_size = params.cell_size;
  double rvar = params.r_variance;
  double tvar = params.t_variance;

  std::vector<Eigen::Vector2d> points;
  std::vector<Eigen::Matrix2d> point_covs;

  for (const auto &elem : data) {
    auto pts = elem.first;
    auto T = elem.second;
    for (const auto &pt : pts) {
      double r2 = pt.squaredNorm();
      double theta = atan2(pt(1), pt(0));
      Eigen::Matrix2d J = Eigen::Rotation2Dd(theta).matrix();
      Eigen::Matrix2d S = Eigen::Vector2d(rvar, r2 * tvar).asDiagonal();
      points.push_back(T * pt);
      point_covs.push_back(T.rotation() * J * S * J.transpose() * T.rotation().transpose());
    }
  }
  NDTMap ret(cell_size);
  ret.LoadPointsWithCovariances(points, point_covs);
  return ret;
}

SNDTMap MakeSNDTMap(
    const std::vector<std::pair<std::vector<Eigen::Vector2d>, Eigen::Affine2d>> &data,
    const SNDTParameters &params) {
  double cell_size = params.cell_size;
  double radius = params.radius;
  double rvar = params.r_variance;
  double tvar = params.t_variance;

  auto n = std::accumulate(data.begin(), data.end(), 0, [](auto a, auto b) {
    return a + (int)b.first.size();
  });
  std::vector<Eigen::Vector2d> points(n);
  std::vector<Eigen::Matrix2d> point_covs(n);

  int j = 0;
  for (const auto &elem : data) {
    auto pts = elem.first;
    auto T = elem.second;
    for (const auto &pt : pts) {
      double r2 = pt.squaredNorm();
      double theta = atan2(pt(1), pt(0));
      Eigen::Matrix2d J = Eigen::Rotation2Dd(theta).matrix();
      Eigen::Matrix2d S = Eigen::Vector2d(rvar, r2 * tvar).asDiagonal();
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
