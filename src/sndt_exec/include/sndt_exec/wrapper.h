/**
 * @file wrapper.h
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief
 * @version 0.1
 * @date 2021-07-30
 *
 * @copyright Copyright (c) 2021
 *
 */
// XXX: This header file contains implementations. When including it, make sure
// that there is only one compilation unit includes it.
#pragma once
#include <common/EgoPointClouds.h>
#include <common/common.h>
#include <nav_msgs/Path.h>
#include <normal2d/normal2d.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>
#include <sndt/matcher.h>

// start, ..., end, end+1
// <<------- T -------->>
std::vector<std::pair<std::vector<Eigen::Vector2d>, Eigen::Affine2d>> Augment(
    const std::vector<common::EgoPointClouds> &vepcs,
    int start,
    int end,
    Eigen::Affine2d &T,
    std::vector<Eigen::Affine2d> &allT) {
  allT.clear();
  std::vector<std::pair<std::vector<Eigen::Vector2d>, Eigen::Affine2d>> ret;
  double dx = 0, dy = 0, dth = 0;
  for (int i = start; i <= end; ++i) {
    Eigen::Affine2d T0i =
        Eigen::Translation2d(dx, dy) * Eigen::Rotation2Dd(dth);
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
  T = Eigen::Translation2d(dx, dy) * Eigen::Rotation2Dd(dth);
  return ret;
}

// start, ..., end, end+1
// <<------- T -------->>
std::vector<Eigen::Vector2d> AugmentPoints(
    const std::vector<common::EgoPointClouds> &vepcs,
    int start,
    int end,
    Eigen::Affine2d &T,
    std::vector<Eigen::Affine2d> &allT) {
  std::vector<Eigen::Vector2d> ret;
  double dx = 0, dy = 0, dth = 0;
  for (int i = start; i <= end; ++i) {
    Eigen::Affine2d T0i =
        Eigen::Translation2d(dx, dy) * Eigen::Rotation2Dd(dth);
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
  T = Eigen::Translation2d(dx, dy) * Eigen::Rotation2Dd(dth);
  return ret;
}

// TODO: this function does not need any covs, fix the name
NDTMap MakeNDT(
    const std::vector<std::pair<std::vector<Eigen::Vector2d>, Eigen::Affine2d>>
        &data,
    NDTParameters &params) {
  params._usedtime.ProcedureStart(UsedTime::Procedure::kNDT);
  std::vector<Eigen::Vector2d> points;
  for (const auto &[pts, tf] : data)
    for (const auto &pt : pts) points.push_back(tf * pt);
  NDTMap ret(params.cell_size);
  ret.LoadPoints(points);
  params._usedtime.ProcedureFinish();
  return ret;
}

NDTMap MakeNDTMap(
    const std::vector<std::pair<std::vector<Eigen::Vector2d>, Eigen::Affine2d>>
        &data,
    NDTParameters &params) {
  params._usedtime.ProcedureStart(UsedTime::Procedure::kNDT);
  double cell_size = params.cell_size;
  double rvar = params.r_variance;
  double tvar = params.t_variance;

  std::vector<Eigen::Vector2d> points;
  std::vector<Eigen::Matrix2d> point_covs;

  for (const auto &[pts, T] : data) {
    for (const auto &pt : pts) {
      double r2 = pt.squaredNorm();
      double theta = atan2(pt(1), pt(0));
      Eigen::Matrix2d J = Eigen::Rotation2Dd(theta).matrix();
      Eigen::Matrix2d S = Eigen::Vector2d(rvar, r2 * tvar).asDiagonal();
      points.push_back(T * pt);
      point_covs.push_back(T.rotation() * J * S * J.transpose() *
                           T.rotation().transpose());
    }
  }
  NDTMap ret(cell_size);
  ret.LoadPointsWithCovariances(points, point_covs);
  params._usedtime.ProcedureFinish();
  return ret;
}

SNDTMap MakeSNDTMap(
    const std::vector<std::pair<std::vector<Eigen::Vector2d>, Eigen::Affine2d>>
        &data,
    SNDTParameters &params) {
  params._usedtime.ProcedureStart(UsedTime::Procedure::kNDT);
  std::vector<Eigen::Vector2d> points;
  std::vector<Eigen::Matrix2d> point_covs;

  for (const auto &[pts, T] : data) {
    for (const auto &pt : pts) {
      double r2 = pt.squaredNorm();
      double theta = atan2(pt(1), pt(0));
      Eigen::Matrix2d J = Eigen::Rotation2Dd(theta).matrix();
      Eigen::Matrix2d S =
          Eigen::Vector2d(params.r_variance, r2 * params.t_variance)
              .asDiagonal();
      points.push_back(T * pt);
      point_covs.push_back(T.rotation() * J * S * J.transpose() *
                           T.rotation().transpose());
    }
  }
  params._usedtime.ProcedureFinish();

  params._usedtime.ProcedureStart(UsedTime::Procedure::kNormal);
  auto normals = ComputeNormals(points, params.radius);
  params._usedtime.ProcedureFinish();

  params._usedtime.ProcedureStart(UsedTime::Procedure::kNDT);
  SNDTMap ret(params.cell_size);
  ret.LoadPointsWithCovariancesAndNormals(points, point_covs, normals);
  params._usedtime.ProcedureFinish();
  return ret;
}

std::vector<Eigen::Vector2d> MakePoints(
    const std::vector<std::pair<std::vector<Eigen::Vector2d>, Eigen::Affine2d>>
        &data) {
  std::vector<Eigen::Vector2d> ret;
  for (const auto &[pts, aff] : data)
    for (size_t i = 0; i < pts.size(); ++i)
      if (pts[i].allFinite()) ret.push_back(aff * pts[i]);
  return ret;
}

void MakeGtLocal(nav_msgs::Path &path, const ros::Time &start) {
  auto startpose = GetPose(path.poses, start);
  Eigen::Affine3d preT;
  tf2::fromMsg(startpose, preT);
  preT = preT.inverse();
  for (size_t i = 0; i < path.poses.size(); ++i) {
    Eigen::Affine3d T;
    tf2::fromMsg(path.poses[i].pose, T);
    Eigen::Affine3d newT = preT * T;
    newT = Conserve2DFromAffine3d(newT);
    path.poses[i].pose = tf2::toMsg(newT);
  }
}

void WriteToFile(const nav_msgs::Path &path, std::string filename) {
  auto fp = fopen(filename.c_str(), "w");
  fprintf(fp, "# time x y z qx qy qz qw\n");
  for (auto &p : path.poses) {
    auto t = p.header.stamp.toSec();
    auto x = p.pose.position.x;
    auto y = p.pose.position.y;
    auto z = p.pose.position.z;
    auto qx = p.pose.orientation.x;
    auto qy = p.pose.orientation.y;
    auto qz = p.pose.orientation.z;
    auto qw = p.pose.orientation.w;
    fprintf(fp, "%f %f %f %f %f %f %f %f\n", t, x, y, z, qx, qy, qz, qw);
  }
  fclose(fp);
}

std::vector<Eigen::Vector2d> GenerateGaussianSamples(
    const Eigen::Vector2d &mean, const Eigen::Matrix2d &cov, int size) {
  std::vector<Eigen::Vector2d> ret;
  std::mt19937 rng(std::random_device{}());
  std::normal_distribution<> dis;
  auto gau = [&rng, &dis]() { return dis(rng); };
  Eigen::Vector2d evals;
  Eigen::Matrix2d evecs;
  ComputeEvalEvec(cov, evals, evecs);
  Eigen::Matrix2d R = evecs * evals.cwiseSqrt().asDiagonal();
  for (int i = 0; i < size; ++i)
    ret.push_back(R * Eigen::Vector2d::NullaryExpr(gau) + mean);
  return ret;
}

// Voxel: recompute mean in voxel
std::vector<Eigen::Vector2d> PCMsgTo2D(const sensor_msgs::PointCloud2 &msg,
                                       double voxel) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(msg, *pc);
  if (voxel != 0) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(pc);
    vg.setLeafSize(voxel, voxel, voxel);
    vg.filter(*pc);
  }

  std::vector<Eigen::Vector2d> ret;
  for (const auto &pt : *pc)
    if (pcl::isFinite(pt)) ret.push_back(Eigen::Vector2d(pt.x, pt.y));
  return ret;
}

// Voxel: recompute mean in voxel
std::vector<Eigen::Vector3d> PCMsgTo3D(const sensor_msgs::PointCloud2 &msg,
                                       double voxel) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(msg, *pc);
  if (voxel != 0) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(pc);
    vg.setLeafSize(voxel, voxel, voxel);
    vg.filter(*pc);
  }

  std::vector<Eigen::Vector3d> ret;
  for (const auto &pt : *pc)
    if (pcl::isFinite(pt)) ret.push_back(Eigen::Vector3d(pt.x, pt.y, pt.z));
  return ret;
}

// Uniform: nearest point to center in voxel
std::vector<Eigen::Vector2d> PCMsgTo2D2(const sensor_msgs::PointCloud2 &msg,
                                        double voxel) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(msg, *pc);
  if (voxel != 0) {
    pcl::UniformSampling<pcl::PointXYZ> us;
    us.setInputCloud(pc);
    us.setRadiusSearch(voxel);
    us.filter(*pc);
  }

  std::vector<Eigen::Vector2d> ret;
  for (const auto &pt : *pc)
    if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z))
      ret.push_back(Eigen::Vector2d(pt.x, pt.y));
  return ret;
}

// Random: just random
std::vector<Eigen::Vector2d> PCMsgTo2D3(const sensor_msgs::PointCloud2 &msg,
                                        int points) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(msg, *pc);
  if (points != 0) {
    pcl::RandomSample<pcl::PointXYZ> rs;
    rs.setInputCloud(pc);
    rs.setSample(points);
    rs.filter(*pc);
  }

  std::vector<Eigen::Vector2d> ret;
  for (const auto &pt : *pc)
    if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z))
      ret.push_back(Eigen::Vector2d(pt.x, pt.y));
  return ret;
}
