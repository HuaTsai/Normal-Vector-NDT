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
#include <common/common.h>
#include <nav_msgs/Path.h>
#include <normal2d/normal2d.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>

void MakeGtLocal(nav_msgs::Path &path, const ros::Time &start) {
  auto startpose = GetPose(path.poses, start);
  Eigen::Affine3d preT;
  tf2::fromMsg(startpose, preT);
  preT = preT.inverse();
  for (size_t i = 0; i < path.poses.size(); ++i) {
    Eigen::Affine3d T;
    tf2::fromMsg(path.poses[i].pose, T);
    Eigen::Affine3d newT = preT * T;
    path.poses[i].pose = tf2::toMsg(newT);
  }
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
