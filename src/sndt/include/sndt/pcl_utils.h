/**
 * @file pcl_utils.h
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief PCL Utilities
 * @version 0.1
 * @date 2021-07-30
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <normal2d/normal2d.h>
#include <pcl/search/kdtree.h>

pcl::KdTreeFLANN<pcl::PointXY> MakeKDTree(const std::vector<Eigen::Vector2d> &points) {
  pcl::PointCloud<pcl::PointXY>::Ptr pc(new pcl::PointCloud<pcl::PointXY>);
  for (const auto &pt : points) {
    pcl::PointXY p;
    p.x = pt(0), p.y = pt(1);
    pc->push_back(p);
  }
  pcl::KdTreeFLANN<pcl::PointXY> ret;
  ret.setInputCloud(pc);
  return ret;
}

std::vector<Eigen::Vector2d> ComputeNormals(
    const std::vector<Eigen::Vector2d> &pc, double radius) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pclpc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr normals(new pcl::PointCloud<pcl::PointXYZ>);
  int n = pc.size();
  for (int i = 0; i < n; ++i)
    pclpc->push_back(pcl::PointXYZ(pc[i](0), pc[i](1), 0));
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  Normal2dEstimation ne;
  ne.setInputCloud(pclpc);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(radius);
  ne.compute(normals);
  std::vector<Eigen::Vector2d> ret;
  for (int i = 0; i < n; ++i)
    ret.push_back(Eigen::Vector2d((*normals)[i].x, (*normals)[i].y));
  return ret;
}

// TODO: implement ComputeNormals
