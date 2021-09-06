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
