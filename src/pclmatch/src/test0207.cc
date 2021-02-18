#include <bits/stdc++.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "normal2d/Normal2dEstimation.hpp"

#include "common/common.h"

using namespace std;

void ComputeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr pc, double radius,
                    pcl::PointCloud<pcl::Normal>::Ptr output) {
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  Normal2dEstimation ne;
  ne.setInputCloud(pc);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(radius);
  ne.compute(output);
}

void ComputeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr pc, double radius,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr output) {
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  Normal2dEstimation ne;
  ne.setInputCloud(pc);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(radius);
  ne.compute(output);
}

void ComputeNormals3D(pcl::PointCloud<pcl::PointXYZ>::Ptr pc, double radius,
                      pcl::PointCloud<pcl::Normal>::Ptr output) {
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(pc);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(1);
  ne.compute(*output);
}

int main() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile(APATH(20210128/cases/spc00.pcd), *pc);

  int sizes = 10;
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(pc);
  pass.setFilterFieldName("y");
  pass.setFilterLimits(-10, 10);
  printf("points: %ld -> ", pc->points.size());
  pass.filter(*pc);
  printf("%ld\n", pc->points.size());
  for (int i = 0; i < sizes; ++i) {
    auto pt = pc->points.at(i);
    printf("  %.2f, %.2f, %.2f\n", pt.x, pt.y, pt.z);
  }

  pcl::PointCloud<pcl::Normal>::Ptr normals3d(new pcl::PointCloud<pcl::Normal>);
  ComputeNormals3D(pc, 1, normals3d);
  cout << "normals3d: " << normals3d->points.size() << endl;
  for (int i = 0; i < sizes; ++i) {
    auto pt = normals3d->points.at(i);
    printf("  %.2f, %.2f, %.2f, %.2f\n", pt.normal_x, pt.normal_y, pt.normal_z, pt.curvature);
  }

  pcl::PointCloud<pcl::Normal>::Ptr normals2d(new pcl::PointCloud<pcl::Normal>);
  ComputeNormals(pc, 1, normals2d);
  cout << "normals2d: " << normals2d->points.size() << endl;
  for (int i = 0; i < sizes; ++i) {
    auto pt = normals2d->points.at(i);
    printf("  %.2f, %.2f, %.2f, %.2f\n", pt.normal_x, pt.normal_y, pt.normal_z, pt.curvature);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr normals2d2(new pcl::PointCloud<pcl::PointXYZ>);
  ComputeNormals(pc, 1, normals2d2);
  cout << "normals2d: " << normals2d2->points.size() << endl;
  for (int i = 0; i < sizes; ++i) {
    auto pt = normals2d2->points.at(i);
    printf("  %.2f, %.2f, %.2f\n", pt.x, pt.y, pt.z);
  }

  pcl::PointCloud<pcl::PointNormal> pcout;
  pcl::concatenateFields(*pc, *normals2d, pcout);
  cout << "pcout: " << pcout.points.size() << endl;
  for (int i = 0; i < sizes; ++i) {
    auto pt = pcout.points.at(i);
    printf("  %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n", pt.x, pt.y, pt.z,
           pt.normal_x, pt.normal_y, pt.normal_z, pt.curvature);
  }
}