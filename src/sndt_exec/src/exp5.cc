// Koide Benchmark
#include <common/other_utils.h>
#include <metric/metric.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/ndt_2d.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <ros/ros.h>
#include <sndt/matcher.h>
#include <sndt_exec/wrapper.h>

#include <iostream>

using namespace pcl;
using namespace pcl::registration;
using namespace std;
using namespace Eigen;

PointCloud<PointXYZ>::Ptr align(
    Registration<PointXYZ, PointXYZ>::Ptr registration,
    const PointCloud<PointXYZ>::Ptr& target_cloud,
    const PointCloud<PointXYZ>::Ptr& source_cloud) {
  registration->setInputTarget(target_cloud);
  registration->setInputSource(source_cloud);
  PointCloud<PointXYZ>::Ptr aligned(new PointCloud<PointXYZ>());

  auto t1 = ros::WallTime::now();
  registration->align(*aligned);
  auto t2 = ros::WallTime::now();
  cout << "single : " << (t2 - t1).toSec() * 1000 << "[msec]" << endl;

  for (int i = 0; i < 10; i++) {
    registration->align(*aligned);
  }
  auto t3 = ros::WallTime::now();
  cout << "10times: " << (t3 - t2).toSec() * 1000 << "[msec]" << endl;
  cout << "fitness: " << registration->getFitnessScore() << endl;
  cout << "result: " << registration->getFinalTransformation() << endl << endl;

  return aligned;
}

void TO2D(PointCloud<PointXYZ>::Ptr pc) {
  for (auto& p : *pc) p.z = 0;
}

vector<Vector2d> TOVV(PointCloud<PointXYZ>::Ptr pc) {
  vector<Vector2d> ret;
  for (auto p : *pc) ret.push_back(Vector2d(p.x, p.y));
  return ret;
}

int main() {
  PointCloud<PointXYZ>::Ptr target_cloud(new PointCloud<PointXYZ>());
  PointCloud<PointXYZ>::Ptr source_cloud(new PointCloud<PointXYZ>());
  io::loadPCDFile(GetDataPath("Koide3/tgt.pcd"), *target_cloud);
  io::loadPCDFile(GetDataPath("Koide3/src.pcd"), *source_cloud);
  vector<double> xxx, yyy;
  for (auto p : *target_cloud) xxx.push_back(p.x), yyy.push_back(p.y);
  cout << Stat(xxx).min << ", " << Stat(xxx).max << endl;
  cout << Stat(yyy).min << ", " << Stat(yyy).max << endl;

  // downsampling
  PointCloud<PointXYZ>::Ptr downsampled(new PointCloud<PointXYZ>());

  VoxelGrid<PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);
  voxelgrid.setInputCloud(target_cloud);
  voxelgrid.filter(*downsampled);
  *target_cloud = *downsampled;

  voxelgrid.setInputCloud(source_cloud);
  voxelgrid.filter(*downsampled);
  source_cloud = downsampled;

  ros::Time::init();

  // benchmark
  cout << "--- GICP ---" << endl;
  GeneralizedIterativeClosestPoint<PointXYZ, PointXYZ>::Ptr gicp(
      new GeneralizedIterativeClosestPoint<PointXYZ, PointXYZ>());
  PointCloud<PointXYZ>::Ptr aligned = align(gicp, target_cloud, source_cloud);

  cout << "--- NDT ---" << endl;
  NormalDistributionsTransform<PointXYZ, PointXYZ>::Ptr ndt(
      new NormalDistributionsTransform<PointXYZ, PointXYZ>());
  ndt->setResolution(1.0);
  aligned = align(ndt, target_cloud, source_cloud);

  // cout << "--- 2D ---" << endl;
  // TO2D(target_cloud);
  // TO2D(source_cloud);

  // cout << "--- GICP ---" << endl;
  // GeneralizedIterativeClosestPoint<PointXYZ, PointXYZ>::Ptr gicp2(
  //     new GeneralizedIterativeClosestPoint<PointXYZ, PointXYZ>());
  // aligned = align(gicp2, target_cloud, source_cloud);

  // cout << "--- ICP ---" << endl;
  // IterativeClosestPoint<PointXYZ, PointXYZ>::Ptr icp2(
  //     new IterativeClosestPoint<PointXYZ, PointXYZ>());
  // aligned = align(icp2, target_cloud, source_cloud);

  // // Postpone: Need compute normal
  // // cout << "--- Point-to-Plane ICP ---" << endl;
  // // IterativeClosestPoint<PointXYZ, PointXYZ>::Ptr icp3(
  // //     new IterativeClosestPoint<PointXYZ, PointXYZ>());
  // // TransformationEstimationPointToPlaneLLS<PointXYZ, PointXYZ>::Ptr
  // trans_lls(
  // //     new TransformationEstimationPointToPlaneLLS<PointXYZ, PointXYZ>);
  // // icp3->setTransformationEstimation(trans_lls);
  // // aligned = align(icp3, target_cloud, source_cloud);

  // cout << "--- NDT ---" << endl;
  // NormalDistributionsTransform<PointXYZ, PointXYZ>::Ptr ndt2(
  //     new NormalDistributionsTransform<PointXYZ, PointXYZ>());
  // ndt2->setResolution(1.0);
  // aligned = align(ndt2, target_cloud, source_cloud);

  // // FIXME: NDT2D not work as we predict, need to know how it works
  // cout << "--- NDT2D ---" << endl;
  // NormalDistributionsTransform2D<PointXYZ, PointXYZ>::Ptr ndt2d(
  //     new NormalDistributionsTransform2D<PointXYZ, PointXYZ>());
  // ndt2d->setGridExtent(Vector2f(50, 100));
  // aligned = align(ndt2d, target_cloud, source_cloud);

  // cout << "--- MY NDT ---" << endl;
  // D2DNDTParameters params5;
  // params5.reject = true;
  // params5.r_variance = params5.t_variance = 0;
  // params5.cell_size = 1;
  // params5.d2 = 0.05;
  // params5._usedtime.Start();
  // vector<pair<vector<Vector2d>, Affine2d>> datat{
  //     {TOVV(target_cloud), Eigen::Affine2d::Identity()}};
  // vector<pair<vector<Vector2d>, Affine2d>> datas{
  //     {TOVV(source_cloud), Eigen::Affine2d::Identity()}};
  // auto tgt5 = MakeNDTMap(datat, params5);
  // auto src5 = MakeNDTMap(datas, params5);
  // auto T5 = D2DNDTMatch(tgt5, src5, params5);
  // cout << params5._usedtime.total() / 1000. << endl;
  // cout << T5.matrix() << endl;
}
