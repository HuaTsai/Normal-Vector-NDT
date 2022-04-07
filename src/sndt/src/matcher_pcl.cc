#include <common/eigen_utils.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <sndt/matcher_pcl.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr MakePointCloud2D(
    const std::vector<Eigen::Vector2d> &points) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr ret(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto pt : points) {
    pcl::PointXYZ p;
    p.x = pt(0), p.y = pt(1), p.z = 0;
    ret->push_back(p);
  }
  return ret;
}

Eigen::Affine2d PCLICP(const std::vector<Eigen::Vector2d> &target_points,
                       const std::vector<Eigen::Vector2d> &source_points,
                       const Eigen::Affine2d &guess_tf) {
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  // pcl::registration::TransformationEstimation2D<pcl::PointXYZ,
  //                                               pcl::PointXYZ>::Ptr
  //     est(new pcl::registration::TransformationEstimation2D<pcl::PointXYZ,
  //                                                           pcl::PointXYZ>);
  // icp.setTransformationEstimation(est);
  icp.setMaximumIterations(500);
  icp.setEuclideanFitnessEpsilon(0.000001);
  icp.setTransformationEpsilon(0.000001);
  auto tgt = MakePointCloud2D(target_points);
  auto src = MakePointCloud2D(source_points);
  pcl::PointCloud<pcl::PointXYZ> out;
  icp.setInputSource(src);
  icp.setInputTarget(tgt);
  icp.align(out, Affine3dFromAffine2d(guess_tf).matrix().cast<float>());
  Eigen::Matrix4d T4 = icp.getFinalTransformation().cast<double>();
  Eigen::Matrix3d T3 = Eigen::Matrix3d::Identity();
  T3.block<2, 2>(0, 0) = T4.block<2, 2>(0, 0);
  T3.block<2, 1>(0, 2) = T4.block<2, 1>(0, 3);
  return Eigen::Affine2d(T3);
}

Eigen::Affine2d PCLNDT(const std::vector<Eigen::Vector2d> &target_points,
                       const std::vector<Eigen::Vector2d> &source_points,
                       const Eigen::Affine2d &guess_tf) {
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  ndt.setResolution(1);
  ndt.setEuclideanFitnessEpsilon(0.000001);
  ndt.setTransformationEpsilon(0.000001);
  ndt.setStepSize(0.1);
  auto tgt = MakePointCloud2D(target_points);
  auto src = MakePointCloud2D(source_points);
  pcl::PointCloud<pcl::PointXYZ> out;
  ndt.setInputSource(src);
  ndt.setInputTarget(tgt);
  ndt.align(out, Affine3dFromAffine2d(guess_tf).matrix().cast<float>());
  Eigen::Matrix4d T4 = ndt.getFinalTransformation().cast<double>();
  Eigen::Matrix3d T3 = Eigen::Matrix3d::Identity();
  T3.block<2, 2>(0, 0) = T4.block<2, 2>(0, 0);
  T3.block<2, 1>(0, 2) = T4.block<2, 1>(0, 3);
  return Eigen::Affine2d(T3);
}
