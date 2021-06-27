#pragma once

#include <angles/angles.h>
#include <std_msgs/Float64MultiArray.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <Eigen/Dense>
#include <gsl/gsl>

namespace common {

// Affine3d
Eigen::Affine3d Affine3dFromXYZRPY(const Eigen::Vector3d &xyz,
                                   const Eigen::Vector3d &rpy) {
  Eigen::Affine3d ret =
      Eigen::Translation3d(xyz(0), xyz(1), xyz(2)) *
      Eigen::AngleAxisd(rpy(2), Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(rpy(1), Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(rpy(0), Eigen::Vector3d::UnitX());
  return ret;
}

Eigen::Affine3d Affine3dFromAffine2d(const Eigen::Affine2d &aff) {
  return Eigen::Translation3d(aff.translation().x(), aff.translation().y(), 0) *
         Eigen::AngleAxisd(Eigen::Rotation2Dd(aff.rotation()).angle(), Eigen::Vector3d::UnitZ());
}

Eigen::Affine3d Affine3dFromMatrix3d(const Eigen::Matrix3d &mtx) {
  return Affine3dFromAffine2d(Eigen::Affine2d(mtx));
}

Eigen::Affine3d Affine3dFromMatrix4f(const Eigen::Matrix4f &mtx) {
  Eigen::Affine3d ret(mtx.cast<double>());
  return ret;
}

// XYZRPY, XYTRadian, XYTDegree
Eigen::Matrix<double, 6, 1> XYZRPYFromAffine3d(const Eigen::Affine3d &mtx) {
  Eigen::Matrix<double, 6, 1> ret;
  ret.block<3, 1>(0, 0) = mtx.translation();
  double roll = angles::normalize_angle(mtx.rotation().eulerAngles(2, 1, 0)(2));
  double pitch = angles::normalize_angle(mtx.rotation().eulerAngles(2, 1, 0)(1));
  double yaw = angles::normalize_angle(mtx.rotation().eulerAngles(2, 1, 0)(0));
  if (fabs(pitch) > M_PI / 2) {
    roll = angles::normalize_angle(roll + M_PI);
    pitch = angles::normalize_angle(-pitch + M_PI);
    yaw = angles::normalize_angle(yaw + M_PI);
  }
  ret.block<3, 1>(0, 3) = Eigen::Vector3d(roll, pitch, yaw);
  return ret;
}

Eigen::Matrix<double, 6, 1> XYZRPYFromMatrix4f(const Eigen::Matrix4f &mtx) {
  return XYZRPYFromAffine3d(Affine3dFromMatrix4f(mtx));
}

Eigen::Vector3d XYTRadianFromMatrix3d(const Eigen::Matrix3d &mtx) {
  Eigen::Vector3d ret;
  Eigen::Affine2d aff(mtx);
  ret(0) = aff.translation()(0);
  ret(1) = aff.translation()(1);
  ret(2) = Eigen::Rotation2Dd(aff.rotation()).angle();
  return ret;
}

Eigen::Vector3d XYTDegreeFromMatrix3d(const Eigen::Matrix3d &mtx) {
  Eigen::Vector3d ret = XYTRadianFromMatrix3d(mtx);
  ret(2) = ret(2) * 180. / M_PI;
  return ret;
}

// Matrix3d, Matrix4f
Eigen::Matrix3d Matrix3dFromXYTRadian(const Eigen::Vector3d &xyt) {
  Eigen::Affine2d aff = Eigen::Translation2d(xyt(0), xyt(1)) *
                        Eigen::Rotation2Dd(xyt(2));
  return aff.matrix();
}

Eigen::Matrix3d Matrix3dFromXYTDegree(const Eigen::Vector3d &xyt) {
  return Matrix3dFromXYTRadian(Eigen::Vector3d(xyt(0), xyt(1), xyt(2) * M_PI / 180.));
}

Eigen::Matrix3d Matrix3dFromMatrix4f(const Eigen::Matrix4f &mtx) {
  auto xyzrpy = XYZRPYFromMatrix4f(mtx);
  return Matrix3dFromXYTRadian(Eigen::Vector3d(xyzrpy(0), xyzrpy(1), xyzrpy(5)));
}

Eigen::Matrix4f Matrix4fFromMatrix3d(const Eigen::Matrix3d &mtx) {
  Eigen::Matrix4f ret = Eigen::Matrix4f::Identity();
  ret.block<2, 2>(0, 0) = mtx.block<2, 2>(0, 0).cast<float>();
  ret.block<2, 1>(0, 3) = mtx.block<2, 1>(0, 2).cast<float>();
  return ret;
}

Eigen::Matrix4f Matrix4fFromXYTRadian(const Eigen::Vector3d &xyt) {
  return common::Matrix4fFromMatrix3d(common::Matrix3dFromXYTRadian(xyt));
}

geometry_msgs::TransformStamped TstFromAffine3d(
    const Eigen::Affine3d &T, const ros::Time &stamp,
    const std::string &frame_id, const std::string &child_frame_id) {
  geometry_msgs::TransformStamped ret = tf2::eigenToTransform(T);
  ret.header.frame_id = frame_id;
  ret.header.stamp = stamp;
  ret.child_frame_id = child_frame_id;
  return ret;
}

geometry_msgs::TransformStamped TstFromMatrix4f(
    const Eigen::Matrix4f &T, const ros::Time &stamp,
    const std::string &frame_id, const std::string &child_frame_id) {
  return TstFromAffine3d(Affine3dFromMatrix4f(T), stamp, frame_id,
                         child_frame_id);
}

Eigen::Affine3d Conserve2DFromAffine3d(const Eigen::Affine3d &T) {
  auto xyzrpy = XYZRPYFromAffine3d(T);
  auto xyz = xyzrpy.block<3, 1>(0, 0);
  auto rpy = xyzrpy.block<3, 1>(0, 3);
  xyz(2) = rpy(0) = rpy(1) = 0;
  return Affine3dFromXYZRPY(xyz, rpy);
}

Eigen::Vector2d TransNormRotDegAbsFromMatrix3d(const Eigen::Matrix3d &mtx) {
  Eigen::Vector2d ret;
  auto xyt = common::XYTDegreeFromMatrix3d(mtx);
  ret(0) = xyt.block<2, 1>(0, 0).norm();
  ret(1) = abs(xyt(2));
  return ret;
}
}  // namespace common
