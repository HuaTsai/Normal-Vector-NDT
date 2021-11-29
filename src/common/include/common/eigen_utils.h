#pragma once
#include <angles/angles.h>
#include <std_msgs/Float64MultiArray.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <Eigen/Dense>

// Affine3d
Eigen::Affine3d Affine3dFromXYZRPY(const Eigen::Vector3d &xyz,
                                   const Eigen::Vector3d &rpy) {
  Eigen::Affine3d ret = Eigen::Translation3d(xyz(0), xyz(1), xyz(2)) *
                        Eigen::AngleAxisd(rpy(2), Eigen::Vector3d::UnitZ()) *
                        Eigen::AngleAxisd(rpy(1), Eigen::Vector3d::UnitY()) *
                        Eigen::AngleAxisd(rpy(0), Eigen::Vector3d::UnitX());
  return ret;
}

Eigen::Affine3d Affine3dFromMatrix4f(const Eigen::Matrix4f &mtx) {
  return Eigen::Affine3d(mtx.cast<double>());
}

// XYZRPY, XYTRadian, XYTDegree
Eigen::Matrix<double, 6, 1> XYZRPYFromAffine3d(const Eigen::Affine3d &mtx) {
  Eigen::Matrix<double, 6, 1> ret;
  ret.block<3, 1>(0, 0) = mtx.translation();
  Eigen::Vector3d ypr = mtx.rotation().eulerAngles(2, 1, 0);
  double yaw = angles::normalize_angle(ypr(0));
  double pitch = angles::normalize_angle(ypr(1));
  double roll = angles::normalize_angle(ypr(2));
  if (fabs(pitch) > M_PI / 2) {
    roll = angles::normalize_angle(roll + M_PI);
    pitch = angles::normalize_angle(-pitch + M_PI);
    yaw = angles::normalize_angle(yaw + M_PI);
  }
  ret.block<3, 1>(3, 0) = Eigen::Vector3d(roll, pitch, yaw);
  return ret;
}

Eigen::Matrix<double, 6, 1> XYZRPYFromMatrix4f(const Eigen::Matrix4f &mtx) {
  return XYZRPYFromAffine3d(Affine3dFromMatrix4f(mtx));
}

Eigen::Vector3d XYTRadianFromAffine2d(const Eigen::Affine2d &aff) {
  Eigen::Vector3d ret;
  ret(0) = aff.translation()(0);
  ret(1) = aff.translation()(1);
  ret(2) = Eigen::Rotation2Dd(aff.rotation()).angle();
  return ret;
}

Eigen::Vector3d XYTDegreeFromAffine2d(const Eigen::Affine2d &aff) {
  Eigen::Vector3d ret = XYTRadianFromAffine2d(aff);
  ret(2) = ret(2) * 180. / M_PI;
  return ret;
}

Eigen::Affine3d Affine3dFromAffine2d(const Eigen::Affine2d &aff) {
  auto xyt = XYTRadianFromAffine2d(aff);
  return Eigen::Translation3d(xyt(0), xyt(1), 0) *
         Eigen::AngleAxisd(xyt(2), Eigen::Vector3d::UnitZ());
}

Eigen::Matrix4f Matrix4fFromMatrix3d(const Eigen::Matrix3d &mtx) {
  Eigen::Matrix4f ret = Eigen::Matrix4f::Identity();
  ret.block<2, 2>(0, 0) = mtx.block<2, 2>(0, 0).cast<float>();
  ret.block<2, 1>(0, 3) = mtx.block<2, 1>(0, 2).cast<float>();
  return ret;
}

geometry_msgs::TransformStamped TstFromAffine3d(
    const Eigen::Affine3d &T,
    const ros::Time &stamp,
    const std::string &frame_id,
    const std::string &child_frame_id) {
  geometry_msgs::TransformStamped ret = tf2::eigenToTransform(T);
  ret.header.frame_id = frame_id;
  ret.header.stamp = stamp;
  ret.child_frame_id = child_frame_id;
  return ret;
}

geometry_msgs::TransformStamped TstFromMatrix4f(
    const Eigen::Matrix4f &T,
    const ros::Time &stamp,
    const std::string &frame_id,
    const std::string &child_frame_id) {
  return TstFromAffine3d(Affine3dFromMatrix4f(T), stamp, frame_id,
                         child_frame_id);
}

Eigen::Affine3d Conserve2DFromAffine3d(const Eigen::Affine3d &T) {
  auto xyzrpy = XYZRPYFromAffine3d(T);
  auto xyz = xyzrpy.block<3, 1>(0, 0);
  auto rpy = xyzrpy.block<3, 1>(3, 0);
  xyz(2) = rpy(0) = rpy(1) = 0;
  return Affine3dFromXYZRPY(xyz, rpy);
}

Eigen::Vector2d TransNormRotDegAbsFromAffine2d(const Eigen::Affine2d &aff) {
  Eigen::Vector2d ret;
  ret(0) = aff.translation().norm();
  ret(1) = abs(Eigen::Rotation2Dd(aff.rotation()).angle() * 180 / M_PI);
  return ret;
}

geometry_msgs::PoseStamped MakePoseStampedMsg(const ros::Time &time,
                                              const Eigen::Affine3d &aff) {
  geometry_msgs::PoseStamped ret;
  ret.header.frame_id = "map";
  ret.header.stamp = time;
  ret.pose = tf2::toMsg(aff);
  return ret;
}

geometry_msgs::PoseStamped MakePoseStampedMsg(const ros::Time &time,
                                              const Eigen::Matrix4f &mtx) {
  return MakePoseStampedMsg(time, Affine3dFromMatrix4f(mtx));
}

std::vector<Eigen::Vector2d> TransformPoints(
    const std::vector<Eigen::Vector2d> &points, const Eigen::Affine2d &aff) {
  std::vector<Eigen::Vector2d> ret(points.size());
  std::transform(points.begin(), points.end(), ret.begin(),
                 [&aff](auto p) { return aff * p; });
  return ret;
}

void TransformPointsInPlace(std::vector<Eigen::Vector2d> &points,
                            const Eigen::Affine2d &aff) {
  std::transform(points.begin(), points.end(), points.begin(),
                 [&aff](auto p) { return aff * p; });
}

std::vector<Eigen::Vector2d> TransformNormals(
    const std::vector<Eigen::Vector2d> &normals, const Eigen::Affine2d &aff) {
  std::vector<Eigen::Vector2d> ret(normals.size());
  std::transform(normals.begin(), normals.end(), ret.begin(), [&aff](auto p) {
    return Eigen::Vector2d(aff.rotation() * p);
  });
  return ret;
}
