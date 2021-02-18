#pragma once

#include <angles/angles.h>
#include <std_msgs/Float64MultiArray.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <Eigen/Dense>
#include <gsl/gsl>

namespace common {
Eigen::Affine3d Affine3dFromMatrix4f(const Eigen::Matrix4f &mtx) {
  Eigen::Affine3d ret(mtx.cast<double>());
  return ret;
}

Eigen::Affine3d Affine3dFromXYZRPY(const std::vector<double> &xyzrpy) {
  Expects(xyzrpy.size() == 6);
  Eigen::Affine3d ret =
      Eigen::Translation3d(xyzrpy.at(0), xyzrpy.at(1), xyzrpy.at(2)) *
      Eigen::AngleAxisd(xyzrpy.at(5), Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(xyzrpy.at(4), Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(xyzrpy.at(3), Eigen::Vector3d::UnitX());
  return ret;
}

std::vector<double> XYZRPYFromAffine3d(const Eigen::Affine3d &mtx) {
  std::vector<double> ret;
  ret.push_back(mtx.translation()(0));
  ret.push_back(mtx.translation()(1));
  ret.push_back(mtx.translation()(2));
  double roll = angles::normalize_angle(mtx.rotation().eulerAngles(2, 1, 0)(2));
  double pitch = angles::normalize_angle(mtx.rotation().eulerAngles(2, 1, 0)(1));
  double yaw = angles::normalize_angle(mtx.rotation().eulerAngles(2, 1, 0)(0));
  if (fabs(pitch) > M_PI / 2) {
    roll = angles::normalize_angle(roll + M_PI);
    pitch = angles::normalize_angle(-pitch + M_PI);
    yaw = angles::normalize_angle(yaw + M_PI);
  }
  ret.push_back(roll);
  ret.push_back(pitch);
  ret.push_back(yaw);
  return ret;
}

std::vector<double> XYZRPYFromMatrix4f(const Eigen::Matrix4f &mtx) {
  return XYZRPYFromAffine3d(Affine3dFromMatrix4f(mtx));
}

Eigen::Matrix3d Matrix3dFromXYTRadian(const std::vector<double> &xyt) {
  Expects(xyt.size() == 3);
  Eigen::Affine2d aff = Eigen::Translation2d(xyt.at(0), xyt.at(1)) *
                        Eigen::Rotation2Dd(xyt.at(2));
  return aff.matrix();
}

Eigen::Matrix3d Matrix3dFromXYTDegree(const std::vector<double> &xyt) {
  return Matrix3dFromXYTRadian({xyt.at(0), xyt.at(1), xyt.at(2) * M_PI / 180.});
}

Eigen::Matrix3d Matrix3dFromMatrix4f(const Eigen::Matrix4f &mtx) {
  auto xyzrpy = XYZRPYFromMatrix4f(mtx);
  return Matrix3dFromXYTRadian({xyzrpy.at(0), xyzrpy.at(1), xyzrpy.at(5)});
}

Eigen::Matrix4f Matrix4fFromMatrix3d(const Eigen::Matrix3d &mtx) {
  Eigen::Matrix4f ret = Eigen::Matrix4f::Identity();
  ret.block<2, 2>(0, 0) = mtx.block<2, 2>(0, 0).cast<float>();
  ret.block<2, 1>(0, 3) = mtx.block<2, 1>(0, 2).cast<float>();
  return ret;
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
  return Affine3dFromXYZRPY(
      {xyzrpy.at(0), xyzrpy.at(1), 0, 0, 0, xyzrpy.at(5)});
}

tf2::Transform ToTf2Transform(const geometry_msgs::Pose &p) {
  tf2::Transform ret;
  tf2::fromMsg(p, ret);
  return ret;
}

tf2::Transform ToTf2Transform(const Eigen::Matrix4f &mtx) {
  tf2::Transform ret;
  tf2::fromMsg(tf2::toMsg(Affine3dFromMatrix4f(mtx)), ret);
  return ret;
}

std_msgs::Float64MultiArray StdMsgFromEigenMatrix(
    const Eigen::MatrixXd &mtx) {
  std_msgs::Float64MultiArray ret;
  std_msgs::MultiArrayDimension dim0;
  dim0.label = "row";
  dim0.size = mtx.rows();
  dim0.stride = mtx.size();
  ret.layout.dim.push_back(dim0);
  std_msgs::MultiArrayDimension dim1;
  dim1.label = "column";
  dim1.size = mtx.cols();
  dim1.stride = mtx.cols();
  ret.layout.dim.push_back(dim1);
  ret.layout.data_offset = 0;
  for (int i = 0; i < mtx.rows(); ++i) {
    for (int j = 0; j < mtx.cols(); ++j) {
      ret.data.push_back(mtx(i, j));
    }
  }
  return ret;
}

Eigen::MatrixXd EigenMatrixFromStdMsg(
    const std_msgs::Float64MultiArray &msg) {
  auto dim0 = msg.layout.dim.at(0);
  auto dim1 = msg.layout.dim.at(1);
  Eigen::MatrixXd ret(dim0.size, dim1.size);
  for (size_t i = 0; i < dim0.size; ++i) {
    for (size_t j = 0; j < dim1.size; ++j) {
      ret(i, j) = msg.data.at(dim1.stride * i + j);
    }
  }
  return ret;
}
}  // namespace common
