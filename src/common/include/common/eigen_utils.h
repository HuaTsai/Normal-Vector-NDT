#pragma once
#include <bits/stdc++.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>

#include <Eigen/Dense>

// Affine3d
Eigen::Affine3d Affine3dFromXYZRPY(const Eigen::Vector3d &xyz,
                                   const Eigen::Vector3d &rpy);

Eigen::Affine3d Affine3dFromMatrix4f(const Eigen::Matrix4f &mtx);

// XYZRPY, XYTRadian, XYTDegree
Eigen::Matrix<double, 6, 1> XYZRPYFromAffine3d(const Eigen::Affine3d &mtx);

Eigen::Matrix<double, 6, 1> XYZRPYFromMatrix4f(const Eigen::Matrix4f &mtx);

Eigen::Vector3d XYTRadianFromAffine2d(const Eigen::Affine2d &aff);

Eigen::Vector3d XYTDegreeFromAffine2d(const Eigen::Affine2d &aff);

Eigen::Affine3d Affine3dFromAffine2d(const Eigen::Affine2d &aff);

Eigen::Matrix4f Matrix4fFromMatrix3d(const Eigen::Matrix3d &mtx);

geometry_msgs::TransformStamped TstFromAffine3d(
    const Eigen::Affine3d &T,
    const ros::Time &stamp,
    const std::string &frame_id,
    const std::string &child_frame_id);

geometry_msgs::TransformStamped TstFromMatrix4f(
    const Eigen::Matrix4f &T,
    const ros::Time &stamp,
    const std::string &frame_id,
    const std::string &child_frame_id);

Eigen::Affine3d Conserve2DFromAffine3d(const Eigen::Affine3d &T);

Eigen::Vector2d TransNormRotDegAbsFromAffine2d(const Eigen::Affine2d &aff);

geometry_msgs::PoseStamped MakePoseStampedMsg(const ros::Time &time,
                                              const Eigen::Affine3d &aff);

geometry_msgs::PoseStamped MakePoseStampedMsg(const ros::Time &time,
                                              const Eigen::Matrix4f &mtx);

std::vector<Eigen::Vector2d> TransformPoints(
    const std::vector<Eigen::Vector2d> &points, const Eigen::Affine2d &aff);

void TransformPointsInPlace(std::vector<Eigen::Vector2d> &points,
                            const Eigen::Affine2d &aff);

std::vector<Eigen::Vector2d> TransformNormals(
    const std::vector<Eigen::Vector2d> &normals, const Eigen::Affine2d &aff);

class Mvn {
 public:
  explicit Mvn(const Eigen::Vector2d &mean, const Eigen::Matrix2d &covariance);
  double pdf(const Eigen::Vector2d &x) const;

 private:
  Eigen::Vector2d mean_;
  Eigen::Matrix2d covariance_;
};
