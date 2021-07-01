#pragma once

#include <tf2_eigen/tf2_eigen.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>

namespace common {
enum class Color {
  kRed,
  kLime,
  kBlue,
  kWhite,
  kBlack,
  kGray,
  kYellow,
  kAqua,
  kFuchsia
};

std_msgs::ColorRGBA MakeColorRGBA(const Color &color, double alpha = 1.) {
  std_msgs::ColorRGBA ret;
  ret.a = alpha;
  ret.r = ret.g = ret.b = 0.;
  if (color == Color::kRed) {
    ret.r = 1.;
  } else if (color == Color::kLime) {
    ret.g = 1.;
  } else if (color == Color::kBlue) {
    ret.b = 1.;
  } else if (color == Color::kWhite) {
    ret.a = 0.;
  } else if (color == Color::kBlack) {
    // nop
  } else if (color == Color::kGray) {
    ret.r = ret.g = ret.b = 0.5;
  } else if (color == Color::kYellow) {
    ret.r = ret.g = 1.;
  } else if (color == Color::kAqua) {
    ret.g = ret.b = 1.;
  } else if (color == Color::kFuchsia) {
    ret.r = ret.b = 1.;
  }
  return ret;
}

// TODO: Move lots of functions to here from ndt_conversins.hpp

visualization_msgs::Marker MakeArrowMarkerByEnds(int idx,
                                                 const std::string &frame_id,
                                                 const Eigen::Vector3d &start,
                                                 const Eigen::Vector3d &end,
                                                 const Color &color) {
  visualization_msgs::Marker ret;
  ret.header.frame_id = frame_id;
  ret.header.stamp = ros::Time::now();
  ret.id = idx;
  ret.type = visualization_msgs::Marker::ARROW;
  ret.action = visualization_msgs::Marker::ADD;
  ret.points.push_back(tf2::toMsg(start));
  ret.points.push_back(tf2::toMsg(end));
  ret.scale.x = 0.05;
  ret.scale.y = 0.2;
  ret.color = MakeColorRGBA(color);
  ret.pose.orientation.w = 1;
  return ret;
}

visualization_msgs::Marker MakeArrowMarkerByPose(
    int idx, const std::string &frame_id, const geometry_msgs::Pose &pose,
    const Color &color) {
  visualization_msgs::Marker ret;
  ret.header.frame_id = frame_id;
  ret.header.stamp = ros::Time::now();
  ret.id = idx;
  ret.type = visualization_msgs::Marker::ARROW;
  ret.action = visualization_msgs::Marker::ADD;
  ret.pose = pose;
  ret.scale.x = 0.05;
  ret.scale.y = 0.2;
  ret.color = MakeColorRGBA(color);
  return ret;
}

visualization_msgs::Marker MakeCovarianceMarker(int idx,
                                                const std::string &frame_id,
                                                const Eigen::Vector2d &pt,
                                                const Eigen::Matrix2d &cov,
                                                const Color &color) {
  visualization_msgs::Marker ret;
  ret.header.frame_id = frame_id;
  ret.header.stamp = ros::Time::now();
  ret.id = idx;
  ret.type = visualization_msgs::Marker::SPHERE;
  ret.action = visualization_msgs::Marker::ADD;
  ret.pose.position.x = pt(0);
  ret.pose.position.y = pt(1);
  ret.color = MakeColorRGBA(color);
  // COMPTUE ORIENTATION & EVALS
  Eigen::Matrix3d cov_ = Eigen::Matrix3d::Identity();
  cov_.block<2, 2>(0, 0) = cov;
  Eigen::EigenSolver<Eigen::Matrix3d> es(cov_);
  Eigen::Matrix3d m_eigVal = es.pseudoEigenvalueMatrix().cwiseSqrt();
  Eigen::Matrix3d m_eigVec = es.pseudoEigenvectors();
  Eigen::Quaterniond q(m_eigVec);
  ret.scale.x = 3 * m_eigVal(0, 0);
  ret.scale.y = 3 * m_eigVal(1, 1);
  ret.scale.z = 0.1;
  ret.pose.orientation.w = q.w();
  ret.pose.orientation.x = q.x();
  ret.pose.orientation.y = q.y();
  ret.pose.orientation.z = q.z();
  return ret;
}

visualization_msgs::Marker MakeTextMarker(int idx, const std::string &frame_id,
                                          const std::string &text,
                                          const geometry_msgs::Pose &pose,
                                          const Color &color) {
  visualization_msgs::Marker ret;
  ret.header.frame_id = frame_id;
  ret.header.stamp = ros::Time::now();
  ret.id = idx;
  ret.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  ret.action = visualization_msgs::Marker::ADD;
  ret.text = text;
  ret.pose = pose;
  ret.scale.z = 0.7;
  ret.color = MakeColorRGBA(color);
  return ret;
}
}  // namespace common
