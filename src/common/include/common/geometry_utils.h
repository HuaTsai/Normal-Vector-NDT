#pragma once
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <gsl/gsl>

geometry_msgs::Pose PoseInterpolate(const geometry_msgs::PoseStamped &p1,
                                    const geometry_msgs::PoseStamped &p2,
                                    const ros::Time &time) {
  Expects(time >= p1.header.stamp && time <= p2.header.stamp);
  geometry_msgs::Pose ret;
  double num = (time - p1.header.stamp).toSec();
  double den = (p2.header.stamp - p1.header.stamp).toSec();
  double ration = num / den;
  tf2::Vector3 v1, v2;
  tf2::Quaternion q1, q2;
  tf2::fromMsg(p1.pose.position, v1);
  tf2::fromMsg(p2.pose.position, v2);
  tf2::fromMsg(p1.pose.orientation, q1);
  tf2::fromMsg(p2.pose.orientation, q2);
  tf2::toMsg(v1.lerp(v2, ration), ret.position);
  tf2::convert(q1.slerp(q2, ration), ret.orientation);
  return ret;
}

geometry_msgs::Pose GetPose(
    const std::vector<geometry_msgs::PoseStamped> &poses,
    const ros::Time &time) {
  if (poses.front().header.stamp == time) return poses.front().pose;
  if (poses.back().header.stamp == time) return poses.back().pose;
  auto end =
      std::lower_bound(poses.begin(), poses.end(), time,
                       [](const geometry_msgs::PoseStamped &a,
                          const ros::Time &b) { return a.header.stamp < b; });
  Ensures(end != poses.end());
  const auto start = std::prev(end);
  return PoseInterpolate(*start, *end, time);
}
