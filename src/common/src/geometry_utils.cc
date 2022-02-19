#include <common/geometry_utils.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

geometry_msgs::Pose PoseInterpolate(const geometry_msgs::PoseStamped &p1,
                                    const geometry_msgs::PoseStamped &p2,
                                    const ros::Time &time) {
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

// poses: p0, p1, p2, ..., pt1, (?), pt2, ..., pn
// find interpolate here --------|
geometry_msgs::Pose GetPose(
    const std::vector<geometry_msgs::PoseStamped> &poses,
    const ros::Time &time) {
  auto p0 = poses.front();
  auto pn = poses.back();
  if (time < p0.header.stamp || time > pn.header.stamp) {
    std::cerr << __FUNCTION__ << ": invalid interpolation time\n";
    std::exit(1);
  }
  if (time == p0.header.stamp) return p0.pose;
  if (time == pn.header.stamp) return pn.pose;
  auto pt2 =
      std::lower_bound(poses.begin(), poses.end(), time,
                       [](const geometry_msgs::PoseStamped &a,
                          const ros::Time &b) { return a.header.stamp < b; });
  auto pt1 = std::prev(pt2);
  return PoseInterpolate(*pt1, *pt2, time);
}
