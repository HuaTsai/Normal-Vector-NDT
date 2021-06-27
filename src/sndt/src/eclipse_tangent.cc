#include <bits/stdc++.h>
#include <visualization_msgs/MarkerArray.h>
#include <ros/ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include "common/common.h"

using namespace std;
using namespace Eigen;
using namespace common;
using namespace Eigen;

visualization_msgs::Marker MakeEclipse(Vector2d rs, Affine2d T) {
  visualization_msgs::Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = ros::Time::now();
  ret.id = 0;
  ret.type = visualization_msgs::Marker::SPHERE;
  ret.action = visualization_msgs::Marker::ADD;
  ret.color = MakeColorRGBA(Color::kLime);
  ret.color.a = 0.7;
  ret.scale.x = rs(0);
  ret.scale.y = rs(1);
  ret.scale.z = 0.1;
  ret.pose = tf2::toMsg(common::Affine3dFromAffine2d(T));
  return ret;
}

vector<Vector2d> FindTangentPoints(visualization_msgs::Marker eclipse, Vector2d point) {
  Affine3d aff3;
  tf2::fromMsg(eclipse.pose, aff3);
  Matrix3d mtx = Matrix3d::Identity();
  mtx.block<2, 2>(0, 0) = aff3.rotation().block<2, 2>(0, 0);
  mtx.block<2, 1>(0, 2) = aff3.translation().block<2, 1>(0, 0);
  Affine2d aff2(mtx);
  auto point2 = aff2.inverse() * point;
  auto rx2 = (eclipse.scale.x / 2) * (eclipse.scale.x / 2);
  auto ry2 = (eclipse.scale.y / 2) * (eclipse.scale.y / 2);
  auto x0 = point2(0), x02 = x0 * x0;
  auto y0 = point2(1), y02 = y0 * y0;
  vector<Vector2d> sols(2);
  if (x02 == rx2) {
    auto msol = (-rx2 * ry2 * ry2 + rx2 * ry2 * y02) / (2 * rx2 * ry2 * x0 * y0);
    sols[0](0) = (msol * rx2 * (msol * x0 - y0)) / (msol * msol * rx2 + ry2);
    sols[0](1) = y0 + msol * (sols[0](0) - x0);
    sols[1](0) = x0;
    sols[1](1) = 0;
  } else {
    auto msol1 = (-x0 * y0 + sqrt(-rx2 * ry2 + rx2 * y02 + ry2 * x02)) / (rx2 - x02);
    sols[0](0) = (msol1 * rx2 * (msol1 * x0 - y0)) / (msol1 * msol1 * rx2 + ry2);
    sols[0](1) = y0 + msol1 * (sols[0](0) - x0);
    auto msol2 = (-x0 * y0 - sqrt(-rx2 * ry2 + rx2 * y02 + ry2 * x02)) / (rx2 - x02);
    sols[1](0) = (msol2 * rx2 * (msol2 * x0 - y0)) / (msol2 * msol2 * rx2 + ry2);
    sols[1](1) = y0 + msol2 * (sols[1](0) - x0);
  }
  for (auto &sol : sols)
    sol = aff2 * sol;
  return sols;
}

visualization_msgs::Marker MakeLines(vector<Vector2d> points, int id) {
  visualization_msgs::Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = ros::Time::now();
  ret.id = id;
  ret.type = visualization_msgs::Marker::LINE_STRIP;
  ret.action = visualization_msgs::Marker::ADD;
  ret.color = MakeColorRGBA(Color::kLime);
  for (auto point : points) {
    geometry_msgs::Point pt;
    pt.x = point(0), pt.y = point(1), pt.z = 0;
    ret.points.push_back(pt);
  }
  ret.scale.x = 0.1;
  ret.pose.orientation.w = 1;
  return ret;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "et");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<visualization_msgs::MarkerArray>("markers", 0, true);
  visualization_msgs::MarkerArray ms;

  Vector2d rxy(atof(argv[1]), atof(argv[2]));
  Affine2d T = Translation2d(atof(argv[3]), atof(argv[4])) * Rotation2Dd(atof(argv[5]) * M_PI / 180.);
  auto eclipse = MakeEclipse(rxy, T);
  ms.markers.push_back(eclipse);
 
  auto point = Vector2d(atof(argv[6]), atof(argv[7]));
  auto pts = FindTangentPoints(eclipse, point);
  ms.markers.push_back(MakeLines({point, pts[1]}, 1));
  ms.markers.push_back(MakeLines({pts[0], point}, 2));

  pub.publish(ms);
  ros::spin();
}
