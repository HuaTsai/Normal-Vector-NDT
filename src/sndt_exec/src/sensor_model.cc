#include <bits/stdc++.h>

#include "common/common.h"
#include "sndt/ndt_visualizations.h"
#include "sndt_exec/wrapper.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
  Vector2d v(0.0625, 0.0001);
  Vector2d pt(2, 1);
  double r2 = pt.squaredNorm();
  double theta = atan2(pt(1), pt(0));
  Matrix2d J = Rotation2Dd(theta).matrix();
  Matrix2d S = Vector2d(v(0), r2 * v(1)).asDiagonal();
  Matrix2d cov = J * S * J.transpose();

  ros::init(argc, argv, "sensor_model");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<visualization_msgs::MarkerArray>("marker", 0, true);
  auto m1 = MarkerOfEclipse(pt, cov);
  auto m2 = MarkerOfPoints(v);
  auto m = JoinMarkers({m1, m2});
  pub.publish(m);
  ros::spin();
}
