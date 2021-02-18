#include <bits/stdc++.h>
#include <visualization_msgs/MarkerArray.h>
#include "common/common.h"

using namespace std;

visualization_msgs::Marker MakePointsMarker(const std::string &frame_id,
                                            const vector<Eigen::Matrix3d> &tfs,
                                            const common::Color &color) {
  visualization_msgs::Marker ret;
  ret.header.frame_id = frame_id;
  ret.header.stamp = ros::Time::now();
  ret.id = 0;
  ret.type = visualization_msgs::Marker::POINTS;
  ret.action = visualization_msgs::Marker::ADD;
  for (const auto &tf : tfs) {
    geometry_msgs::Point pt;
    pt.x = tf(0, 2);
    pt.y = tf(1, 2);
    ret.points.push_back(pt);
  }
  ret.scale.x = 0.2;
  ret.scale.y = 0.2;
  ret.color = common::MakeColorRGBA(color);
  ret.pose.orientation.w = 1;
  return ret;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "test0128");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<visualization_msgs::Marker>("points", 0, true);
  common::RandomTransformGenerator2D rtg;
  rtg.SetTranslationRadiusBound(5, 7);
  rtg.SetRotationDegreeBound(0, 15);
  pub.publish(MakePointsMarker("map", rtg.Generate(100), common::Color::kAqua));
  ros::spin();
}
