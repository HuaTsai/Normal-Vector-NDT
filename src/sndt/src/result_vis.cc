#include <bits/stdc++.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>
#include <std_msgs/Int64.h>
// #include "sndt/ndt_conversions.hpp"
// #include "sndt/ndt_matcher_2d.hpp"
#include "common/common.h"

using namespace std;

common::MatchResults mrs;
ros::Publisher pub_pc;
ros::Publisher pub_pc2;
ros::Publisher pub_pc3;

pcl::PointCloud<pcl::PointXYZ> PCLFromMatrixXd(const Eigen::MatrixXd &mtx) {
  pcl::PointCloud<pcl::PointXYZ> ret;
  for (int i = 0; i < mtx.cols(); ++i) {
    ret.push_back(pcl::PointXYZ(mtx(0, i), mtx(1, i), mtx(2, i)));
  }
  ret.header.frame_id = "map";
  return ret;
}

void subcb(const std_msgs::Int64::ConstPtr &msg) {
  // rostopic pub /subcb std_msgs/Int64 <idx>
  auto results = mrs.results();
  if (msg->data < 0 || msg->data >= (int)results.size()) {
    ROS_WARN("Invalid index %ld", msg->data);
  } else {
    pub_pc.publish(PCLFromMatrixXd(mrs.source()));
    pub_pc2.publish(PCLFromMatrixXd(mrs.target()));
    auto output = PCLFromMatrixXd(mrs.source());
    Eigen::Vector3d rxyt = results.col(msg->data);
    Eigen::Vector3d axyt = Eigen::Vector3d::Map(mrs.actual().data(), 3);
    auto mtx = common::Matrix4fFromMatrix3d(common::Matrix3dFromXYTDegree({rxyt(0), rxyt(1), rxyt(2)}));
    pcl::transformPointCloud(output, output, mtx);
    pub_pc3.publish(output);
    double r = (rxyt - axyt).block<2, 1>(0, 0).norm();
    double a = abs((rxyt - axyt)(2));
    ROS_INFO("Show index %ld: %f, %f", msg->data, r, a);
  }
}

int main(int argc, char **argv) {
  common::SerializeIn(argv[1], mrs);
  ROS_INFO("Reading MRS ... Done!");
  ROS_INFO("There are %ld matching results!", mrs.results().cols());

  ros::init(argc, argv, "result_vis");
  ros::NodeHandle nh;
  pub_pc = nh.advertise<sensor_msgs::PointCloud2>("pc", 0, true);
  pub_pc2 = nh.advertise<sensor_msgs::PointCloud2>("pc2", 0, true);
  pub_pc3 = nh.advertise<sensor_msgs::PointCloud2>("pc3", 0, true);
  ros::Publisher pub_normal = nh.advertise<visualization_msgs::MarkerArray>("normal", 0, true);
  ros::Publisher pub_normal2 = nh.advertise<visualization_msgs::MarkerArray>("normal2", 0, true);
  ros::Subscriber sub = nh.subscribe<std_msgs::Int64>("idx", 0, subcb);

  ros::spin();
}
