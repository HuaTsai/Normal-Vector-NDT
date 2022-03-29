#include <bits/stdc++.h>
#include <common/common.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Int32.h>
using namespace std;

vector<sensor_msgs::PointCloud2> vpcs;
ros::Publisher pub;

void cb(const std_msgs::Int32 &num) {
  int n = num.data;
  if (n < 0 || n > (int)vpcs.size() - 1) return;
  pub.publish(vpcs[n]);
}

int main(int argc, char **argv) {
  SerializationInput(JoinPath(APATH, "1Data/log24/lidar.ser"), vpcs);
  for (auto &pcs : vpcs) {
    pcl::PointCloud<pcl::PointXYZ> pc, pc2;
    pcl::fromROSMsg(pcs, pc);
    for (auto pt : pc) {
      if (pt.z < -1) continue;
      pc2.push_back(pt);
    }
    pcl::toROSMsg(pc2, pcs);
    pcs.header.frame_id = "map";
  }
  ros::init(argc, argv, "seelidar");
  ros::NodeHandle nh;
  pub = nh.advertise<sensor_msgs::PointCloud2>("pc", 0, true);
  ros::Subscriber sub = nh.subscribe("idx", 0, cb);
  ros::spin();
}
