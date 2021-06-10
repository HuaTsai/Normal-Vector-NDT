#include <bits/stdc++.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Empty.h>
#include "sndt/EgoPointClouds.h"
#include "common/common.h"

using namespace std;

vector<sensor_msgs::PointCloud2> pcs;

void cb(const sensor_msgs::PointCloud2::ConstPtr &pc) {
  pcs.push_back(*pc);
}

void savecb(const std_msgs::Empty::ConstPtr &msg) {
  // rostopic pub /savepc std_msgs/Empty -1
  common::SerializationOutput("/home/ee904/Desktop/HuaTsai/NormalNDT/Research/pcs.ser", pcs);
  ROS_INFO("Saving... Done");
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "savepc");
  ros::NodeHandle nh;
  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("merge_pc", 1000, cb);
  ros::Subscriber sub_save = nh.subscribe<std_msgs::Empty>("savepc", 0, savecb);
  ros::spin();
}