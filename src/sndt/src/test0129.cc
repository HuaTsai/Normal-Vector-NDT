#include <bits/stdc++.h>
#include <pcl_ros/point_cloud.h>
#include "common/common.h"
#include <pcl/common/transforms.h>

using namespace std;

int main(int argc, char **argv) {
  pcl::PointCloud<pcl::PointXYZ> pc, pc2;
  pcl::io::loadPCDFile(APATH(20210128/cases/spc00.pcd), pc);
  pcl::io::loadPCDFile(APATH(20210128/cases/tpc00.pcd), pc2);
  auto mtx = common::Matrix3dFromXYTRadian(common::ReadFromFile(APATH(20210128/cases/res00.txt)));
  pcl::transformPointCloud(pc, pc, common::Matrix4fFromMatrix3d(mtx));
  ros::init(argc, argv, "test0129");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("pc", 0, true);
  ros::Publisher pub2 = nh.advertise<sensor_msgs::PointCloud2>("pc2", 0, true);
  pc.header.frame_id = pc2.header.frame_id = "map";
  pub.publish(pc);
  pub2.publish(pc2);
  ros::spin();
}
