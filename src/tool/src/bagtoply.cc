#include <bits/stdc++.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <pcl/io/ply_io.h>

using namespace std;

int main(int argc, char **argv) {
  ros::init(argc, argv, "bagtoply");
  rosbag::Bag bag;
  bag.open(argv[1]);
  for (rosbag::MessageInstance const m : rosbag::View(bag)) {
    sensor_msgs::PointCloud2::ConstPtr pcmsg = m.instantiate<sensor_msgs::PointCloud2>();
    if (pcmsg != nullptr && m.getTopic() == "nuscenes_lidar") {
      // auto str = to_string(pcmsg->header.stamp.sec) + "." + to_string(pcmsg->header.stamp.nsec) + ".pcd";
      // pcl::io::savePCDFile(str, *pcmsg);
      auto str = to_string(pcmsg->header.stamp.sec) + "." + to_string(pcmsg->header.stamp.nsec) + ".ply";
      // pcl::PCLPointCloud2 pc2;
      // pcl_conversions::toPCL(*pcmsg, pc2);
      pcl::PointCloud<pcl::PointXYZ> pc;
      pcl::fromROSMsg(*pcmsg, pc);
      pcl::io::savePLYFile(str, pc);
    }
  }
  bag.close();
}
