#pragma once
#include <conti_radar/Measurement.h>
#include <pcl_ros/point_cloud.h>

sensor_msgs::PointCloud2 conti_to_rospc(const conti_radar::Measurement &msg, int label) {
  sensor_msgs::PointCloud2 ret;
  pcl::PointCloud<pcl::PointXYZL>::Ptr pc(new pcl::PointCloud<pcl::PointXYZL>);
  for (size_t i = 0; i < msg.points.size(); ++i) {
    pcl::PointXYZL point;
    if (msg.points.at(i).invalid_state == 0x00) {
      point.x = msg.points.at(i).longitude_dist;
      point.y = msg.points.at(i).lateral_dist;
      point.z = 0;
      point.label = label;
      // point.intensity = msg.points.at(i).rcs;
      pc->points.push_back(point);
    }
  }
  pcl::toROSMsg(*pc, ret);
  ret.header = msg.header;
  ret.is_dense = true;
  return ret;
}

