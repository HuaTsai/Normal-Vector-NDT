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

void conti_to_rospcall(const conti_radar::MeasurementConstPtr &msg,
                       sensor_msgs::PointCloud2::Ptr ros_pc2,
                       sensor_msgs::PointCloud2::Ptr ros_pc2_i) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc_i(new pcl::PointCloud<pcl::PointXYZI>);

  for (size_t i = 0; i < msg->points.size(); ++i) {
    pcl::PointXYZI point;
    if (msg->points.at(i).invalid_state == 0x00) {
      point.x = msg->points.at(i).longitude_dist;
      point.y = msg->points.at(i).lateral_dist;
      point.z = 0;
      point.intensity = msg->points.at(i).rcs;
      pc->points.push_back(point);
    } else {
      point.x = msg->points.at(i).longitude_dist;
      point.y = msg->points.at(i).lateral_dist;
      point.z = 0;
      point.intensity = msg->points.at(i).rcs;
      pc_i->points.push_back(point);
    }
  }

  pcl::toROSMsg(*pc, *ros_pc2);
  ros_pc2->header = msg->header;
  ros_pc2->is_dense = true;

  pcl::toROSMsg(*pc_i, *ros_pc2_i);
  ros_pc2_i->header = msg->header;
  ros_pc2_i->is_dense = true;
}
