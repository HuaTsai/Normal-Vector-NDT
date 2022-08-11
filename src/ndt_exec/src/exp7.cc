// Bunny Test
#include <common/common.h>
#include <metric/metric.h>
#include <nav_msgs/Path.h>
#include <ndt/matcher.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

#include <boost/program_options.hpp>

using PointCloudType = pcl::PointCloud<pcl::PointXYZ>;
using Marker = visualization_msgs::Marker;
using namespace std;
using namespace Eigen;

Marker MP(const vector<Vector3d> &points, bool red) {
  Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = ros::Time::now();
  ret.id = 0;
  ret.type = Marker::SPHERE_LIST;
  ret.action = Marker::ADD;
  ret.scale.x = ret.scale.y = ret.scale.z = 0.05;
  ret.pose = tf2::toMsg(Affine3d::Identity());
  ret.color.a = 1;
  ret.color.r = red ? 1 : 0;
  ret.color.g = red ? 0 : 1;
  for (const auto &point : points) {
    geometry_msgs::Point pt;
    pt.x = point(0), pt.y = point(1), pt.z = point(2);
    ret.points.push_back(pt);
  }
  return ret;
}

int main(int argc, char **argv) {
  PointCloudType::Ptr source_pcl = PointCloudType::Ptr(new PointCloudType);
  PointCloudType::Ptr target_pcl = PointCloudType::Ptr(new PointCloudType);
  pcl::io::loadPCDFile<pcl::PointXYZ>(
      JoinPath(WSPATH, "src/ndt/data/bunny1.pcd"), *source_pcl);
  pcl::io::loadOBJFile<pcl::PointXYZ>(
      JoinPath(WSPATH, "src/ndt/data/bunny.obj"), *target_pcl);
  vector<Vector3d> source;
  vector<Vector3d> target;
  for (const auto &pt : *source_pcl)
    source.push_back(Vector3d(pt.x, pt.y, pt.z));
  for (const auto &pt : *target_pcl)
    target.push_back(10 * Vector3d(pt.x, pt.y, pt.z));
  cout << source.size() << ", " << target.size() << endl;

  ros::init(argc, argv, "exp7");
  ros::NodeHandle nh;
  ros::Publisher pub1 = nh.advertise<Marker>("src", 0, true);
  ros::Publisher pub2 = nh.advertise<Marker>("tgt", 0, true);
  ros::Publisher pub3 = nh.advertise<Marker>("out", 0, true);
  pub1.publish(MP(source, false));
  pub2.publish(MP(target, true));

  auto m = NDTMatcher::GetBasic({kNVNDT, k1to1}, 1);
  m.SetSource(source);
  m.SetTarget(target);
  auto res = m.Align();
  auto output = TransformPoints(source, res);
  pub3.publish(MP(output, false));

  ros::spin();
}
