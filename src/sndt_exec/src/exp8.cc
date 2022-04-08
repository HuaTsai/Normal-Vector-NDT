// Rabbit Visualization
#include <common/common.h>
#include <ndt/matcher.h>
#include <ndt/visuals.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>

using namespace std;
using namespace Eigen;
using PointCloudType = pcl::PointCloud<pcl::PointXYZ>;

constexpr auto kCell = MarkerOptions::kCell;
constexpr auto kGreen = MarkerOptions::kGreen;
constexpr auto kRed = MarkerOptions::kRed;
constexpr auto kCov = MarkerOptions::kCov;

visualization_msgs::Marker MText(std::string text,
                                 const Eigen::Vector3d &position) {
  visualization_msgs::Marker ret;
  ret.header.frame_id = "map";
  ret.id = 0;
  ret.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  ret.text = text;
  ret.pose.position.x = position(0);
  ret.pose.position.y = position(1);
  ret.pose.position.z = position(2);
  ret.pose.orientation.w = 1;
  ret.scale.z = 0.7;
  ret.color.r = ret.color.g = ret.color.b = 1;
  ret.color.a = 1;
  return ret;
}

int main(int argc, char **argv) {
  PointCloudType::Ptr source_pcl = PointCloudType::Ptr(new PointCloudType);
  PointCloudType::Ptr target_pcl = PointCloudType::Ptr(new PointCloudType);
  pcl::io::loadPCDFile<pcl::PointXYZ>(
      JoinPath(WSPATH, "src/ndt/data/bunny2.pcd"), *source_pcl);
  pcl::io::loadPCDFile<pcl::PointXYZ>(
      JoinPath(WSPATH, "src/ndt/data/bunny2.pcd"), *target_pcl);
  source_pcl->header.frame_id = "map";
  target_pcl->header.frame_id = "map";
  vector<Vector3d> src, tgt;
  for (const auto &pt : *source_pcl) src.push_back(Vector3d(pt.x, pt.y, pt.z));
  for (const auto &pt : *target_pcl) tgt.push_back(Vector3d(pt.x, pt.y, pt.z));

  auto subtf =
      Affine3dFromXYZRPY({1, 1, 1, Deg2Rad(5), Deg2Rad(5), Deg2Rad(5)});
  pcl::transformPointCloud(*source_pcl, *source_pcl, subtf.cast<float>());
  TransformPointsInPlace(src, subtf);

  ros::init(argc, argv, "exp8");
  ros::NodeHandle nh;
  // clang-format off
  ros::Publisher pub1 = nh.advertise<sensor_msgs::PointCloud2>("rabbit1", 0, true);
  ros::Publisher pub2 = nh.advertise<sensor_msgs::PointCloud2>("rabbit2", 0, true);
  ros::Publisher pub3 = nh.advertise<sensor_msgs::PointCloud2>("rabbit3", 0, true);
  ros::Publisher pub4 = nh.advertise<sensor_msgs::PointCloud2>("rabbit4", 0, true);
  ros::Publisher pub5 = nh.advertise<visualization_msgs::MarkerArray>("markers1", 0, true);
  ros::Publisher pub6 = nh.advertise<visualization_msgs::MarkerArray>("markers2", 0, true);
  ros::Publisher pub7 = nh.advertise<visualization_msgs::MarkerArray>("markers3", 0, true);
  ros::Publisher pub8 = nh.advertise<visualization_msgs::Marker>("marker1", 0, true);
  ros::Publisher pub9 = nh.advertise<visualization_msgs::Marker>("marker2", 0, true);
  // clang-format on
  pub1.publish(*source_pcl);
  pub2.publish(*target_pcl);

  auto m1 = NDTMatcher::GetBasic({kLS, kNDT, k1to1}, 0.5);
  m1.SetSource(src);
  m1.SetTarget(tgt);
  m1.Align();

  auto m2 = NDTMatcher::GetBasic({kLS, kNNDT, k1to1}, 0.5);
  m2.SetSource(src);
  m2.SetTarget(tgt);
  m2.Align();

  printf("iter: %d, opt: %.2f, ttl: %.2f\n", m1.iteration(),
         m1.timer().optimize() / 1000., m1.timer().total() / 1000.);
  printf("iter: %d, opt: %.2f, ttl: %.2f\n", m2.iteration(),
         m2.timer().optimize() / 1000., m2.timer().total() / 1000.);

  pub5.publish(MarkerOfNDT(m1.tmap(), {kRed, kCov}));
  size_t idx;
  cout << "Index: ";
  while (cin >> idx) {
    PointCloudType o1, o2;
    if (idx < m1.tfs().size()) {
      pcl::transformPointCloud(*source_pcl, o1, m1.tfs()[idx].cast<float>());
      pub3.publish(o1);
      pub6.publish(MarkerOfNDT(m1.smap(), {kGreen, kCov}, m1.tfs()[idx]));
      pub8.publish(MText("NDT: " + to_string(idx), Eigen::Vector3d(4, -6, 0)));
    }

    if (idx < m2.tfs().size()) {
      pcl::transformPointCloud(*source_pcl, o2, m2.tfs()[idx].cast<float>());
      pub4.publish(o2);
      pub7.publish(MarkerOfNDT(m2.smap(), {kGreen, kCov}, m2.tfs()[idx]));
      pub9.publish(
          MText("Normal + NDT: " + to_string(idx), Eigen::Vector3d(4, -6, 0)));
    }
    ros::spinOnce();
    cout << "Index: ";
  }
}
