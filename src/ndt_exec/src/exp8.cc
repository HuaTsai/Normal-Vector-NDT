// Bunny Visualization
#include <common/common.h>
#include <ndt/matcher.h>
#include <ndt/visuals.h>
#include <pcl/common/transforms.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>

using namespace std;
using namespace Eigen;
using PointCloudType = pcl::PointCloud<pcl::PointXYZ>;

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
  pcl::io::loadOBJFile<pcl::PointXYZ>(
      JoinPath(WSPATH, "src/ndt/data/bunny.obj"), *source_pcl);
  pcl::io::loadOBJFile<pcl::PointXYZ>(
      JoinPath(WSPATH, "src/ndt/data/bunny.obj"), *target_pcl);
  for (auto &pt : *source_pcl) pt.x *= 10., pt.y *= 10., pt.z *= 10.;
  for (auto &pt : *target_pcl) pt.x *= 10., pt.y *= 10., pt.z *= 10.;
  source_pcl->header.frame_id = "map";
  target_pcl->header.frame_id = "map";
  vector<Vector3d> src, tgt;
  for (const auto &pt : *source_pcl) src.push_back(Vector3d(pt.x, pt.y, pt.z));
  for (const auto &pt : *target_pcl) tgt.push_back(Vector3d(pt.x, pt.y, pt.z));

  Affine3d subtf;
  if (argc == 7) {
    subtf = Affine3dFromXYZRPY({atof(argv[1]), atof(argv[2]), atof(argv[3]),
                                Deg2Rad(atof(argv[4])), Deg2Rad(atof(argv[5])),
                                Deg2Rad(atof(argv[6]))});
  } else {
    subtf = Affine3dFromXYZRPY({1, 1, 1, Deg2Rad(5), Deg2Rad(5), Deg2Rad(5)});
  }
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

  double cs = 0.5;
  auto m1 = NDTMatcher::GetBasic({kNDT, k1to1, kAnalytic, kNoReject}, cs);
  m1.SetSource(src);
  m1.SetTarget(tgt);
  auto res1 = m1.Align();
  // clang-format off
  printf("NDT:\n  (%f, %f)\n  corr: %d, iter: %d, bud: %.2f, nm: %.2f, ndt: %.2f, opt: %.2f, oth: %.2f, ttl: %.2f\n",
      TransNormRotDegAbsFromAffine3d(res1 * subtf)(0),
      TransNormRotDegAbsFromAffine3d(res1 * subtf)(1),
      m1.corres(),
      m1.iteration(),
      m1.timer().build() / 1000.,
      m1.timer().normal() / 1000.,
      m1.timer().ndt() / 1000.,
      m1.timer().optimize() / 1000.,
      m1.timer().others() / 1000.,
      m1.timer().total() / 1000.);
  // clang-format on

  auto m2 = NDTMatcher::GetBasic({kNNDT, k1to1, kAnalytic, kNoReject}, cs);
  m2.SetSource(src);
  m2.SetTarget(tgt);
  auto res2 = m2.Align();
  // clang-format off
  printf("NVNDT:\n  (%f, %f)\n  corr: %d, iter: %d, bud: %.2f, nm: %.2f, ndt: %.2f, opt: %.2f, oth: %.2f, ttl: %.2f\n",
      TransNormRotDegAbsFromAffine3d(res2 * subtf)(0),
      TransNormRotDegAbsFromAffine3d(res2 * subtf)(1),
      m2.corres(),
      m2.iteration(),
      m2.timer().build() / 1000.,
      m2.timer().normal() / 1000.,
      m2.timer().ndt() / 1000.,
      m2.timer().optimize() / 1000.,
      m2.timer().others() / 1000.,
      m2.timer().total() / 1000.);
  // clang-format on

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
      pub9.publish(MText("Normal Vector + NDT: " + to_string(idx),
                         Eigen::Vector3d(4, -6, 0)));
    }
    ros::spinOnce();
    cout << "Index: ";
  }
}
