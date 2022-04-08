// Fake Data
#include <common/common.h>
#include <ndt/matcher.h>
#include <ndt/visuals.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>

using namespace std;
using namespace Eigen;
using PointCloudType = pcl::PointCloud<pcl::PointXYZ>;

constexpr auto kGreen = MarkerOptions::kGreen;
constexpr auto kRed = MarkerOptions::kRed;
constexpr auto kCell = MarkerOptions::kCell;
constexpr auto kCov = MarkerOptions::kCov;

vector<Vector3d> MakeFakeData() {
  vector<Vector3d> ret;
  for (double x = -3.8; x < 3.8; x += 0.05) {
    double z = 0.3 * (x + 4) * sin(x + 4) + 0.1;
    ret.push_back(Vector3d(x, 0.2, z));
    ret.push_back(Vector3d(x, 0.3, z));
    ret.push_back(Vector3d(x, 0.4, z));
  }
  return ret;
}

visualization_msgs::MarkerArray Arrows(const std::vector<Eigen::Vector3d> &pts,
                                       const std::vector<Eigen::Vector3d> &dirs,
                                       bool red) {
  visualization_msgs::MarkerArray ret;
  int id = 0;
  for (size_t i = 0; i < pts.size(); ++i) {
    visualization_msgs::Marker ar;
    ar.id = id++;
    ar.header.frame_id = "map";
    ar.header.stamp = ros::Time::now();
    ar.type = visualization_msgs::Marker::ARROW;
    ar.scale.x = 0.05;
    ar.scale.y = 0.2;
    ar.pose.orientation.w = 1;
    ar.color.a = 1, ar.color.r = red ? 1 : 0, ar.color.g = red ? 0 : 1;
    geometry_msgs::Point pt;
    pt.x = pts[i](0), pt.y = pts[i](1), pt.z = pts[i](2);
    ar.points.push_back(pt);
    pt.x += dirs[i](0), pt.y += dirs[i](1), pt.z += dirs[i](2);
    ar.points.push_back(pt);
    ret.markers.push_back(ar);
  }
  return ret;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "exp8_2");
  ros::NodeHandle nh;
  // clang-format off
  ros::Publisher pub1 = nh.advertise<visualization_msgs::MarkerArray>("markers1", 0, true);
  ros::Publisher pub2 = nh.advertise<visualization_msgs::MarkerArray>("markers2", 0, true);
  ros::Publisher pub3 = nh.advertise<visualization_msgs::MarkerArray>("markers3", 0, true);
  ros::Publisher pub4 = nh.advertise<visualization_msgs::MarkerArray>("markers4", 0, true);
  ros::Publisher pu1 = nh.advertise<visualization_msgs::Marker>("marker1", 0, true);
  ros::Publisher pu2 = nh.advertise<visualization_msgs::Marker>("marker2", 0, true);
  ros::Publisher pu3 = nh.advertise<visualization_msgs::Marker>("marker3", 0, true);
  ros::Publisher pu4 = nh.advertise<visualization_msgs::Marker>("marker4", 0, true);
  // clang-format on

  auto tgt = MakeFakeData();
  auto src = TransformPoints(tgt, Affine3dFromXYZRPY({-0.3, 0, 1, 0, 0.2, 0}));

  auto op1 = {kLS, kNDT, k1to1, kNoReject};
  auto m1 = NDTMatcher::GetBasic(op1, 0.5, 0.05);
  m1.SetSource(src);
  m1.SetTarget(tgt);
  auto res1 = m1.Align();
  cout << XYZRPYFromAffine3d(res1).transpose() << endl;
  cout << m1.iteration() << ", " << m1.timer().optimize() << endl;

  auto op2 = {kLS, kNNDT, k1to1, kNoReject};
  auto m2 = NDTMatcher::GetBasic(op2, 0.5, 0.05);
  m2.SetSource(src);
  m2.SetTarget(tgt);
  auto res2 = m2.Align();
  cout << XYZRPYFromAffine3d(res2).transpose() << endl;
  cout << m2.iteration() << ", " << m2.timer().optimize() << endl;

  vector<Vector3d> pts1, dirs1, pts2, dirs2;
  for (auto elem : *m1.tmap()) {
    const Cell &cell = elem.second;
    pts1.push_back(cell.GetMean());
    dirs1.push_back(cell.GetNormal());
  }
  for (auto elem : *m1.smap()) {
    const Cell &cell = elem.second;
    pts2.push_back(cell.GetMean());
    dirs2.push_back(cell.GetNormal());
  }
  pub1.publish(MarkerOfNDT(m1.tmap(), {kRed, kCell, kCov}));
  pub2.publish(MarkerOfNDT(m1.smap(), {kGreen, kCell, kCov}));
  pub3.publish(Arrows(pts1, dirs1, true));
  pub4.publish(Arrows(pts2, dirs2, false));
  pu1.publish(MarkerOfPoints(tgt, true));
  pu2.publish(MarkerOfPoints(src, false));
  pu3.publish(MarkerOfPoints(TransformPoints(src, res1), false));
  pu4.publish(MarkerOfPoints(TransformPoints(src, res2), false));
  ros::spin();
}
