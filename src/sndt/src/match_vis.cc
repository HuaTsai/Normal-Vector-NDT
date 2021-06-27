#include <bits/stdc++.h>
#include "common/common.h"
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>
#include <std_msgs/Int64.h>
#include "sndt/ndt_map_2d.hpp"
#include "sndt/ndt_conversions.hpp"
#include "sndt/ndt_matcher_2d.hpp"
#include "normal2d/Normal2dEstimation.h"
#include <pcl/search/kdtree.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;

common::MatchInternal mit;
ros::Publisher pub_pc;
ros::Publisher pub_pc2;
ros::Publisher pub_pc3;
ros::Publisher pub_gt;
ros::Publisher pub_markers;
ros::Publisher pub_markers2;
ros::Publisher pub_map;
ros::Publisher pub_map2;
ros::Publisher pub_grid;
ros::Publisher pub_grid2;
ros::Publisher pub_normal;
ros::Publisher pub_normal2;
ros::Publisher pub_normalcov;
ros::Publisher pub_normalcov2;
shared_ptr<tf2_ros::StaticTransformBroadcaster> stb;

pcl::PointCloud<pcl::PointXYZ> PCLFromMatrixXd(const Eigen::MatrixXd &mtx, string frame) {
  pcl::PointCloud<pcl::PointXYZ> ret;
  for (int i = 0; i < mtx.cols(); ++i) {
    ret.push_back(pcl::PointXYZ(mtx(0, i), mtx(1, i), mtx(2, i)));
  }
  ret.header.frame_id = frame;
  return ret;
}

void ComputeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr pc, double radius,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr normals) {
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  Normal2dEstimation ne;
  ne.setInputCloud(pc);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(radius);
  ne.compute(normals);
}

void PubNDTMsg(ros::Publisher *mappub, ros::Publisher *gridpub,
               ros::Publisher *normalpub, ros::Publisher *normalcovpub,
               pcl::PointCloud<pcl::PointXYZ>::Ptr pc, double cell_size,
               double radius, common::Color color) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr no(new pcl::PointCloud<pcl::PointXYZ>);
  ComputeNormals(pc, radius, no);
  NDTMap map(cell_size);
  map.LoadPointCloud(*pc, *no);
  auto msg = ToMessage(map, pc->header.frame_id);
  mappub->publish(msg);
  gridpub->publish(GridMarkerArrayFromNDTMapMsg(msg, color));
  normalpub->publish(NormalMarkerArrayFromNDTMapMsg(msg, color));
  normalcovpub->publish(NormalCovMarkerArrayFromNDTMapMsg(msg, common::Color::kGray));
}

void subcb(const std_msgs::Int64::ConstPtr &msg) {
  // rostopic pub /idx std_msgs/Int64 <idx>
  auto corrs = mit.corrs();
  auto tfs = mit.tfs();
  int maxcols = -1;
  for (const auto &corr : corrs) {
    maxcols = max(maxcols, static_cast<int>(corr.cols()));
  }
  if (msg->data < 0 || msg->data >= (int)corrs.size()) {
    ROS_WARN("Invalid iteration %ld", msg->data);
  } else {
    int idx = msg->data;
    auto corr = corrs.at(idx);
    auto tf = tfs.at(idx);

    auto mtx = common::Matrix4fFromMatrix3d(common::Matrix3dFromXYTDegree({tf(0), tf(1), tf(2)}));
    auto gmtx = common::Matrix4fFromMatrix3d(common::Matrix3dFromXYTRadian({2.73557, -0.0355944, 0.0110073}));

    // NDT
    if (mit.cell_size() != 0) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZ>);
      *tgt = PCLFromMatrixXd(mit.target(), "map");
      PubNDTMsg(&pub_map, &pub_grid, &pub_normal, &pub_normalcov, tgt,
                mit.cell_size(), 2, common::Color::kRed);

      pcl::PointCloud<pcl::PointXYZ>::Ptr src(new pcl::PointCloud<pcl::PointXYZ>);
      *src = PCLFromMatrixXd(mit.source(), "map2");
      PubNDTMsg(&pub_map2, &pub_grid2, &pub_normal2, &pub_normalcov2, src,
                mit.cell_size(), 2, common::Color::kLime);
    }

    stb->sendTransform(common::TstFromMatrix4f(mtx, ros::Time::now(), "map", "map2"));
    stb->sendTransform(common::TstFromMatrix4f(gmtx, ros::Time::now(), "map", "gt"));

    pub_pc.publish(PCLFromMatrixXd(mit.target(), "map"));
    pub_pc2.publish(PCLFromMatrixXd(mit.source(), "map2"));
    pub_pc3.publish(PCLFromMatrixXd(mit.source(), "map3"));
    pub_gt.publish(PCLFromMatrixXd(mit.source(), "gt"));

    // corr.row(6).array() -= corr.row(6).minCoeff();
    // corr.row(6).normalize();
    // corr.row(6).array() += 0.5;

    visualization_msgs::MarkerArray ma;
    visualization_msgs::MarkerArray ma2;
    for (int i = 0; i < corr.cols(); ++i) {
      auto start = corr.block<3, 1>(0, i);
      auto end = corr.block<3, 1>(3, i);
      auto mc = common::MakeArrowMarkerByEnds(i, "map", start, end, common::Color::kFuchsia);
      ma.markers.push_back(mc);

      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << corr(6, i);
      auto p = tf2::toMsg(common::Affine3dFromXYZRPY(start, Eigen::Vector3d::Zero()));
      auto ms = common::MakeTextMarker(i, "map", oss.str(), p, common::Color::kBlack);
      ma2.markers.push_back(ms);
    }
    // corrs
    for (int i = corr.cols(); i < maxcols; ++i) {
      visualization_msgs::Marker m;
      m.id = i;
      m.action = visualization_msgs::Marker::DELETE;
      ma.markers.push_back(m);
      ma2.markers.push_back(m);
    }
    pub_markers.publish(ma);
    pub_markers2.publish(ma2);
    ROS_INFO("Show iteration %d with %ld correspondences", idx, corr.cols());
  }
}

int main(int argc, char **argv) {
  string infile;
  int idx;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("infile", po::value<string>(&infile)->required(), "Path of the mits file")
      ("index", po::value<int>(&idx)->required(), "Matching id");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  vector<common::MatchInternal> mits;
  common::SerializeIn(infile, mits);
  mit = mits[idx];
  ROS_INFO("Reading MIT ... Done!");
  ROS_INFO("There are %ld matching iterations!", mit.corrs().size());
  Eigen::Vector3d result = mit.tfs().back();

  // Eigen::Vector3d actual(2.73557, -0.0355944, 0.0110073);
  // Eigen::Vector3d diff = result - actual;
  // ROS_INFO("Result: %f, %f, %f", result(0), result(1), result(2));
  // ROS_INFO("Actual: %f, %f, %f", actual(0), actual(1), actual(2));
  // ROS_INFO("Diff: %f, %f", diff.block<2, 1>(0, 0).norm(), abs(diff(2)));

  Eigen::Vector3d guess(mit.tfs()[0](0), mit.tfs()[0](1), mit.tfs()[0](2));
  Eigen::Vector3d diff = result - guess;
  ROS_INFO("Result: %f, %f, %f", result(0), result(1), result(2));
  ROS_INFO(" Guess: %f, %f, %f", guess(0), guess(1), guess(2));
  ROS_INFO("  Diff: %f, %f", diff.block<2, 1>(0, 0).norm(), abs(diff(2)));

  ros::init(argc, argv, "match_vis");
  ros::NodeHandle nh;
  stb = make_shared<tf2_ros::StaticTransformBroadcaster>();
  pub_pc = nh.advertise<sensor_msgs::PointCloud2>("pc", 0, true);
  pub_pc2 = nh.advertise<sensor_msgs::PointCloud2>("pc2", 0, true);
  pub_pc3 = nh.advertise<sensor_msgs::PointCloud2>("pc3", 0, true);
  pub_gt = nh.advertise<sensor_msgs::PointCloud2>("gt", 0, true);
  pub_markers = nh.advertise<visualization_msgs::MarkerArray>("markers", 0, true);
  pub_markers2 = nh.advertise<visualization_msgs::MarkerArray>("markers2", 0, true);
  pub_map = nh.advertise<sndt::NDTMapMsg>("map", 0, true);
  pub_map2 = nh.advertise<sndt::NDTMapMsg>("map2", 0, true);
  pub_grid = nh.advertise<visualization_msgs::MarkerArray>("grid", 0, true);
  pub_grid2 = nh.advertise<visualization_msgs::MarkerArray>("grid2", 0, true);
  pub_normal = nh.advertise<visualization_msgs::MarkerArray>("normal", 0, true);
  pub_normal2 = nh.advertise<visualization_msgs::MarkerArray>("normal2", 0, true);
  pub_normalcov = nh.advertise<visualization_msgs::MarkerArray>("normalcov", 0, true);
  pub_normalcov2 = nh.advertise<visualization_msgs::MarkerArray>("normalcov2", 0, true);
  ros::Subscriber sub = nh.subscribe<std_msgs::Int64>("idx", 0, subcb);

  ros::spin();
}
