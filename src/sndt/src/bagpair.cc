#include <bits/stdc++.h>
#include <rosbag/view.h>
#include <boost/program_options.hpp>
#include <geometry_msgs/Transform.h>
#include <tf2_eigen/tf2_eigen.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <sndt/wrapper.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <std_msgs/Int32.h>
#include <ros/ros.h>
#include "common/common.h"

using namespace std;
namespace po = boost::program_options;
typedef pcl::PointCloud<pcl::PointXYZ> PCXYZ;

vector<Eigen::Matrix3d> guesses, results;
vector<sensor_msgs::PointCloud2> curpcs, prepcs;
ros::Publisher pub1, pub2, pub3;
tf2_ros::StaticTransformBroadcaster *stb;

Eigen::MatrixXd MatrixXdFromRosMsg(const sensor_msgs::PointCloud2 &msg) {
  PCXYZ pclcloud;
  pcl::fromROSMsg(msg, pclcloud);
  return MatrixXdFromPCL(pclcloud);
}

void cb(int idx) {
  // auto result = results[idx];
  auto curpc = curpcs[idx];
  auto prepc = prepcs[idx];
  common::MatchPackage mp;
  mp.source = MatrixXdFromRosMsg(curpc);
  mp.target = MatrixXdFromRosMsg(prepc);
  mp.guess = guesses[idx];
  common::MatchInternal mit;
  DoSNDT(mp, mit, {3, 1, 2, 0.001});
  Eigen::Matrix4f mtx4f = common::Matrix4fFromMatrix3d(mp.result);

  auto time = ros::Time::now();
  common::TstFromMatrix4f(mtx4f, time, "map", "");
  prepc.header.stamp = curpc.header.stamp = time;
  curpc.header.frame_id = "map";
  prepc.header.frame_id = "map";

  pub1.publish(curpc);
  pub2.publish(prepc);
  // stb->sendTransform();
}

int main(int argc, char **argv) {
  int idx;
  string bagfile;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("bagfile", po::value<string>(&bagfile)->required(), "Path of the rosbag file")
      ("index,i", po::value<int>(&idx)->required(), "Matching id");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  rosbag::Bag bag;
  bag.open(bagfile);
  for (rosbag::MessageInstance const m : rosbag::View(bag)) {
    geometry_msgs::Transform::ConstPtr tfmsg = m.instantiate<geometry_msgs::Transform>();
    if (tfmsg && m.getTopic() == "guess") {
      Eigen::Matrix4f mtx4f = tf2::transformToEigen(*tfmsg).matrix().cast<float>();
      guesses.push_back(common::Matrix3dFromMatrix4f(mtx4f));
    } else if (tfmsg && m.getTopic() == "result") {
      Eigen::Matrix4f mtx4f = tf2::transformToEigen(*tfmsg).matrix().cast<float>();
      results.push_back(common::Matrix3dFromMatrix4f(mtx4f));
    }

    sensor_msgs::PointCloud2::ConstPtr pcmsg = m.instantiate<sensor_msgs::PointCloud2>();
    if (pcmsg && m.getTopic() == "curpc") {
      curpcs.push_back(*pcmsg);
    } else if (pcmsg && m.getTopic() == "prepc") {
      prepcs.push_back(*pcmsg);
    }
  }
  bag.close();

  ros::init(argc, argv, "bagpair");
  ros::NodeHandle nh;
  stb = new tf2_ros::StaticTransformBroadcaster();
  pub1 = nh.advertise<sensor_msgs::PointCloud2>("curpc", 0, true);
  pub2 = nh.advertise<sensor_msgs::PointCloud2>("prepc", 0, true);

  ros::spin();
}
