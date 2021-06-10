#include <bits/stdc++.h>
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <tf2_eigen/tf2_eigen.h>
#include <boost/program_options.hpp>
#include "common/common.h"
#include "dbg/dbg.h"

namespace po = boost::program_options;
using namespace std;

void MakeLocalGt(nav_msgs::Path &path) {
  Eigen::Affine3d preT;
  tf2::fromMsg(path.poses[0].pose, preT);
  preT = preT.inverse();
  path.poses[0].pose = tf2::toMsg(Eigen::Affine3d::Identity());
  for (size_t i = 1; i < path.poses.size(); ++i) {
    Eigen::Affine3d T;
    tf2::fromMsg(path.poses[i].pose, T);
    Eigen::Affine3d newT = preT * T;
    dbg(newT.matrix());
    newT = common::Conserve2DFromAffine3d(newT);
    dbg(newT.matrix());
    path.poses[i].pose = tf2::toMsg(newT);
  }
}

visualization_msgs::MarkerArray MakeTexts(const nav_msgs::Path &path) {
  visualization_msgs::MarkerArray ret;
  for (size_t i = 0; i < path.poses.size(); ++i) {
    auto m = common::MakeTextMarker(i, "map", to_string(i), path.poses[i].pose, common::Color::kBlack);
    ret.markers.push_back(m);
  }
  return ret;
}

int main(int argc, char **argv) {
  string pathfile1, pathfile2;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("path1,p1", po::value<string>(&pathfile1)->required(), "Serialized path1 file")
      ("path2,p2", po::value<string>(&pathfile2)->required(), "Serialized path2 file");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  nav_msgs::Path path1, path2;
  common::SerializationInput(pathfile1, path1);
  common::SerializationInput(pathfile2, path2);
  MakeLocalGt(path2);
 
  ros::init(argc, argv, "showpath");
  ros::NodeHandle nh;
  ros::Publisher pub1 = nh.advertise<nav_msgs::Path>("path", 0, true);
  ros::Publisher pub2 = nh.advertise<nav_msgs::Path>("path2", 0, true);
  ros::Publisher pub3 = nh.advertise<visualization_msgs::MarkerArray>("marker", 0, true);

  path2.header.stamp = path1.header.stamp = ros::Time::now();

  pub1.publish(path1);
  pub2.publish(path2);
  pub3.publish(MakeTexts(path1));
  ros::spin();
}
