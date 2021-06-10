#include <bits/stdc++.h>
#include <nav_msgs/Path.h>
#include "common/common.h"

using namespace std;

void MakeLocalGt(nav_msgs::Path &path, ros::Time start) {
  auto startpose = common::GetPose(path.poses, start);
  Eigen::Affine3d preT;
  tf2::fromMsg(startpose, preT);
  preT = preT.inverse();
  path.poses[0].pose = tf2::toMsg(Eigen::Affine3d::Identity());
  for (size_t i = 1; i < path.poses.size(); ++i) {
    Eigen::Affine3d T;
    tf2::fromMsg(path.poses[i].pose, T);
    Eigen::Affine3d newT = preT * T;
    newT = common::Conserve2DFromAffine3d(newT);
    path.poses[i].pose = tf2::toMsg(newT);
  }
}

void WriteToFile(const nav_msgs::Path &path, string filename) {
  auto fp = fopen(filename.c_str(), "w");
  fprintf(fp, "# time x y z qx qy qz qw\n");
  for (auto &p : path.poses) {
    auto t = p.header.stamp.toSec();
    auto x = p.pose.position.x;
    auto y = p.pose.position.y;
    auto z = p.pose.position.z;
    auto qx = p.pose.orientation.x;
    auto qy = p.pose.orientation.y;
    auto qz = p.pose.orientation.z;
    auto qw = p.pose.orientation.w;
    fprintf(fp, "%f %f %f %f %f %f %f %f\n", t, x, y, z, qx, qy, qz, qw);
  }
  fclose(fp);
}

int main(int argc, char **argv) {
  nav_msgs::Path path, pathgt, pathgtsync;
  common::SerializationInput(argv[1], path);
  common::SerializationInput(argv[2], pathgt);
  // for (auto &p : path.poses) {
  //   geometry_msgs::PoseStamped pst;
  //   pst.header = p.header;
  //   pst.pose = common::GetPose(pathgt.poses, p.header.stamp);
  // }
  MakeLocalGt(pathgt, path.poses[0].header.stamp);
  WriteToFile(path, "stamped_traj_estimate.txt");
  WriteToFile(pathgt, "stamped_groundtruth.txt");
}
