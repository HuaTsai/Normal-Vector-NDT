#include <bits/stdc++.h>
#include <nav_msgs/Path.h>
#include <common/common.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;

void MakeGtLocal(nav_msgs::Path &path, ros::Time start) {
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
  string estpathfile, gtpathfile, outfolder;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("estpath,e", po::value<string>(&estpathfile)->required(), "Path of estpath.ser")
      ("gtpath,g", po::value<string>(&gtpathfile)->required(), "Path of gtpath.ser")
      ("outpath,o", po::value<string>(&outfolder)->required(), "Path of output folder");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  nav_msgs::Path estpath, gtpath;
  SerializationInput(estpathfile, estpath);
  SerializationInput(gtpathfile, gtpath);
  MakeGtLocal(gtpath, estpath.poses[0].header.stamp);
  WriteToFile(estpath, JoinPath(outfolder, "stamped_traj_estimate.txt"));
  WriteToFile(gtpath, JoinPath(outfolder, "stamped_groundtruth.txt"));
}
