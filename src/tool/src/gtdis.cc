/**
 * @file gtdis.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Compute ground truth distribution of xyt 
 * @version 0.1
 * @date 2021-07-18
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include <bits/stdc++.h>
#include <rosbag/view.h>
#include <tf2_msgs/TFMessage.h>
#include <nav_msgs/Path.h>
#include <boost/program_options.hpp>
#include <tf2_eigen/tf2_eigen.h>
#include <common/common.h>

namespace po = boost::program_options;
using namespace std;
using namespace Eigen;

void GetPosesDis(const vector<string> &bag_paths) {
  vector<Affine3d> affs;
  for (const auto &bag_path : bag_paths) {
    rosbag::Bag bag;
    bag.open(bag_path);
    for (rosbag::MessageInstance const m : rosbag::View(bag)) {
      tf2_msgs::TFMessage::ConstPtr tfmsg = m.instantiate<tf2_msgs::TFMessage>();
      if (tfmsg) {
        auto tf = tfmsg->transforms.at(0);
        if (tf.header.frame_id == "map" && tf.child_frame_id == "car") {
          geometry_msgs::Pose p;
          p.position.x = tf.transform.translation.x;
          p.position.y = tf.transform.translation.y;
          p.position.z = tf.transform.translation.z;
          p.orientation = tf.transform.rotation;
          Affine3d aff;
          tf2::fromMsg(p, aff);
          affs.push_back(aff);
        }
      }
    }
    bag.close();
  }
  vector<double> xs, ys, ts;
  for (int i = 0; i < (int)affs.size() - 5; ++i) {
    Affine3d dT = affs[i].inverse() * affs[i + 5];
    Affine3d dTcon = common::Conserve2DFromAffine3d(dT);
    xs.push_back(dTcon.translation()(0));
    ys.push_back(dTcon.translation()(1));
    ts.push_back(Rotation2Dd(dTcon.matrix().block<2, 2>(0, 0)).angle());
  }
  // copy(xs.begin(), xs.end(), ostream_iterator<double>(cout, ", "));
  cout << "x: (" << *min_element(xs.begin(), xs.end()) << ", " << *max_element(xs.begin(), xs.end()) << ")" << endl;
  // copy(ys.begin(), ys.end(), ostream_iterator<double>(cout, ", "));
  cout << "y: (" << *min_element(ys.begin(), ys.end()) << ", " << *max_element(ys.begin(), ys.end()) << ")" << endl;
  // copy(ts.begin(), ts.end(), ostream_iterator<double>(cout, ", "));
  cout << "t: (" << *min_element(ts.begin(), ts.end()) << ", " << *max_element(ts.begin(), ts.end()) << ")" << endl;
}

int main(int argc, char **argv) {
  string data, outfolder;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (log24, log62-1, log62-2)");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  GetPosesDis(GetBagsPath(data));
}
