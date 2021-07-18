/**
 * @file gtpath.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Read tf in bag files and write to gtpath.ser
 * @version 0.1
 * @date 2021-07-17
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include <bits/stdc++.h>
#include <rosbag/view.h>
#include <tf2_msgs/TFMessage.h>
#include <nav_msgs/Path.h>
#include <boost/program_options.hpp>
#include "common/common.h"

namespace po = boost::program_options;
using namespace std;

vector<string> GetDataPath(string data) {
  vector<string> ret;
  if (data == "log24") {
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log24_1535729278446231_scene-0299.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log24_1535729298446271_scene-0300.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log24_1535729318549677_scene-0301.bag");
  } else if (data == "log62-1") {
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193241547892_scene-0997.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193261546825_scene-0998.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193281648047_scene-0999.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193301547950_scene-1000.bag");
  } else if (data == "log62-2") {
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193461547574_scene-1004.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193481898177_scene-1005.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193501549291_scene-1006.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193521798725_scene-1007.bag");
  } else {
    cerr << "No specified data " << data << endl;
    exit(-1);
  }
  return ret;
}

vector<geometry_msgs::PoseStamped> GetPoses(const vector<string> &bag_paths) {
  vector<geometry_msgs::PoseStamped> ret;
  for (const auto &bag_path : bag_paths) {
    rosbag::Bag bag;
    bag.open(bag_path);
    for (rosbag::MessageInstance const m : rosbag::View(bag)) {
      tf2_msgs::TFMessage::ConstPtr tfmsg = m.instantiate<tf2_msgs::TFMessage>();
      if (tfmsg) {
        auto tf = tfmsg->transforms.at(0);
        if (tf.header.frame_id == "map" && tf.child_frame_id == "car") {
          geometry_msgs::PoseStamped pst;
          pst.header = tf.header;
          pst.pose.position.x = tf.transform.translation.x;
          pst.pose.position.y = tf.transform.translation.y;
          pst.pose.position.z = tf.transform.translation.z;
          pst.pose.orientation = tf.transform.rotation;
          ret.push_back(pst);
        }
      }
    }
    bag.close();
  }
  return ret;
}

int main(int argc, char **argv) {
  string data, outfolder;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (log24, log62-1, log62-2)")
      ("outfolder,o", po::value<string>(&outfolder)->required(), "Output folder path");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  auto vp = GetPoses(GetDataPath(data));
  nav_msgs::Path path;
  path.header.frame_id = "map";
  path.poses = vp;
  common::SerializationOutput(outfolder + "/gt_" + data + ".ser", path);
}