#include <bits/stdc++.h>
#include <pcl_ros/point_cloud.h>
#include <nav_msgs/Path.h>
#include <rosbag/view.h>
#include <tf2_msgs/TFMessage.h>
#include "common/common.h"
#include "sndt/EgoPointClouds.h"

using namespace std;

vector<string> GetDataPath(string data) {
  vector<string> ret;
  if (data == "log24") {
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log24_1535729278446231_scene-0299.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log24_1535729298446271_scene-0300.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log24_1535729318549677_scene-0301.bag");
  } else if (data == "log62") {
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193241547892_scene-0997.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193261546825_scene-0998.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193281648047_scene-0999.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193301547950_scene-1000.bag");
  } else if (data == "log62-2") {
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193461547574_scene-1004.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193481898177_scene-1005.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193501549291_scene-1006.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log62_1542193521798725_scene-1007.bag");
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
      if (tfmsg != nullptr) {
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
  // argv[1]: log24, log62, log62-2
  auto vp = GetPoses(GetDataPath(argv[1]));
  nav_msgs::Path path;
  path.header.frame_id = "map";
  path.poses = vp;
  common::SerializationOutput(APATH(20210422/pathgt.ser), path);
}