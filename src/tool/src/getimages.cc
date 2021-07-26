#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <rosbag/view.h>
#include "common/common.h"
#include <sensor_msgs/CompressedImage.h>

using namespace std;
namespace po = boost::program_options;

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

int main(int argc, char **argv) {
  string data, outfolder;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,i", po::value<string>(&data)->required(), "Data Path")
      ("outfolder,o", po::value<string>(&outfolder)->required(), "Output folder");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  vector<sensor_msgs::CompressedImage> imb, imbl, imbr, imf, imfl, imfr;
  auto paths = GetDataPath(data);
  for (auto path : paths) {
    rosbag::Bag bag;
    bag.open(path);
    for (rosbag::MessageInstance const m : rosbag::View(bag)) {
      sensor_msgs::CompressedImage::ConstPtr msg = m.instantiate<sensor_msgs::CompressedImage>();
      if (msg && m.getTopic() == "image_back/compressed") {
        imb.push_back(*msg);
      } else if (msg && m.getTopic() == "image_back_left/compressed") {
        imbl.push_back(*msg);
      } else if (msg && m.getTopic() == "image_back_right/compressed") {
        imbr.push_back(*msg);
      } else if (msg && m.getTopic() == "image_front/compressed") {
        imf.push_back(*msg);
      } else if (msg && m.getTopic() == "image_front_left/compressed") {
        imfl.push_back(*msg);
      } else if (msg && m.getTopic() == "image_front_right/compressed") {
        imfr.push_back(*msg);
      }
    }
    bag.close();
  }
  common::SerializationOutput(outfolder + "/" + "back.ser", imb);
  common::SerializationOutput(outfolder + "/" + "back_left.ser", imbl);
  common::SerializationOutput(outfolder + "/" + "back_right.ser", imbr);
  common::SerializationOutput(outfolder + "/" + "front.ser", imf);
  common::SerializationOutput(outfolder + "/" + "front_left.ser", imfl);
  common::SerializationOutput(outfolder + "/" + "front_right.ser", imfr);
}
