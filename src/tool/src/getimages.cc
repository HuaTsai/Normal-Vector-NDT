#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <rosbag/view.h>
#include <common/common.h>
#include <sensor_msgs/CompressedImage.h>

using namespace std;
namespace po = boost::program_options;

int main(int argc, char **argv) {
  string data, outfolder;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,i", po::value<string>(&data)->required(), "Data (log24, log35-1, log62-1, log62-2)")
      ("outfolder,o", po::value<string>(&outfolder)->required(), "Output folder");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  vector<sensor_msgs::CompressedImage> imb, imbl, imbr, imf, imfl, imfr;
  auto paths = GetBagsPath(data);
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
  SerializationOutput(JoinPath(outfolder, "back.ser"), imb);
  SerializationOutput(JoinPath(outfolder, "back_left.ser"), imbl);
  SerializationOutput(JoinPath(outfolder, "back_right.ser"), imbr);
  SerializationOutput(JoinPath(outfolder, "front.ser"), imf);
  SerializationOutput(JoinPath(outfolder, "front_left.ser"), imfl);
  SerializationOutput(JoinPath(outfolder, "front_right.ser"), imfr);
}
