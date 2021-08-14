#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <rosbag/view.h>
#include <common/common.h>
#include <sensor_msgs/PointCloud2.h>

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

  vector<sensor_msgs::PointCloud2> pcs;
  auto paths = GetBagsPath(data);
  for (auto path : paths) {
    rosbag::Bag bag;
    bag.open(path);
    for (rosbag::MessageInstance const m : rosbag::View(bag)) {
      sensor_msgs::PointCloud2::ConstPtr msg = m.instantiate<sensor_msgs::PointCloud2>();
      if (msg && m.getTopic() == "nuscenes_lidar") {
        pcs.push_back(*msg);
      }
    }
    bag.close();
  }
  SerializationOutput(JoinPath(outfolder, "lidar.ser"), pcs);
}
