#include <bits/stdc++.h>
#include <common/common.h>
#include <nav_msgs/Path.h>
#include <rosbag/view.h>
#include <tf2_msgs/TFMessage.h>

#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;

int main(int argc, char **argv) {
  string data;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (log24, log35-1, log62-1, log62-2)");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  auto bag_paths = GetBagsPath(data);
  for (const auto &bag_path : bag_paths) {
    rosbag::Bag bag;
    bag.open(bag_path);
    for (rosbag::MessageInstance const m : rosbag::View(bag)) {
      tf2_msgs::TFMessage::ConstPtr tfmsg =
          m.instantiate<tf2_msgs::TFMessage>();
      if (tfmsg) {
        auto tf = tfmsg->transforms.at(0);
        printf("%s -> %s: %f, %f, %f, %f, %f, %f, %f\n",
               tf.header.frame_id.c_str(), tf.child_frame_id.c_str(),
               tf.transform.translation.x, tf.transform.translation.y,
               tf.transform.translation.z, tf.transform.rotation.w,
               tf.transform.rotation.x, tf.transform.rotation.y,
               tf.transform.rotation.z);
      }
    }
    bag.close();
  }
}
