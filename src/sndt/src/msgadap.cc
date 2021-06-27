#include <bits/stdc++.h>
#include <pcl_ros/point_cloud.h>
#include <nav_msgs/Path.h>
#include <boost/program_options.hpp>
#include "common/common.h"
#include "common/EgoPointClouds.h"
#include "sndt/EgoPointClouds.h"
#include "dbg/dbg.h"
#include "sndt/wrapper.hpp"

namespace po = boost::program_options;
using namespace std;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

vector<geometry_msgs::Point> PointCloudAdapter(
    const sensor_msgs::PointCloud2 &msg) {
  vector<geometry_msgs::Point> ret;
  PointCloud pc;
  pcl::fromROSMsg(msg, pc);
  for (auto &p : pc.points) {
    geometry_msgs::Point pt;
    pt.x = p.x; pt.y = p.y; pt.z = p.z;
    ret.push_back(pt);
  }
  return ret;
}

int main(int argc, char **argv) {
  string infile;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("infile", po::value<string>(&infile)->required(), "EgoPointClouds path");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  vector<sndt::EgoPointClouds> v1;
  common::SerializationInput(infile, v1);
  vector<common::EgoPointClouds> v2(v1.size());
  for (size_t i = 0; i < v1.size(); ++i) {
    v2[i].stamp = v1[i].stamp;
    v2[i].vxyt = v1[i].vxyt;
    v2[i].augpc = PointCloudAdapter(v1[i].augpc);
    for (auto &p : v1[i].pcs) {
      common::PointCloudSensor pcs;
      pcs.id = p.id;
      pcs.origin = p.origin;
      pcs.points = PointCloudAdapter(p.pc);
      v2[i].pcs.push_back(pcs);
    }
  }
  string str("_c.ser");
  string outfile = infile.substr(0, infile.size() - 4) + str;
  common::SerializationOutput(outfile, v2);
}
