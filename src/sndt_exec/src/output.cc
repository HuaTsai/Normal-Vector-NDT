#include <bits/stdc++.h>
#include <common/common.h>
#include <metric/metric.h>
#include <sndt/visuals.h>
#include <sndt_exec/wrapper.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_2D.h>
#include <boost/program_options.hpp>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  vector<int> m;
  double voxel;
  string data;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  po::notify(vm);

  ros::init(argc, argv, "output");
  ros::NodeHandle nh;
  auto pub = nh.advertise<sensor_msgs::PointCloud2>("pc", 0, true);
  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath(data), "lidar.ser"), vpc);
  for (size_t i = 0; i < vpc.size(); ++i) {
    auto tgt = PCMsgTo2D(vpc[i], voxel);
    // auto src = PCMsgTo2D(vpc[i + 1], voxel);
    pcl::PointCloud<pcl::PointXYZ> pc;
    for (auto p : tgt) {
      pcl::PointXYZ pt;
      pt.x = (aff2 * p)(0), pt.y = (aff2 * p)(1), pt.z = 0;
      pc.push_back(pt);
    }
    pc.header.frame_id = "map";
    pub.publish(pc);
    ros::Rate(10).sleep();
    pcl::io::savePCDFile(to_string(i) + ".pcd", pc);
  }
}