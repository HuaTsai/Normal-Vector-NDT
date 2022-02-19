#include <bits/stdc++.h>
#include <common/common.h>
#include <sndt/visuals.h>
#include <sndt_exec/wrapper.h>

#include <boost/program_options.hpp>
#include <tqdm/tqdm.h>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

Affine2d GetTF(const nav_msgs::Path &path, const ros::Time &t0, const ros::Time &tj) {
  Affine3d aff0, affj;
  tf2::fromMsg(GetPose(path.poses, t0), aff0);
  tf2::fromMsg(GetPose(path.poses, tj), affj);
  auto ret3 = Conserve2DFromAffine3d(aff0.inverse() * affj);
  Affine2d ret = Translation2d(ret3.translation()(0), ret3.translation()(1)) *
                 Rotation2Dd(ret3.rotation().block<2, 2>(0, 0));
  return ret;
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  int f;
  double cell_size = 1.5, voxel, radius = 1.5;
  string data;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("f,f", po::value<int>(&f)->default_value(1), "Frames");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  po::notify(vm);

  nav_msgs::Path gtpath;
  SerializationInput(JoinPath(GetDataPath(data), "gt.ser"), gtpath);
  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath(data), "lidar.ser"), vpc);

  ros::init(argc, argv, "augmap");
  ros::NodeHandle nh;
  auto pubt = nh.advertise<Marker>("tgt", 0, true);

  vector<Affine2d> Trs;
  vector<int> nums;
  Affine2d Tr = Affine2d::Identity();
  vector<Vector2d> map;
  int n = vpc.size() - 1;
  tqdm bar;
  for (int i = 0; i + f < n; i += f) {
    bar.progress(i, n);
    auto tgt = PCMsgTo2D(vpc[i], voxel);
    auto src = PCMsgTo2D(vpc[i + f], voxel);
    vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, aff2}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{{src, aff2}};

    // ICPParameters params1;
    // params1.reject = true;
    // params1._usedtime.Start();
    // auto tgt1 = MakePoints(datat, params1);
    // auto src1 = MakePoints(datas, params1);
    // auto T1 = ICPMatch(tgt1, src1, params1);
    // Tr = Tr * T1;

    D2DNDTParameters params7;
    params7.cell_size = cell_size;
    params7.r_variance = params7.t_variance = 0;
    params7.d2 = 0.04;
    params7._usedtime.Start();
    auto tgt7 = MakeNDTMap(datat, params7);
    auto src7 = MakeNDTMap(datas, params7);
    auto T7 = SNDTMatch2(tgt7, src7, params7);
    Tr = Tr * T7;

    nums.push_back(i);
    Trs.push_back(Tr);

    // auto tj = vpc[i].header.stamp;
    // auto T = GetTF(gtpath, vpc[0].header.stamp, tj);
    // TransformPointsInPlace(tgt, T * aff2);
    // for (auto pt : tgt) map.push_back(pt);
  }
  bar.finish();

  while (1) {
    for (int i = 0; i < nums.size(); ++i) {
      auto tgt = PCMsgTo2D(vpc[nums[i]], voxel);
      TransformPointsInPlace(tgt, Trs[i] * aff2);
      pubt.publish(MarkerOfPoints(tgt, 0.5));
      ros::Rate(20).sleep();
    }
  }
  // pubt.publish(MarkerOfPoints(map, 0.1));
  ros::spin();
}