#include <bits/stdc++.h>
#include <common/common.h>
#include <sndt/visuals.h>
#include <sndt_exec/wrapper.h>
#include <tqdm/tqdm.h>

#include <boost/program_options.hpp>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

Affine2d GetBenchMark(const nav_msgs::Path &gtpath,
                      const ros::Time &t1,
                      const ros::Time &t2) {
  Affine3d To, Ti;
  tf2::fromMsg(GetPose(gtpath.poses, t2), To);
  tf2::fromMsg(GetPose(gtpath.poses, t1), Ti);
  Affine3d Tgt3 = Conserve2DFromAffine3d(Ti.inverse() * To);
  Affine2d ret = Translation2d(Tgt3.translation()(0), Tgt3.translation()(1)) *
                 Rotation2Dd(Tgt3.rotation().block<2, 2>(0, 0));
  return ret;
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  int n, f;
  double cell_size, voxel, radius = 1.5;
  string data;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("n,n", po::value<int>(&n)->required(), "To where")
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

  ros::init(argc, argv, "evalsingle");
  ros::NodeHandle nh;
  auto pubt = nh.advertise<Marker>("tgt", 0, true);
  auto pubs = nh.advertise<Marker>("src", 0, true);
  auto pubg = nh.advertise<Marker>("gt", 0, true);
  auto pub1 = nh.advertise<MarkerArray>("markers1", 0, true);
  auto pub2 = nh.advertise<MarkerArray>("markers2", 0, true);
  auto pub3 = nh.advertise<MarkerArray>("markers3", 0, true);
  auto pub4 = nh.advertise<MarkerArray>("markers4", 0, true);

  auto tgt = PCMsgTo2D(vpc[n], voxel);
  auto src = PCMsgTo2D(vpc[n + f], voxel);
  auto Tgt = GetBenchMark(gtpath, vpc[n].header.stamp, vpc[n + f].header.stamp);
  cout << "Interval: " << (vpc[n + f].header.stamp - vpc[n].header.stamp).toSec() << endl;
  cout << "Bench Mark: " << XYTDegreeFromAffine2d(Tgt).transpose() << endl;

  vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, aff2}};
  vector<pair<vector<Vector2d>, Affine2d>> datas{{src, aff2}};

  ICPParameters params1;
  params1.reject = true;
  params1._usedtime.Start();
  auto tgt1 = MakePoints(datat, params1);
  auto src1 = MakePoints(datas, params1);
  auto T1 = ICPMatch(tgt1, src1, params1);
  cout << "m1: " << params1._ceres_iteration << ", ";
  cout << TransNormRotDegAbsFromAffine2d(Tgt.inverse() * T1).transpose() << endl;

  Pt2plICPParameters params2;
  params2.reject = true;
  params2._usedtime.Start();
  auto tgt2 = MakePoints(datat, params2);
  auto src2 = MakePoints(datas, params2);
  auto T2 = Pt2plICPMatch(tgt2, src2, params2);
  cout << "m2: " << params2._ceres_iteration << ", ";
  cout << TransNormRotDegAbsFromAffine2d(Tgt.inverse() * T2).transpose() << endl;

  SICPParameters params3;
  params3.radius = radius;
  params3.reject = true;
  params3._usedtime.Start();
  auto tgt3 = MakePoints(datat, params3);
  auto src3 = MakePoints(datas, params3);
  auto T3 = SICPMatch(tgt3, src3, params3);
  cout << "m3: " << params3._ceres_iteration << ", ";
  cout << TransNormRotDegAbsFromAffine2d(Tgt.inverse() * T3).transpose() << endl;

  D2DNDTParameters params5;
  params5.cell_size = cell_size;
  params5.r_variance = params5.t_variance = 0;
  params5.d2 = 0.05;
  params5._usedtime.Start();
  auto tgt5 = MakeNDTMap(datat, params5);
  auto src5 = MakeNDTMap(datas, params5);
  auto T5 = D2DNDTMatch(tgt5, src5, params5);
  cout << "m5: " << params5._ceres_iteration << ", ";
  cout << TransNormRotDegAbsFromAffine2d(Tgt.inverse() * T5).transpose() << endl;

  D2DNDTParameters params7;
  params7.cell_size = cell_size;
  params7.r_variance = params7.t_variance = 0;
  params7.d2 = 0.05;
  params7.inspect = true;
  params7._usedtime.Start();
  auto tgt7 = MakeNDTMap(datat, params7);
  auto src7 = MakeNDTMap(datas, params7);
  cout << tgt7.ToString() << endl;
  cout << src7.ToString() << endl;
  auto T7 = SNDTMatch2(tgt7, src7, params7);
  params5._usedtime.Show();
  params7._usedtime.Show();
  // for (auto sols : params7._sols) {
  //   for (auto sol : sols) {
  //     cout << XYTDegreeFromAffine2d(sol).transpose() << endl;
  //   }
  //   cout << endl;
  // }
  cout << endl;
  cout << "m7: " << params7._ceres_iteration << ", ";
  cout << TransNormRotDegAbsFromAffine2d(Tgt.inverse() * T7).transpose() << endl;
  for (int i = 0; i < 10; ++i) cout << params5._corres[0][i].first << " - " << params5._corres[0][i].second << ", ";
  cout << endl;
  for (int i = 0; i < 10; ++i) cout << params7._corres[0][i].first << " - " << params7._corres[0][i].second << ", ";
  cout << endl << params5._corres[0].size() << " - " << params7._corres[0].size() << endl;

  TransformPointsInPlace(tgt, aff2);
  TransformPointsInPlace(src, aff2);
  pubt.publish(MarkerOfPoints(tgt, 0.5, Color::kRed));
  // pubs.publish(MarkerOfPoints(src));
  pubs.publish(MarkerOfPoints(TransformPoints(src, T7)));
  pubg.publish(MarkerOfPoints(TransformPoints(src, Tgt)));

  ros::spin();
}