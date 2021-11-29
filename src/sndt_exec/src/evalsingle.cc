#include <bits/stdc++.h>
#include <common/common.h>
#include <sndt_exec/wrapper.h>
#include <tqdm/tqdm.h>
#include <boost/program_options.hpp>
#include <sndt/visuals.h>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

void AddTime(vector<double> &t, const UsedTime &time) {
  t[0] += time.ndt / 1000.;
  t[1] += time.normal / 1000.;
  t[2] += time.build / 1000.;
  t[3] += time.optimize / 1000.;
  t[4] += time.others / 1000.;
  t[5] += time.total() / 1000.;
}

vector<double> CompactTime(const UsedTime &time) {
  vector<double> ret;
  ret.push_back(time.ndt / 1000.);
  ret.push_back(time.normal / 1000.);
  ret.push_back(time.build / 1000.);
  ret.push_back(time.optimize / 1000.);
  ret.push_back(time.others / 1000.);
  ret.push_back(time.total() / 1000.);
  return ret;
}

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

  int n, m;
  double cell_size, voxel, radius = 1.5;
  string data;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("n,n", po::value<int>(&n)->default_value(-1), "To where")
      ("m,m", po::value<int>(&m)->default_value(0), "Method");
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
  Affine3d Tr3, Tr5, Tr7;
  Tr3.setIdentity(), Tr5.setIdentity(), Tr7.setIdentity();

  ros::init(argc, argv, "evalsingle");
  ros::NodeHandle nh;
  auto pubt = nh.advertise<Marker>("tgt", 0, true);
  auto pubs = nh.advertise<Marker>("src", 0, true);
  auto pubg = nh.advertise<Marker>("gt", 0, true);
  auto pub1 = nh.advertise<MarkerArray>("markers1", 0, true);
  auto pub2 = nh.advertise<MarkerArray>("markers2", 0, true);
  auto pub3 = nh.advertise<MarkerArray>("markers3", 0, true);
  auto pub4 = nh.advertise<MarkerArray>("markers4", 0, true);
  
  if (n == -1) return 1;
  auto tgt = PCMsgTo2D(vpc[n], voxel);
  auto src = PCMsgTo2D(vpc[n + 1], voxel);
  auto Tgt = GetBenchMark(gtpath, vpc[n].header.stamp, vpc[n + 1].header.stamp);

  vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, aff2}};
  vector<pair<vector<Vector2d>, Affine2d>> datas{{src, aff2}};

  Affine2d T = Affine2d::Identity();
  if (m == 3) {
    SICPParameters params3;
    params3.radius = radius;
    auto tgt3 = MakePoints(datat, params3);
    auto src3 = MakePoints(datas, params3);
    T = SICPMatch(tgt3, src3, params3);
  } else if (m == 5) {

    // vector<double> cs = {4, 3, 2, 1, 0.5};
    // for (auto c : cs) {
    //   D2DNDTParameters params5;
    //   params5.cell_size = c;
    //   params5.r_variance = params5.t_variance = 0;
    //   auto tgt5 = MakeNDTMap(datat, params5);
    //   auto src5 = MakeNDTMap(datas, params5);
    //   T = D2DNDTMDMatch(tgt5, src5, params5, T);
    //   cout << XYTDegreeFromAffine2d(T).transpose() << endl;
    //   if (c == cs.back()) {
    //     pub1.publish(MarkerArrayOfNDTMap(tgt5, true));
    //     pub2.publish(MarkerArrayOfNDTMap(src5.PseudoTransformCells(T)));
        // pub3.publish(MarkerArrayOfNDTMap(src5.PseudoTransformCells(Tgt)));
    //     pub4.publish(MarkerArrayOfCorres(src5, tgt5, params5._sols.back().front(), params5._corres.back()));
    //     pub3.publish(MarkerArrayOfCorres(src5, tgt5, params5._sols.back().back(), params5._corres.back()));
    //   }
    // }

    D2DNDTParameters params5;
    params5.cell_size = cell_size;
    params5.r_variance = params5.t_variance = 0;
    auto tgt5 = MakeNDTMap(datat, params5);
    auto src5 = MakeNDTMap(datas, params5);
    T = D2DNDTMDMatch(tgt5, src5, params5, T);
    pub1.publish(MarkerArrayOfNDTMap(tgt5, true));
    pub2.publish(MarkerArrayOfNDTMap(src5.PseudoTransformCells(T)));
    pub3.publish(MarkerArrayOfCorres(src5, tgt5, params5._sols[0].front(), params5._corres[0]));
    pub4.publish(MarkerArrayOfCorres(src5, tgt5, params5._sols.back().back(), params5._corres.back()));

  } else if (m == 7) {
    D2DNDTParameters params7;
    params7.cell_size = cell_size;
    params7.r_variance = params7.t_variance = 0;
    auto tgt7 = MakeNDTMap(datat, params7);
    auto src7 = MakeNDTMap(datas, params7);
    T = SNDTMDMatch2(tgt7, src7, params7);
  }

  cout << XYTDegreeFromAffine2d(Tgt.inverse() * T).transpose() << endl;

  TransformPointsInPlace(tgt, aff2);
  TransformPointsInPlace(src, aff2);
  pubt.publish(MarkerOfPoints(tgt, 0.5, Color::kRed));
  pubs.publish(MarkerOfPoints(TransformPoints(src, T)));
  pubg.publish(MarkerOfPoints(TransformPoints(src, Tgt)));

  ros::spin();
}