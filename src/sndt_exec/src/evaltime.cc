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

void InitFirstPose(nav_msgs::Path &path, const ros::Time &time) {
  path.header.frame_id = "map";
  path.header.stamp = time;
  path.poses.push_back(MakePoseStampedMsg(time, Eigen::Affine3d::Identity()));
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

  Affine3d Tr3, Tr4, Tr5, Tr5_2, Tr7;
  Tr3.setIdentity(), Tr4.setIdentity(), Tr5.setIdentity(), Tr5_2.setIdentity(), Tr7.setIdentity();
  nav_msgs::Path path3, path4, path5, path5_2, path7;
  InitFirstPose(path3, vpc[0].header.stamp);
  InitFirstPose(path4, vpc[0].header.stamp);
  InitFirstPose(path5, vpc[0].header.stamp);
  InitFirstPose(path5_2, vpc[0].header.stamp);
  InitFirstPose(path7, vpc[0].header.stamp);
  double terr3 = 0, terr4 = 0, terr5 = 0, terr5_2 = 0, terr7 = 0;
  double aerr3 = 0, aerr4 = 0, aerr5 = 0, aerr5_2 = 0, aerr7 = 0;
  vector<double> t3(6), t4(6), t5(6), t5_2(6), t7(6);

  ros::init(argc, argv, "evaltime");
  ros::NodeHandle nh;
  auto pubt = nh.advertise<Marker>("tgt", 0, true);
  auto pubs = nh.advertise<Marker>("src", 0, true);
  auto pubg = nh.advertise<Marker>("gt", 0, true);

  Affine2d Tg3, Tg4, Tg5, Tg5_2, Tg7;
  Tg3.setIdentity(), Tg4.setIdentity(), Tg5.setIdentity(), Tg5_2.setIdentity(), Tg7.setIdentity();
  tqdm bar;
  if (n == -1) n = vpc.size() - 1;
  for (int i = 0; i < n; ++i) {
    bar.progress(i, n);
    auto tgt = PCMsgTo2D(vpc[i], voxel);
    auto src = PCMsgTo2D(vpc[i + 1], voxel);
    auto Tgt =
        GetBenchMark(gtpath, vpc[i].header.stamp, vpc[i + 1].header.stamp);

    vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, aff2}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{{src, aff2}};

    SICPParameters params3;
    params3.radius = radius;
    auto tgt3 = MakePoints(datat, params3);
    auto src3 = MakePoints(datas, params3);
    auto T3 = SICPMatch(tgt3, src3, params3, Tg3);
    Tg3 = T3;
    AddTime(t3, params3._usedtime);
    Tr3 = Tr3 * Affine3dFromAffine2d(T3);
    path3.poses.push_back(MakePoseStampedMsg(vpc[i + 1].header.stamp, Tr3));
    terr3 += TransNormRotDegAbsFromAffine2d(Tgt.inverse() * T3)(0);
    aerr3 += TransNormRotDegAbsFromAffine2d(Tgt.inverse() * T3)(1);

    P2DNDTParameters params4;
    params4.cell_size = cell_size;
    params4.r_variance = params4.t_variance = 0;
    auto tgt4 = MakeNDTMap(datat, params4);
    auto src4 = MakePoints(datas, params4);
    auto T4 = P2DNDTMatch(tgt4, src4, params4);
    Tg4 = T4;
    AddTime(t4, params4._usedtime);
    Tr4 = Tr4 * Affine3dFromAffine2d(T4);
    path4.poses.push_back(MakePoseStampedMsg(vpc[i + 1].header.stamp, Tr4));
    terr4 += TransNormRotDegAbsFromAffine2d(Tgt.inverse() * T4)(0);
    aerr4 += TransNormRotDegAbsFromAffine2d(Tgt.inverse() * T4)(1);

    D2DNDTParameters params5;
    params5.cell_size = cell_size;
    params5.r_variance = params5.t_variance = 0;
    auto tgt5 = MakeNDTMap(datat, params5);
    auto src5 = MakeNDTMap(datas, params5);
    auto T5 = D2DNDTMDMatch(tgt5, src5, params5, Tg5);
    Tg5 = T5;
    AddTime(t5, params5._usedtime);
    Tr5 = Tr5 * Affine3dFromAffine2d(T5);
    path5.poses.push_back(MakePoseStampedMsg(vpc[i + 1].header.stamp, Tr5));
    terr5 += TransNormRotDegAbsFromAffine2d(Tgt.inverse() * T5)(0);
    aerr5 += TransNormRotDegAbsFromAffine2d(Tgt.inverse() * T5)(1);

    D2DNDTParameters params5_2;
    params5_2.cell_size = cell_size;
    params5_2.r_variance = params5_2.t_variance = 0;
    auto tgt5_2 = MakeNDTMap(datat, params5_2);
    auto src5_2 = MakeNDTMap(datas, params5_2);
    auto T5_2 = D2DNDTMatch(tgt5_2, src5_2, params5_2, Tg5_2);
    Tg5_2 = T5_2;
    AddTime(t5_2, params5_2._usedtime);
    Tr5_2 = Tr5_2 * Affine3dFromAffine2d(T5_2);
    path5_2.poses.push_back(MakePoseStampedMsg(vpc[i + 1].header.stamp, Tr5_2));
    terr5_2 += TransNormRotDegAbsFromAffine2d(Tgt.inverse() * T5_2)(0);
    aerr5_2 += TransNormRotDegAbsFromAffine2d(Tgt.inverse() * T5_2)(1);

    D2DNDTParameters params7;
    params7.cell_size = cell_size;
    params7.r_variance = params7.t_variance = 0;
    auto tgt7 = MakeNDTMap(datat, params7);
    auto src7 = MakeNDTMap(datas, params7);
    auto T7 = SNDTMatch2(tgt7, src7, params7, Tg7);
    Tg7 = T7;
    AddTime(t7, params7._usedtime);
    Tr7 = Tr7 * Affine3dFromAffine2d(T7);
    path7.poses.push_back(MakePoseStampedMsg(vpc[i + 1].header.stamp, Tr7));
    terr7 += TransNormRotDegAbsFromAffine2d(Tgt.inverse() * T7)(0);
    aerr7 += TransNormRotDegAbsFromAffine2d(Tgt.inverse() * T7)(1);
  }
  bar.finish();
  cout << "err3: " << terr3 / n << ", " << aerr3 / n << endl;
  cout << "err4: " << terr4 / n << ", " << aerr4 / n << endl;
  cout << "err5: " << terr5 / n << ", " << aerr5 / n << endl;
  cout << "e5_2: " << terr5_2 / n << ", " << aerr5_2 / n << endl;
  cout << "err7: " << terr7 / n << ", " << aerr7 / n << endl;
  for (auto &t : t3) t /= n;
  for (auto &t : t4) t /= n;
  for (auto &t : t5) t /= n;
  for (auto &t : t5_2) t /= n;
  for (auto &t : t7) t /= n;
  printf("3: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n", t3[0], t3[1], t3[2], t3[3], t3[4], t3[5]);
  printf("4: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n", t4[0], t4[1], t4[2], t4[3], t4[4], t4[5]);
  printf("5: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n", t5[0], t5[1], t5[2], t5[3], t5[4], t5[5]);
  printf("5: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n", t5_2[0], t5_2[1], t5_2[2], t5_2[3], t5_2[4], t5_2[5]);
  printf("7: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n", t7[0], t7[1], t7[2], t7[3], t7[4], t7[5]);

  /*
  err3: 0.0447029, -0.00688182
  err5: 0.0471102, -0.00487229
  err7: 0.0397519, -0.00395374
  0.00, 4.35, 36.57, 14.26, 4.54, 59.72
  3.48, 0.00, 3.56, 6.26, 0.84, 14.14
  3.42, 0.00, 4.90, 2.23, 1.04, 11.59
  */
  auto pub3 = nh.advertise<nav_msgs::Path>("path3", 0, true);
  auto pub4 = nh.advertise<nav_msgs::Path>("path4", 0, true);
  auto pub5 = nh.advertise<nav_msgs::Path>("path5", 0, true);
  auto pub5_2 = nh.advertise<nav_msgs::Path>("path5_2", 0, true);
  auto pub7 = nh.advertise<nav_msgs::Path>("path7", 0, true);
  auto pubgt = nh.advertise<nav_msgs::Path>("pathg", 0, true);
  auto pubm = nh.advertise<visualization_msgs::MarkerArray>("marker", 0, true);
  vector<Marker> ms;
  for (int i = 0; i < path5.poses.size(); i += 10) {
    Vector2d p(path5.poses[i].pose.position.x, path5.poses[i].pose.position.y);
    ms.push_back(MarkerOfText(to_string(i), p));
  }
  auto ma = JoinMarkers(ms);
  MakeGtLocal(gtpath, path3.poses[0].header.stamp);
  pubm.publish(ma);
  pub3.publish(path3);
  pub4.publish(path4);
  pub5.publish(path5);
  pub5_2.publish(path5_2);
  pub7.publish(path7);
  pubgt.publish(gtpath);
  ros::spin();
}
