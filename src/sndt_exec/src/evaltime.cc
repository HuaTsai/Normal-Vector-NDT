#include <bits/stdc++.h>
#include <common/common.h>
#include <metric/metric.h>
#include <sndt/visuals.h>
#include <sndt_exec/wrapper.h>
#include <tqdm/tqdm.h>

#include <boost/program_options.hpp>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

int n;

void PrintTime(string str, const UsedTime &ut) {
  double den = (n - 1) * 1000.;
  std::printf("%s: nm: %.2f, ndt: %.2f, bud: %.2f, opt: %.2f, oth: %.2f, ttl: %.2f\n",
              str.c_str(), ut.normal() / den, ut.ndt() / den, ut.build() / den,
              ut.optimize() / den, ut.others() / den, ut.total() / den);
}

nav_msgs::Path InitFirstPose(const ros::Time &time) {
  nav_msgs::Path path;
  path.header.frame_id = "map";
  path.header.stamp = time;
  path.poses.push_back(MakePoseStampedMsg(time, Eigen::Affine3d::Identity()));
  return path;
}

void PrintResult(string str,
                 const nav_msgs::Path &est,
                 const nav_msgs::Path &gt) {
  TrajectoryEvaluation te;
  te.set_estpath(est);
  te.set_gtpath(gt);

  te.set_evaltype(TrajectoryEvaluation::EvalType::kAbsolute);
  auto ate = te.ComputeRMSError2D();
  te.set_evaltype(TrajectoryEvaluation::EvalType::kRelativeBySingle);
  // te.set_evaltype(TrajectoryEvaluation::EvalType::kRelativeByLength);
  te.set_length(100);
  auto rpe = te.ComputeRMSError2D();
  cout << str << ": len " << te.gtlength() << " ate " << ate.first.rms << ", "
       << ate.second.rms << ", rpe " << rpe.first.rms << ", " << rpe.second.rms
       << endl;
  cout << " tl: ";
  rpe.first.PrintResult();
  cout << "rot: ";
  rpe.second.PrintResult();
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  vector<int> m;
  double cell_size, voxel, radius = 1.5, d2 = -1;
  string data;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("n,n", po::value<int>(&n)->default_value(-1), "To where")
      ("m,m", po::value<vector<int>>(&m)->required()->multitoken(), "Method")
      ("d2", po::value<double>(&d2), "d2");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  po::notify(vm);
  unordered_set<int> ms(m.begin(), m.end());
  if ((ms.count(4) || ms.count(5) || ms.count(6) || ms.count(7)) && d2 == -1) {
    cout << "d2 required!" << endl;
    return 1;
  }

  nav_msgs::Path gtpath;
  SerializationInput(JoinPath(GetDataPath(data), "gt.ser"), gtpath);
  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath(data), "lidar.ser"), vpc);

  unordered_map<int, Affine3d> Tr{
      {1, Affine3d::Identity()},  {2, Affine3d::Identity()},
      {3, Affine3d::Identity()},  {4, Affine3d::Identity()},
      {42, Affine3d::Identity()}, {5, Affine3d::Identity()},
      {52, Affine3d::Identity()}, {6, Affine3d::Identity()},
      {62, Affine3d::Identity()}, {7, Affine3d::Identity()},
      {72, Affine3d::Identity()}};
  auto t0 = vpc[0].header.stamp;
  unordered_map<int, nav_msgs::Path> path{
      {1, InitFirstPose(t0)},  {2, InitFirstPose(t0)},  {3, InitFirstPose(t0)},
      {4, InitFirstPose(t0)},  {42, InitFirstPose(t0)}, {5, InitFirstPose(t0)},
      {52, InitFirstPose(t0)}, {6, InitFirstPose(t0)},  {62, InitFirstPose(t0)},
      {7, InitFirstPose(t0)},  {72, InitFirstPose(t0)}};
  unordered_map<int, UsedTime> usedtime{
      {1, UsedTime()},  {2, UsedTime()}, {3, UsedTime()},  {4, UsedTime()},
      {42, UsedTime()}, {5, UsedTime()}, {52, UsedTime()}, {6, UsedTime()},
      {62, UsedTime()}, {7, UsedTime()}, {72, UsedTime()}};
  unordered_map<int, vector<double>> its{{1, {}},  {2, {}}, {3, {}},  {4, {}},
                                         {42, {}}, {5, {}}, {52, {}}, {6, {}},
                                         {62, {}}, {7, {}}, {72, {}}};

  ros::init(argc, argv, "evaltime");
  ros::NodeHandle nh;

  tqdm bar;
  if (n == -1) n = vpc.size() - 1;
  for (int i = 0; i < n; ++i) {
    bar.progress(i, n);
    auto tgt = PCMsgTo2D(vpc[i], voxel);
    auto src = PCMsgTo2D(vpc[i + 1], voxel);
    auto tj = vpc[i + 1].header.stamp;
  // for (int i = 0; i + 11 < n; i += 10) {
  //   bar.progress(i, n);
  //   auto tgt = PCMsgTo2D(vpc[i], voxel);
  //   auto src = PCMsgTo2D(vpc[i + 10], voxel);
  //   auto tj = vpc[i + 10].header.stamp;

    vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, aff2}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{{src, aff2}};

    if (ms.count(1)) {
      ICPParameters params1;
      params1.reject = true;
      params1._usedtime.Start();
      auto tgt1 = MakePoints(datat, params1);
      auto src1 = MakePoints(datas, params1);
      auto T1 = ICPMatch(tgt1, src1, params1);
      usedtime[1] = usedtime[1] + params1._usedtime;
      Tr[1] = Tr[1] * Affine3dFromAffine2d(T1);
      path[1].poses.push_back(MakePoseStampedMsg(tj, Tr[1]));
      its[1].push_back(params1._ceres_iteration);
    }

    if (ms.count(2)) {
      Pt2plICPParameters params2;
      params2.reject = true;
      params2._usedtime.Start();
      auto tgt2 = MakePoints(datat, params2);
      auto src2 = MakePoints(datas, params2);
      auto T2 = Pt2plICPMatch(tgt2, src2, params2);
      usedtime[2] = usedtime[2] + params2._usedtime;
      Tr[2] = Tr[2] * Affine3dFromAffine2d(T2);
      path[2].poses.push_back(MakePoseStampedMsg(tj, Tr[2]));
      its[2].push_back(params2._ceres_iteration);
    }

    if (ms.count(3)) {
      SICPParameters params3;
      params3.radius = radius;
      params3.reject = true;
      params3._usedtime.Start();
      auto tgt3 = MakePoints(datat, params3);
      auto src3 = MakePoints(datas, params3);
      auto T3 = SICPMatch(tgt3, src3, params3);
      usedtime[3] = usedtime[3] + params3._usedtime;
      Tr[3] = Tr[3] * Affine3dFromAffine2d(T3);
      path[3].poses.push_back(MakePoseStampedMsg(tj, Tr[3]));
      its[3].push_back(params3._ceres_iteration);
    }

    if (ms.count(4)) {
      P2DNDTParameters params4;
      params4.cell_size = cell_size;
      params4.r_variance = params4.t_variance = 0;
      params4.d2 = d2;
      params4._usedtime.Start();
      auto tgt4 = MakeNDTMap(datat, params4);
      auto src4 = MakePoints(datas, params4);
      auto T4 = P2DNDTMatch(tgt4, src4, params4);
      usedtime[4] = usedtime[4] + params4._usedtime;
      Tr[4] = Tr[4] * Affine3dFromAffine2d(T4);
      path[4].poses.push_back(MakePoseStampedMsg(tj, Tr[4]));
      its[4].push_back(params4._ceres_iteration);
    }

    if (ms.count(42)) {
      P2DNDTParameters params4;
      params4.cell_size = cell_size;
      params4.r_variance = params4.t_variance = 0;
      params4.reject = true;
      params4._usedtime.Start();
      auto tgt4 = MakeNDTMap(datat, params4);
      auto src4 = MakePoints(datas, params4);
      auto T4 = P2DNDTMDMatch(tgt4, src4, params4);
      usedtime[42] = usedtime[42] + params4._usedtime;
      Tr[42] = Tr[42] * Affine3dFromAffine2d(T4);
      path[42].poses.push_back(MakePoseStampedMsg(tj, Tr[42]));
      its[4].push_back(params4._ceres_iteration);
    }

    if (ms.count(5)) {
      D2DNDTParameters params5;
      params5.cell_size = cell_size;
      params5.r_variance = params5.t_variance = 0;
      params5.d2 = d2;
      params5._usedtime.Start();
      auto tgt5 = MakeNDTMap(datat, params5);
      auto src5 = MakeNDTMap(datas, params5);
      auto T5 = D2DNDTMatch(tgt5, src5, params5);
      usedtime[5] = usedtime[5] + params5._usedtime;
      Tr[5] = Tr[5] * Affine3dFromAffine2d(T5);
      path[5].poses.push_back(MakePoseStampedMsg(tj, Tr[5]));
      its[5].push_back(params5._ceres_iteration);
    }

    if (ms.count(52)) {
      D2DNDTParameters params5;
      params5.cell_size = cell_size;
      params5.r_variance = params5.t_variance = 0;
      params5.reject = true;
      // params5.d2 = d2;
      params5._usedtime.Start();
      auto tgt5 = MakeNDTMap(datat, params5);
      auto src5 = MakeNDTMap(datas, params5);
      auto T5 = D2DNDTMDMatch(tgt5, src5, params5);
      usedtime[52] = usedtime[52] + params5._usedtime;
      Tr[52] = Tr[52] * Affine3dFromAffine2d(T5);
      path[52].poses.push_back(MakePoseStampedMsg(tj, Tr[52]));
      its[52].push_back(params5._ceres_iteration);
    }

    if (ms.count(6)) {
      SNDTParameters params6;
      params6.cell_size = cell_size;
      params6.r_variance = params6.t_variance = 0;
      params6.d2 = d2;
      params6._usedtime.Start();
      auto tgt6 = MakeSNDTMap(datat, params6);
      auto src6 = MakeSNDTMap(datas, params6);
      auto T6 = SNDTMatch(tgt6, src6, params6);
      usedtime[6] = usedtime[6] + params6._usedtime;
      Tr[6] = Tr[6] * Affine3dFromAffine2d(T6);
      path[6].poses.push_back(MakePoseStampedMsg(tj, Tr[6]));
      its[6].push_back(params6._ceres_iteration);
    }

    if (ms.count(62)) {
      SNDTParameters params6;
      params6.cell_size = cell_size;
      params6.r_variance = params6.t_variance = 0;
      params6.reject = true;
      params6._usedtime.Start();
      auto tgt6 = MakeSNDTMap(datat, params6);
      auto src6 = MakeSNDTMap(datas, params6);
      auto T6 = SNDTMDMatch(tgt6, src6, params6);
      usedtime[62] = usedtime[62] + params6._usedtime;
      Tr[62] = Tr[62] * Affine3dFromAffine2d(T6);
      path[62].poses.push_back(MakePoseStampedMsg(tj, Tr[62]));
      its[62].push_back(params6._ceres_iteration);
    }

    if (ms.count(7)) {
      D2DNDTParameters params7;
      params7.cell_size = cell_size;
      params7.r_variance = params7.t_variance = 0;
      params7.d2 = d2;
      params7._usedtime.Start();
      auto tgt7 = MakeNDTMap(datat, params7);
      auto src7 = MakeNDTMap(datas, params7);
      auto T7 = SNDTMatch2(tgt7, src7, params7);
      usedtime[7] = usedtime[7] + params7._usedtime;
      Tr[7] = Tr[7] * Affine3dFromAffine2d(T7);
      path[7].poses.push_back(MakePoseStampedMsg(tj, Tr[7]));
      its[7].push_back(params7._ceres_iteration);
    }

    if (ms.count(72)) {
      D2DNDTParameters params7;
      params7.cell_size = cell_size;
      params7.r_variance = params7.t_variance = 0;
      params7.reject = true;
      // params7.d2 = d2;
      params7._usedtime.Start();
      auto tgt7 = MakeNDTMap(datat, params7);
      auto src7 = MakeNDTMap(datas, params7);
      auto T7 = SNDTMDMatch2(tgt7, src7, params7);
      usedtime[72] = usedtime[72] + params7._usedtime;
      Tr[72] = Tr[72] * Affine3dFromAffine2d(T7);
      path[72].poses.push_back(MakePoseStampedMsg(tj, Tr[72]));
      its[72].push_back(params7._ceres_iteration);
    }
  }
  bar.finish();

  for (int i = 0; i < its[1].size(); ++i) {
    cout << i << ": " << its[1][i] - its[2][i] << endl;
  }

  auto pub1 = nh.advertise<nav_msgs::Path>("path1", 0, true);
  auto pub2 = nh.advertise<nav_msgs::Path>("path2", 0, true);
  auto pub3 = nh.advertise<nav_msgs::Path>("path3", 0, true);
  auto pub4 = nh.advertise<nav_msgs::Path>("path4", 0, true);
  auto pub42 = nh.advertise<nav_msgs::Path>("path42", 0, true);
  auto pub5 = nh.advertise<nav_msgs::Path>("path5", 0, true);
  auto pub52 = nh.advertise<nav_msgs::Path>("path52", 0, true);
  auto pub6 = nh.advertise<nav_msgs::Path>("path6", 0, true);
  auto pub62 = nh.advertise<nav_msgs::Path>("path62", 0, true);
  auto pub7 = nh.advertise<nav_msgs::Path>("path7", 0, true);
  auto pub72 = nh.advertise<nav_msgs::Path>("path72", 0, true);
  auto pubgt = nh.advertise<nav_msgs::Path>("pathg", 0, true);

  MakeGtLocal(gtpath, t0);
  if (ms.count(1)) pub1.publish(path[1]);
  if (ms.count(2)) pub2.publish(path[2]);
  if (ms.count(3)) pub3.publish(path[3]);
  if (ms.count(4)) pub4.publish(path[4]);
  if (ms.count(42)) pub42.publish(path[42]);
  if (ms.count(5)) pub5.publish(path[5]);
  if (ms.count(52)) pub52.publish(path[52]);
  if (ms.count(6)) pub6.publish(path[6]);
  if (ms.count(62)) pub62.publish(path[62]);
  if (ms.count(7)) pub7.publish(path[7]);
  if (ms.count(72)) pub72.publish(path[72]);
  pubgt.publish(gtpath);

  if (ms.count(1)) PrintResult("method 1", path[1], gtpath);
  if (ms.count(2)) PrintResult("method 2", path[2], gtpath);
  if (ms.count(3)) PrintResult("method 3", path[3], gtpath);
  if (ms.count(4)) PrintResult("method 4", path[4], gtpath);
  if (ms.count(42)) PrintResult("method 42", path[42], gtpath);
  if (ms.count(5)) PrintResult("method 5", path[5], gtpath);
  if (ms.count(52)) PrintResult("method 52", path[52], gtpath);
  if (ms.count(6)) PrintResult("method 6", path[6], gtpath);
  if (ms.count(62)) PrintResult("method 62", path[62], gtpath);
  if (ms.count(7)) PrintResult("method 7", path[7], gtpath);
  if (ms.count(72)) PrintResult("method 72", path[72], gtpath);

  if (ms.count(1)) PrintTime("method 1", usedtime[1]);
  if (ms.count(2)) PrintTime("method 2", usedtime[2]);
  if (ms.count(3)) PrintTime("method 3", usedtime[3]);
  if (ms.count(4)) PrintTime("method 4", usedtime[4]);
  if (ms.count(42)) PrintTime("method 42", usedtime[42]);
  if (ms.count(5)) PrintTime("method 5", usedtime[5]);
  if (ms.count(52)) PrintTime("method 52", usedtime[52]);
  if (ms.count(6)) PrintTime("method 6", usedtime[6]);
  if (ms.count(62)) PrintTime("method 62", usedtime[62]);
  if (ms.count(7)) PrintTime("method 7", usedtime[7]);
  if (ms.count(72)) PrintTime("method 72", usedtime[72]);

  if (ms.count(1)) cout << "method 1: " << Average(its[1]) << endl;
  if (ms.count(2)) cout << "method 2: " << Average(its[2]) << endl;
  if (ms.count(3)) cout << "method 3: " << Average(its[3]) << endl;
  if (ms.count(4)) cout << "method 4: " << Average(its[4]) << endl;
  if (ms.count(42)) cout << "method 42: " << Average(its[42]) << endl;
  if (ms.count(5)) cout << "method 5: " << Average(its[5]) << endl;
  if (ms.count(52)) cout << "method 52: " << Average(its[52]) << endl;
  if (ms.count(6)) cout << "method 6: " << Average(its[6]) << endl;
  if (ms.count(62)) cout << "method 62: " << Average(its[62]) << endl;
  if (ms.count(7)) cout << "method 7: " << Average(its[7]) << endl;
  if (ms.count(72)) cout << "method 72: " << Average(its[72]) << endl;

  ros::spin();
}