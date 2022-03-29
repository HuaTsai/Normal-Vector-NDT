// auto d2 or assigned d2
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

nav_msgs::Path InitFirstPose(const ros::Time &time) {
  nav_msgs::Path path;
  path.header.frame_id = "map";
  path.header.stamp = time;
  path.poses.push_back(MakePoseStampedMsg(time, Eigen::Affine3d::Identity()));
  return path;
}

double Dist(const nav_msgs::Path &gt, ros::Time t1, ros::Time t2) {
  auto p1 = GetPose(gt.poses, t1);
  auto p2 = GetPose(gt.poses, t2);
  return Vector2d(p1.position.x - p2.position.x, p1.position.y - p2.position.y)
      .norm();
}

void PrintResult(string str,
                 const nav_msgs::Path &est,
                 const nav_msgs::Path &gt) {
  TrajectoryEvaluation te;
  te.set_estpath(est);
  te.set_gtpath(gt);
  te.set_evaltype(TrajectoryEvaluation::EvalType::kAbsolute);
  auto ate = te.ComputeRMSError2D();
  te.set_evaltype(TrajectoryEvaluation::EvalType::kRelativeByLength);
  te.set_length(100);
  auto rpe = te.ComputeRMSError2D();
  printf("len: %.2f, ate: %.2f / %.2f, rpe %.2f / %.2f\n", te.gtlength(),
         ate.first.rms, ate.second.rms, rpe.first.rms, rpe.second.rms);
  cout << " tl: ";
  rpe.first.PrintResult();
  cout << "rot: ";
  rpe.second.PrintResult();
}

void PrintTime(string str,
               const UsedTime &ut,
               int n,
               const vector<double> &its) {
  double den = n * 1000.;
  std::printf(
      "%s: nm: %.2f, ndt: %.2f, bud: %.2f, opt: %.2f, oth: %.2f, ttl: %.2f, "
      "iter: %.2f\n",
      str.c_str(), ut.normal() / den, ut.ndt() / den, ut.build() / den,
      ut.optimize() / den, ut.others() / den, ut.total() / den, Stat(its).mean);
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  string d;
  double c, r, d2, v;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("h,h", "Produce help message")
      ("d,d", po::value<string>(&d)->required(), "Data (logxx)")
      ("v,v", po::value<double>(&v)->default_value(0), "Voxel")
      ("c,c", po::value<double>(&c)->default_value(1.5), "Cell Size")
      ("r,r", po::value<double>(&r)->required(), "Meters to Next Match")
      ("d2", po::value<double>(&d2)->required(), "d2");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  po::notify(vm);

  nav_msgs::Path gtpath;
  SerializationInput(JoinPath(GetDataPath(d), "gt.ser"), gtpath);
  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath(d), "lidar.ser"), vpc);
  int n = vpc.size() - 1;

  cout << "  all n = " << n << endl;
  vector<int> ids{0};
  for (int i = 0, j = 1; i < n && j < n; ++j) {
    if (Dist(gtpath, vpc[i].header.stamp, vpc[j].header.stamp) < r) continue;
    ids.push_back(j);
    i = j;
  }
  cout << "valid n = " << ids.size() << endl;

  auto t0 = vpc[0].header.stamp;
  nav_msgs::Path path1 = InitFirstPose(t0);
  nav_msgs::Path path2 = InitFirstPose(t0);
  Affine3d Tr1 = Affine3d::Identity();
  Affine3d Tr2 = Affine3d::Identity();
  UsedTime ut1, ut2;
  vector<double> its1, its2;

  tqdm bar;
  for (size_t i = 0; i < ids.size() - 1; ++i) {
    bar.progress(i, ids.size());
    auto tgt = PCMsgTo2D(vpc[ids[i]], v);
    auto src = PCMsgTo2D(vpc[ids[i + 1]], v);
    auto tj = vpc[ids[i + 1]].header.stamp;

    vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, aff2}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{{src, aff2}};

    D2DNDTParameters params1;
    params1.cell_size = c;
    params1.r_variance = params1.t_variance = 0;
    params1.d2 = d2;
    params1._usedtime.Start();
    auto tgt1 = MakeNDTMap(datat, params1);
    auto src1 = MakeNDTMap(datas, params1);
    auto T1 = D2DNDTMatch(tgt1, src1, params1);
    ut1 = ut1 + params1._usedtime;
    Tr1 = Tr1 * Affine3dFromAffine2d(T1);
    path1.poses.push_back(MakePoseStampedMsg(tj, Tr1));
    its1.push_back(params1._ceres_iteration);
    
    D2DNDTParameters params2;
    params2.cell_size = c;
    params2.r_variance = params2.t_variance = 0;
    params2.d2 = -1;
    params2._usedtime.Start();
    auto tgt2 = MakeNDTMap(datat, params2);
    auto src2 = MakeNDTMap(datas, params2);
    auto T2 = D2DNDTMatch(tgt2, src2, params2);
    ut2 = ut2 + params2._usedtime;
    Tr2 = Tr2 * Affine3dFromAffine2d(T2);
    path2.poses.push_back(MakePoseStampedMsg(tj, Tr2));
    its2.push_back(params2._ceres_iteration);
  }
  bar.finish();

  ros::init(argc, argv, "evaltime");
  ros::NodeHandle nh;
  auto pub1 = nh.advertise<nav_msgs::Path>("path1", 0, true);
  auto pub2 = nh.advertise<nav_msgs::Path>("path2", 0, true);
  auto pubgt = nh.advertise<nav_msgs::Path>("pathg", 0, true);

  MakeGtLocal(gtpath, t0);
  pub1.publish(path1);
  pub2.publish(path2);
  pubgt.publish(gtpath);
  PrintResult("method before", path1, gtpath);
  PrintResult("method after", path2, gtpath);
  PrintTime("method before", ut1, ids.size() - 1, its1);
  PrintTime("method after", ut2, ids.size() - 1, its2);
  ros::spin();
}