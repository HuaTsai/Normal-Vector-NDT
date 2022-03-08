// On road test: icp vs. sicp
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

struct Res {
  Res() : Tr(Affine2d::Identity()) {}
  vector<double> it, iit, iiit, bud, nm, ndt, opt, oth, ttl;
  nav_msgs::Path path;
  Affine2d Tr;
  void Print() {
    printf("%.2f & %.2f & %.2f & %.2f & %.2f\n", Stat(it).mean, Stat(iit).mean,
           Stat(iiit).mean, Stat(opt).mean / 1000., Stat(ttl).mean / 1000.);
  }
};

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

void PrintResult(const nav_msgs::Path &est, const nav_msgs::Path &gt) {
  TrajectoryEvaluation te;
  te.set_estpath(est);
  te.set_gtpath(gt);
  te.set_evaltype(TrajectoryEvaluation::EvalType::kAbsolute);
  auto ate = te.ComputeRMSError2D();
  te.set_evaltype(TrajectoryEvaluation::EvalType::kRelativeByLength);
  te.set_length(100);
  auto rpe = te.ComputeRMSError2D();
  printf("ate: %.2f & %.2f\n", ate.first.rms, ate.second.rms);
  printf("rpe: %.2f & %.2f\n", rpe.first.rms, rpe.second.rms);
  printf("   tl: %.2f / %.2f / %.2f\n", rpe.first.min, rpe.first.max,
         rpe.first.mean);
  printf("  rot: %.2f / %.2f / %.2f\n", rpe.second.min, rpe.second.max,
         rpe.second.mean);
}

void Updates(const CommonParameters &params,
             Res &res,
             ros::Time tj,
             Affine2d T) {
  res.it.push_back(params._iteration);
  res.iit.push_back(params._ceres_iteration);
  res.iiit.push_back(params._search_iteration);
  res.bud.push_back(params._usedtime.build());
  res.nm.push_back(params._usedtime.normal());
  res.ndt.push_back(params._usedtime.ndt());
  res.opt.push_back(params._usedtime.optimize());
  res.oth.push_back(params._usedtime.others());
  res.ttl.push_back(params._usedtime.total());
  res.Tr = res.Tr * T;
  res.path.poses.push_back(
      MakePoseStampedMsg(tj, Affine3dFromAffine2d(res.Tr)));
}

void PrintTime(const Res &a, const Res &b) {
  printf("icp = [%.2f, %.2f, %.2f, %.2f, %.2f]\n", Stat(a.bud).mean / 1000.,
         Stat(a.nm).mean / 1000., Stat(a.ndt).mean / 1000.,
         Stat(a.opt).mean / 1000., Stat(a.oth).mean / 1000.);
  printf("sicp = [%.2f, %.2f, %.2f, %.2f, %.2f]\n", Stat(b.bud).mean / 1000.,
         Stat(b.nm).mean / 1000., Stat(b.ndt).mean / 1000.,
         Stat(b.opt).mean / 1000., Stat(b.oth).mean / 1000.);
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  string d;
  double r, v;
  bool tr;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("h,h", "Produce help message")
      ("d,d", po::value<string>(&d)->required(), "Data (logxx)")
      ("v,v", po::value<double>(&v)->default_value(0), "Voxel")
      ("r,r", po::value<double>(&r)->required(), "Meters to Next Match")
      ("tr", po::value<bool>(&tr)->default_value(false)->implicit_value(true), "Trust Region");
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

  vector<int> ids{0};
  for (int i = 0, j = 1; i < n && j < n; ++j) {
    if (Dist(gtpath, vpc[i].header.stamp, vpc[j].header.stamp) < r) continue;
    ids.push_back(j);
    i = j;
  }
  cout << n << " -> " << ids.size() << endl;

  auto t0 = vpc[0].header.stamp;
  Res r1, r3;
  r1.path = InitFirstPose(t0);
  r3.path = InitFirstPose(t0);
  vector<double> e1t, e1r, e3t, e3r;

  tqdm bar;
  for (size_t i = 0; i < ids.size() - 1; ++i) {
    bar.progress(i, ids.size());
    auto tgt = PCMsgTo2D(vpc[ids[i]], v);
    auto src = PCMsgTo2D(vpc[ids[i + 1]], v);
    auto tj = vpc[ids[i + 1]].header.stamp;

    vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, aff2}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{{src, aff2}};

    // ICP Method
    ICPParameters params1;
    params1.reject = true;
    params1._usedtime.Start();
    auto tgt1 = MakePoints(datat, params1);
    auto src1 = MakePoints(datas, params1);
    Affine2d T1;
    if (tr)
      T1 = ICPMatch(tgt1, src1, params1);
    else
      T1 = IMatch(tgt1, src1, params1);
    e1t.push_back(TransNormRotDegAbsFromAffine2d(T1)(0));
    e1r.push_back(TransNormRotDegAbsFromAffine2d(T1)(1));
    Updates(params1, r1, tj, T1);

    // Symmetric ICP Method
    SICPParameters params3;
    params3.reject = true;
    params3._usedtime.Start();
    auto tgt3 = MakePoints(datat, params3);
    auto src3 = MakePoints(datas, params3);
    Affine2d T3;
    if (tr)
      T3 = SICPMatch(tgt3, src3, params3);
    else
      T3 = PMatch(tgt3, src3, params3);
    e3t.push_back(TransNormRotDegAbsFromAffine2d(T3)(0));
    e3r.push_back(TransNormRotDegAbsFromAffine2d(T3)(1));
    Updates(params3, r3, tj, T3);
  }
  bar.finish();
  Stat(e1t).PrintResult();
  Stat(e1r).PrintResult();
  Stat(e3t).PrintResult();
  Stat(e3r).PrintResult();
  PrintTime(r1, r3);

  ros::init(argc, argv, "exp1_2");
  ros::NodeHandle nh;
  auto pub1 = nh.advertise<nav_msgs::Path>("path1", 0, true);
  auto pub3 = nh.advertise<nav_msgs::Path>("path3", 0, true);
  auto pubgt = nh.advertise<nav_msgs::Path>("pathg", 0, true);

  MakeGtLocal(gtpath, t0);
  pub1.publish(r1.path);
  pub3.publish(r3.path);
  pubgt.publish(gtpath);
  PrintResult(r1.path, gtpath);
  PrintResult(r3.path, gtpath);
  r1.Print();
  r3.Print();
  ros::spin();
}
