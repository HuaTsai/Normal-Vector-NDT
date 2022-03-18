// Experiment 1:
//   topic: gt meters vs. iterations
//   methods: NDT, SNDT
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

void PrintData(string str, const vector<double> &data) {
  printf("%s = np.array([", str.c_str());
  for (auto d : data) printf("%.2f, ", d);
  printf("])\n");
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
  // printf("tl = np.array([");
  // for (auto d : rpe.first.data) printf("%.2f, ", d);
  // printf("])\nrot = np.array([");
  // for (auto d : rpe.second.data) printf("%.2f, ", d);
  // printf("])\n");
}

Affine2d GetBenchMark(const nav_msgs::Path &gtpath, const ros::Time &t1, const ros::Time &t2) {
  Affine3d To, Ti;
  tf2::fromMsg(GetPose(gtpath.poses, t2), To);
  tf2::fromMsg(GetPose(gtpath.poses, t1), Ti);
  Affine3d Tgt3 = Conserve2DFromAffine3d(Ti.inverse() * To);
  Affine2d ret = Translation2d(Tgt3.translation()(0), Tgt3.translation()(1)) *
                 Rotation2Dd(Tgt3.rotation().block<2, 2>(0, 0));
  return ret;
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
  printf("ndt = [%.2f, %.2f, %.2f, %.2f, %.2f]\n", Stat(a.bud).mean / 1000.,
         Stat(a.nm).mean / 1000., Stat(a.ndt).mean / 1000.,
         Stat(a.opt).mean / 1000., Stat(a.oth).mean / 1000.);
  printf("sndt = [%.2f, %.2f, %.2f, %.2f, %.2f]\n", Stat(b.bud).mean / 1000.,
         Stat(b.nm).mean / 1000., Stat(b.ndt).mean / 1000.,
         Stat(b.opt).mean / 1000., Stat(b.oth).mean / 1000.);
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  bool tr;
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
      ("d2", po::value<double>(&d2)->required(), "d2")
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
  Res r5, r7;
  r5.path = InitFirstPose(t0);
  r7.path = InitFirstPose(t0);
  vector<double> e5t, e5r, e7t, e7r;
  MakeGtLocal(gtpath, t0);

  tqdm bar;
  for (size_t i = 0; i < ids.size() - 1; ++i) {
    bar.progress(i, ids.size());
    auto tgt = PCMsgTo2D(vpc[ids[i]], v);
    auto src = PCMsgTo2D(vpc[ids[i + 1]], v);
    auto tj = vpc[ids[i + 1]].header.stamp;
    auto ben = GetBenchMark(gtpath, vpc[ids[i]].header.stamp, vpc[ids[i + 1]].header.stamp).inverse();

    vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, aff2}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{{src, aff2}};

    if (c != 0) {
      D2DNDTParameters params5;
      params5.reject = true;
      params5.cell_size = c;
      params5.d2 = d2;
      params5._usedtime.Start();
      auto tgt5 = MakeNDT(datat, params5);
      auto src5 = MakeNDT(datas, params5);
      Affine2d T5;
      if (tr)
        T5 = DMatch(tgt5, src5, params5);
      else
        T5 = D2DNDTMatch(tgt5, src5, params5);
      e5t.push_back(TransNormRotDegAbsFromAffine2d(ben * T5)(0));
      e5r.push_back(TransNormRotDegAbsFromAffine2d(ben * T5)(1));
      Updates(params5, r5, tj, T5);

      D2DNDTParameters params7;
      params7.reject = true;
      params7.cell_size = c;
      params7.d2 = d2;
      params7._usedtime.Start();
      auto tgt7 = MakeNDT(datat, params7);
      auto src7 = MakeNDT(datas, params7);
      Affine2d T7;
      if (tr)
        T7 = SMatch(tgt7, src7, params7);
      else
        T7 = SNDTMatch2(tgt7, src7, params7);
      e7t.push_back(TransNormRotDegAbsFromAffine2d(ben * T7)(0));
      e7r.push_back(TransNormRotDegAbsFromAffine2d(ben * T7)(1));
      Updates(params7, r7, tj, T7);
    } else {
      Affine2d T5 = Affine2d::Identity();
      for (double cs : {4., 2., 1.}) {
        D2DNDTParameters params5;
        params5.cell_size = cs;
        params5.d2 = d2;
        params5._usedtime.Start();
        auto tgt5 = MakeNDT(datat, params5);
        auto src5 = MakeNDT(datas, params5);
        T5 = D2DNDTMatch(tgt5, src5, params5, T5);
      }

      Affine2d T7 = Affine2d::Identity();
      for (double cs : {4., 2., 1.}) {
        D2DNDTParameters params7;
        params7.cell_size = cs;
        params7.d2 = d2;
        params7._usedtime.Start();
        auto tgt7 = MakeNDTMap(datat, params7);
        auto src7 = MakeNDTMap(datas, params7);
        T7 = SNDTMatch2(tgt7, src7, params7, T7);
      }
    }
  }
  bar.finish();
  Stat(e5t).PrintResult();// PrintData("ndttl", e5t);
  Stat(e5r).PrintResult();// PrintData("ndtrot", e5r);
  Stat(e7t).PrintResult();// PrintData("sndttl", e7t);
  Stat(e7r).PrintResult();// PrintData("sndtrot", e7r);
  PrintTime(r5, r7);

  ros::init(argc, argv, "exp1");
  ros::NodeHandle nh;
  auto pub5 = nh.advertise<nav_msgs::Path>("path5", 0, true);
  auto pub7 = nh.advertise<nav_msgs::Path>("path7", 0, true);
  auto pubgt = nh.advertise<nav_msgs::Path>("pathg", 0, true);

  pub5.publish(r5.path);
  pub7.publish(r7.path);
  pubgt.publish(gtpath);
  PrintResult(r5.path, gtpath);
  PrintResult(r7.path, gtpath);
  r5.Print();
  r7.Print();
  ros::spin();
}
