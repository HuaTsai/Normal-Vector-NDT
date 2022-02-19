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
      ut.optimize() / den, ut.others() / den, ut.total() / den, Average(its));
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
  nav_msgs::Path path5 = InitFirstPose(t0);
  nav_msgs::Path path7 = InitFirstPose(t0);
  Affine3d Tr5 = Affine3d::Identity();
  Affine3d Tr7 = Affine3d::Identity();
  UsedTime ut5, ut7;
  vector<double> its5, its7;

  tqdm bar;
  for (int i = 0; i < ids.size() - 1; ++i) {
    bar.progress(i, ids.size());
    auto tgt = PCMsgTo2D(vpc[ids[i]], v);
    auto src = PCMsgTo2D(vpc[ids[i + 1]], v);
    auto tj = vpc[ids[i + 1]].header.stamp;

    vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, aff2}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{{src, aff2}};

    if (c != 0) {
      D2DNDTParameters params5;
      params5.cell_size = c;
      params5.r_variance = params5.t_variance = 0;
      params5.d2 = d2;
      params5._usedtime.Start();
      auto tgt5 = MakeNDTMap(datat, params5);
      auto src5 = MakeNDTMap(datas, params5);
      auto T5 = D2DNDTMatch(tgt5, src5, params5);
      ut5 = ut5 + params5._usedtime;
      Tr5 = Tr5 * Affine3dFromAffine2d(T5);
      path5.poses.push_back(MakePoseStampedMsg(tj, Tr5));
      its5.push_back(params5._ceres_iteration);

      D2DNDTParameters params7;
      params7.cell_size = c;
      params7.r_variance = params7.t_variance = 0;
      params7.d2 = d2;
      params7._usedtime.Start();
      auto tgt7 = MakeNDTMap(datat, params7);
      auto src7 = MakeNDTMap(datas, params7);
      auto T7 = SNDTMatch2(tgt7, src7, params7);
      ut7 = ut7 + params7._usedtime;
      Tr7 = Tr7 * Affine3dFromAffine2d(T7);
      path7.poses.push_back(MakePoseStampedMsg(tj, Tr7));
      its7.push_back(params7._ceres_iteration);
    } else {
      Affine2d T5 = Affine2d::Identity();
      int it5 = 0;
      for (double cs : {4., 2., 1., 0.5}) {
        D2DNDTParameters params5;
        params5.cell_size = cs;
        params5.r_variance = params5.t_variance = 0;
        params5.d2 = d2;
        params5._usedtime.Start();
        auto tgt5 = MakeNDTMap(datat, params5);
        auto src5 = MakeNDTMap(datas, params5);
        T5 = D2DNDTMatch(tgt5, src5, params5, T5);
        ut5 = ut5 + params5._usedtime;
        it5 += params5._ceres_iteration;
      }
      Tr5 = Tr5 * Affine3dFromAffine2d(T5);
      path5.poses.push_back(MakePoseStampedMsg(tj, Tr5));
      its5.push_back(it5);

      Affine2d T7 = Affine2d::Identity();
      int it7 = 0;
      for (double cs : {4., 2., 1., 0.5}) {
        D2DNDTParameters params7;
        params7.cell_size = cs;
        params7.r_variance = params7.t_variance = 0;
        params7.d2 = d2;
        params7._usedtime.Start();
        auto tgt7 = MakeNDTMap(datat, params7);
        auto src7 = MakeNDTMap(datas, params7);
        T7 = SNDTMatch2(tgt7, src7, params7, T7);
        ut7 = ut7 + params7._usedtime;
        it7 += params7._ceres_iteration;
      }
      Tr7 = Tr7 * Affine3dFromAffine2d(T7);
      path7.poses.push_back(MakePoseStampedMsg(tj, Tr7));
      its7.push_back(it7);
    }
  }
  bar.finish();

  ros::init(argc, argv, "evaltime");
  ros::NodeHandle nh;
  auto pub5 = nh.advertise<nav_msgs::Path>("path5", 0, true);
  auto pub7 = nh.advertise<nav_msgs::Path>("path7", 0, true);
  auto pubgt = nh.advertise<nav_msgs::Path>("pathg", 0, true);

  MakeGtLocal(gtpath, t0);
  pub5.publish(path5);
  pub7.publish(path7);
  pubgt.publish(gtpath);
  PrintResult("method 5", path5, gtpath);
  PrintResult("method 7", path7, gtpath);
  PrintTime("method 5", ut5, ids.size() - 1, its5);
  PrintTime("method 7", ut7, ids.size() - 1, its7);
  ros::spin();
}