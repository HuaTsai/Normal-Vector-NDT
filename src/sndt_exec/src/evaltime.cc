#include <bits/stdc++.h>
#include <common/common.h>
#include <metric/metric.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_2D.h>
#include <rosbag/bag.h>
#include <sndt/visuals.h>
#include <sndt_exec/wrapper.h>
#include <tqdm/tqdm.h>
#include <nav_msgs/Odometry.h>
#include <tf2_msgs/TFMessage.h>

#include <boost/program_options.hpp>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

int n;

void PrintTime(string str, const UsedTime &ut) {
  double den = (n - 1) * 1000.;
  std::printf(
      "%s: nm: %.2f, ndt: %.2f, bud: %.2f, opt: %.2f, oth: %.2f, ttl: %.2f\n",
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
  // te.set_evaltype(TrajectoryEvaluation::EvalType::kRelativeBySingle);
  te.set_evaltype(TrajectoryEvaluation::EvalType::kRelativeByLength);
  te.set_length(100);
  auto rpe = te.ComputeRMSError2D();
  cout << str << ": len " << te.gtlength() << " ate " << ate.first.rms << ", "
       << ate.second.rms << ", rpe " << rpe.first.rms << ", " << rpe.second.rms
       << endl;
  printf("%.2f / %.2f\n", rpe.first.rms, rpe.second.rms);
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
  int f;
  double cell_size, voxel, radius = 1.5, d2 = -1;
  bool bagout;
  string data;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("n,n", po::value<int>(&n)->default_value(-1), "To where")
      ("f,f", po::value<int>(&f)->default_value(1), "Frames")
      ("m,m", po::value<vector<int>>(&m)->required()->multitoken(), "Method")
      ("d2", po::value<double>(&d2), "d2")
      ("bagout,b", po::value<bool>(&bagout)->default_value(false)->implicit_value(true), "Write RosBag");
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

  rosbag::Bag bag;
  if (bagout) bag.open("replay.bag", rosbag::bagmode::Write);

  nav_msgs::Path gtpath;
  SerializationInput(JoinPath(GetDataPath(data), "gt.ser"), gtpath);
  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath(data), "lidar.ser"), vpc);

  unordered_map<int, Affine3d> Tr{
      {0, Affine3d::Identity()}, {1, Affine3d::Identity()},
      {2, Affine3d::Identity()}, {3, Affine3d::Identity()},
      {5, Affine3d::Identity()}, {7, Affine3d::Identity()},
      {8, Affine3d::Identity()}};
  auto t0 = vpc[0].header.stamp;
  unordered_map<int, nav_msgs::Path> path{
      {0, InitFirstPose(t0)}, {1, InitFirstPose(t0)}, {2, InitFirstPose(t0)},
      {3, InitFirstPose(t0)}, {5, InitFirstPose(t0)}, {7, InitFirstPose(t0)},
      {8, InitFirstPose(t0)}};
  unordered_map<int, UsedTime> usedtime{{1, UsedTime()},
                                        {2, UsedTime()},
                                        {3, UsedTime()},
                                        {5, UsedTime()},
                                        {7, UsedTime()},
                                        {8, UsedTime()}};
  unordered_map<int, vector<double>> its{
      {1, {}}, {2, {}}, {3, {}}, {5, {}}, {7, {}}, {8, {}}};

  ros::init(argc, argv, "evaltime");
  ros::NodeHandle nh;

  tqdm bar;
  if (n == -1) n = vpc.size() - 1;
  for (int i = 0; i + f < n; i += f) {
    bar.progress(i, n);
    auto tgt = PCMsgTo2D(vpc[i], voxel);
    auto src = PCMsgTo2D(vpc[i + f], voxel);
    auto tj = vpc[i + f].header.stamp;

    vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, aff2}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{{src, aff2}};

    if (ms.count(0)) {
      pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
      pcl::registration::TransformationEstimation2D<pcl::PointXYZ,
                                                    pcl::PointXYZ>::Ptr
          est(new pcl::registration::TransformationEstimation2D<pcl::PointXYZ,
                                                                pcl::PointXYZ>);
      icp.setTransformationEstimation(est);
      icp.setMaximumIterations(500);
      icp.setEuclideanFitnessEpsilon(0.000001);
      icp.setTransformationEpsilon(0.000001);
      pcl::PointCloud<pcl::PointXYZ>::Ptr tgt0(
          new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr src0(
          new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ> out;
      for (auto pt : tgt) {
        pcl::PointXYZ p;
        p.x = (aff2 * pt)(0), p.y = (aff2 * pt)(1), p.z = 0;
        tgt0->push_back(p);
      }
      for (auto pt : src) {
        pcl::PointXYZ p;
        p.x = (aff2 * pt)(0), p.y = (aff2 * pt)(1), p.z = 0;
        src0->push_back(p);
      }
      icp.setInputSource(src0);
      icp.setInputTarget(tgt0);
      icp.align(out);
      Eigen::Matrix4d T00 = icp.getFinalTransformation().cast<double>();
      Matrix3d T0 = Matrix3d::Identity();
      T0.block<2, 2>(0, 0) = T00.block<2, 2>(0, 0);
      T0.block<2, 1>(0, 2) = T00.block<2, 1>(0, 3);
      Tr[0] = Tr[0] * Affine3dFromAffine2d(Affine2d(T0));
      path[0].poses.push_back(MakePoseStampedMsg(tj, Tr[0]));
    }

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

      nav_msgs::Odometry odom;
      odom.header.frame_id = "map";
      odom.header.stamp = tj;
      odom.child_frame_id = "base_link";
      odom.pose.pose = tf2::toMsg(Tr[7]);
      if (bagout) bag.write("odom", tj, odom);
      tf2_msgs::TFMessage tfmsg;
      geometry_msgs::TransformStamped tfs = tf2::eigenToTransform(Tr[7]);
      tfs.header.frame_id = "map";
      tfs.header.stamp = tj;
      tfs.child_frame_id = "base_link"; 
      tfmsg.transforms.push_back(tfs);
      if (bagout) bag.write("tf", vpc[0].header.stamp, tfmsg);
    }

    if (ms.count(8)) {
      Affine2d T8 = Affine2d::Identity();
      int it = 0;
      for (double cs : {4., 2., 1., 0.5}) {
        D2DNDTParameters params8;
        params8.cell_size = cs;
        params8.r_variance = params8.t_variance = 0;
        params8.d2 = d2;
        params8._usedtime.Start();
        auto tgt7 = MakeNDTMap(datat, params8);
        auto src7 = MakeNDTMap(datas, params8);
        T8 = SNDTMatch2(tgt7, src7, params8, T8);
        usedtime[8] = usedtime[8] + params8._usedtime;
        it += params8._ceres_iteration;
      }
      Tr[8] = Tr[8] * Affine3dFromAffine2d(T8);
      path[8].poses.push_back(MakePoseStampedMsg(tj, Tr[8]));
      its[8].push_back(it);
    }
  }
  if (bagout) { bag.close(); }
  bar.finish();

  // for (size_t i = 0; i < its[1].size(); ++i)
  //   cout << its[1][i] << ((i + 1 == its[1].size()) ? "\n" : ",");
  // for (size_t i = 0; i < its[2].size(); ++i)
  //   cout << its[2][i] << ((i + 1 == its[2].size()) ? "\n" : ",");
  // for (size_t i = 0; i < its[3].size(); ++i)
  //   cout << its[3][i] << ((i + 1 == its[3].size()) ? "\n" : ",");
  // for (size_t i = 0; i < its[5].size(); ++i)
  //   cout << its[5][i] << ((i + 1 == its[5].size()) ? "\n" : ",");
  // for (size_t i = 0; i < its[7].size(); ++i)
  //   cout << its[7][i] << ((i + 1 == its[7].size()) ? "\n" : ",");

  auto pub0 = nh.advertise<nav_msgs::Path>("path0", 0, true);
  auto pub1 = nh.advertise<nav_msgs::Path>("path1", 0, true);
  auto pub2 = nh.advertise<nav_msgs::Path>("path2", 0, true);
  auto pub3 = nh.advertise<nav_msgs::Path>("path3", 0, true);
  auto pub5 = nh.advertise<nav_msgs::Path>("path5", 0, true);
  auto pub7 = nh.advertise<nav_msgs::Path>("path7", 0, true);
  auto pubgt = nh.advertise<nav_msgs::Path>("pathg", 0, true);

  MakeGtLocal(gtpath, t0);
  pubgt.publish(gtpath);

  if (ms.count(0)) pub0.publish(path[0]);
  if (ms.count(1)) pub1.publish(path[1]);
  if (ms.count(2)) pub2.publish(path[2]);
  if (ms.count(3)) pub3.publish(path[3]);
  if (ms.count(5)) pub5.publish(path[5]);
  if (ms.count(7)) pub7.publish(path[7]);
  if (ms.count(8)) pub7.publish(path[8]);

  if (ms.count(0)) PrintResult("method 0", path[0], gtpath);
  if (ms.count(1)) PrintResult("method 1", path[1], gtpath);
  if (ms.count(2)) PrintResult("method 2", path[2], gtpath);
  if (ms.count(3)) PrintResult("method 3", path[3], gtpath);
  if (ms.count(5)) PrintResult("method 5", path[5], gtpath);
  if (ms.count(7)) PrintResult("method 7", path[7], gtpath);
  if (ms.count(8)) PrintResult("method 8", path[8], gtpath);

  if (ms.count(1)) PrintTime("method 1", usedtime[1]);
  if (ms.count(2)) PrintTime("method 2", usedtime[2]);
  if (ms.count(3)) PrintTime("method 3", usedtime[3]);
  if (ms.count(5)) PrintTime("method 5", usedtime[5]);
  if (ms.count(7)) PrintTime("method 7", usedtime[7]);
  if (ms.count(8)) PrintTime("method 8", usedtime[8]);

  if (ms.count(1)) cout << "method 1: " << Average(its[1]) << endl;
  if (ms.count(2)) cout << "method 2: " << Average(its[2]) << endl;
  if (ms.count(3)) cout << "method 3: " << Average(its[3]) << endl;
  if (ms.count(5)) cout << "method 5: " << Average(its[5]) << endl;
  if (ms.count(7)) cout << "method 7: " << Average(its[7]) << endl;
  if (ms.count(8)) cout << "method 8: " << Average(its[8]) << endl;

  ros::spin();
}
