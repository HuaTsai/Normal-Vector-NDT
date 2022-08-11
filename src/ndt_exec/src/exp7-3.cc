// Single Scene Test: d2
#include <common/common.h>
#include <metric/metric.h>
#include <nav_msgs/Path.h>
#include <ndt/matcher.h>
#include <ndt_exec/wrapper.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>
#include <visualization_msgs/MarkerArray.h>

#include <boost/program_options.hpp>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;
using Marker = visualization_msgs::Marker;

struct Res {
  void Show() {
    printf(" its: %f / %f\n", Stat(its).mean, Stat(its).max);
    printf(" err: %f / %f\n", Stat(terr).rms, Stat(rerr).rms);
    printf("corr: %f\n", Stat(corr).mean);
    double den = its.size() * 1000.;
    printf("time: %f / %f\n", timer.optimize() / den, timer.total() / den);
  }
  vector<double> its;
  vector<double> terr;
  vector<double> rerr;
  vector<double> corr;
  Timer timer;
};

Affine3d BM(const nav_msgs::Path &gt,
            const ros::Time &t1,
            const ros::Time &t2) {
  Affine3d To, Ti;
  tf2::fromMsg(GetPose(gt.poses, t2), To);
  tf2::fromMsg(GetPose(gt.poses, t1), Ti);
  return Ti.inverse() * To;
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  string d;
  int n, f;
  bool nndt;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("h,h", "Produce help message")
      ("d,d", po::value<string>(&d)->required(), "Data (logxx)")
      ("n,n", po::value<int>(&n)->default_value(0), "N")
      ("f,f", po::value<int>(&f)->default_value(1), "F")
      ("x,x", po::value<bool>(&nndt)->default_value(true)->implicit_value(false), "NDT");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  po::notify(vm);

  nav_msgs::Path gt;
  SerializationInput(JoinPath(GetDataPath(d), "gt.ser"), gt);
  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath(d), "lidar.ser"), vpc);

  auto tgt = PCMsgTo3D(vpc[n], 0);
  auto src = PCMsgTo3D(vpc[n + f], 0);
  TransformPointsInPlace(tgt, aff3);
  TransformPointsInPlace(src, aff3);
  auto ben = BM(gt, vpc[n].header.stamp, vpc[n + f].header.stamp);
  cout << "Bench Mark: " << TransNormRotDegAbsFromAffine3d(ben).transpose()
       << endl;

  for (auto d2 : {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4,
                  0.5, 0.6, 0.7, 0.8, 0.9, 1.0}) {
    auto op = {kNVNDT, k1to1};
    auto m = NDTMatcher::GetIter(op, {0.5, 1, 2}, d2);
    m.SetSource(src);
    m.SetTarget(tgt);
    auto res = m.Align();
    auto err = TransNormRotDegAbsFromAffine3d(res * ben.inverse());
    if (err(0) > 0.1 || err(1) > 0.1) continue;
    printf("%6f: %.6f / %.6f / %.2f (%d)\n", d2, err(0), err(1),
           m.timer().optimize() / 1000., m.iteration());
  }
}
