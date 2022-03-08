// On road pairwise ground truth
#include <bits/stdc++.h>
#include <common/common.h>
#include <metric/metric.h>
#include <sndt/visuals.h>
#include <sndt_exec/wrapper.h>

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
  string d;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("h,h", "Produce help message")
      ("d,d", po::value<string>(&d)->required(), "Data (logxx)");
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
  auto t0 = vpc[0].header.stamp;
  MakeGtLocal(gtpath, t0);
  vector<double> r, t;
  cout << PCMsgTo2D(vpc[10], 0.3).size() << endl;
  for (size_t i = 0; i < vpc.size() - 1; ++i) {
    auto aff = GetBenchMark(gtpath, vpc[i].header.stamp, vpc[i + 1].header.stamp);
    r.push_back(TransNormRotDegAbsFromAffine2d(aff)(0));
    t.push_back(TransNormRotDegAbsFromAffine2d(aff)(1));
  }
  Stat(r).PrintResult();
  Stat(t).PrintResult();
}
