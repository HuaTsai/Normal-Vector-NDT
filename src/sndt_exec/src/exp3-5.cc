#include <metric/metric.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <sndt/matcher.h>
#include <sndt/matcher_pcl.h>
#include <sndt/visuals.h>
#include <sndt_exec/wrapper.h>
#include <tqdm/tqdm.h>

#include <boost/program_options.hpp>

using namespace std;
using namespace Eigen;
using namespace visualization_msgs;
namespace po = boost::program_options;

struct Res {
  Res() : n(0) {}
  vector<double> tl_1, tl_f, rot_1, rot_f;
  vector<double> it, iit, iiit, opt, ttl;
  int n;
  void Print() {
    printf(
        "%.4f & %.4f & %.4f & %.4f & %.2f & %.2f & %.2f & %.2f & %.2f & %d\n",
        Stat(tl_1).rms, Stat(tl_f).rms, Stat(rot_1).rms, Stat(rot_f).rms,
        Stat(it).mean, Stat(iit).mean, Stat(iiit).mean, Stat(opt).mean / 1000.,
        Stat(ttl).mean / 1000., n);
  }
};

void NPArray(string str, vector<double> data) {
  printf("%s = np.array([", str.c_str());
  for (auto d : data) printf("%f, ", d);
  printf("])\n");
}

bool SuccessMatch(Affine2d aff) {
  auto diff = TransNormRotDegAbsFromAffine2d(aff);
  return diff(0) < 0.2 && diff(1) < 3;
}

void Updates(const Affine2d &Tf,
             const CommonParameters &params,
             const Affine2d &aff,
             Res &res) {
  if (!SuccessMatch(Tf * aff)) return;
  Affine2d T1 = params._sols[0].back();
  auto d1 = TransNormRotDegAbsFromAffine2d(T1 * aff);
  auto df = TransNormRotDegAbsFromAffine2d(Tf * aff);
  res.tl_1.push_back(d1(0));
  res.rot_1.push_back(d1(1));
  res.tl_f.push_back(df(0));
  res.rot_f.push_back(df(1));
  res.it.push_back(params._iteration);
  res.iit.push_back(params._ceres_iteration);
  res.iiit.push_back(params._search_iteration);
  res.opt.push_back(params._usedtime.optimize());
  res.ttl.push_back(params._usedtime.total());
  ++res.n;
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  double voxel;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  po::notify(vm);

  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath("log35-1"), "lidar.ser"), vpc);
  auto tgt = PCMsgTo2D(vpc[725], voxel);
  TransformPointsInPlace(tgt, aff2);

  ros::init(argc, argv, "exp3_5");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<visualization_msgs::Marker>("src", 0, true);
  pub.publish(MarkerOfPoints(tgt));
  ros::spin();
}
