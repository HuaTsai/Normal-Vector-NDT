// Error Before vs. After @ Real Data
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

  bool tr;
  int samples;
  double cell_size, voxel, d2;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("samples,s", po::value<int>(&samples)->default_value(100), "Transform Samples")
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

  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath("log35-1"), "lidar.ser"), vpc);
  auto tgt = PCMsgTo2D(vpc[725], voxel);
  TransformPointsInPlace(tgt, aff2);

  vector<double> y1, y3, y5, y7;
  vector<double> xs(21);
  for (int i = 0; i < 21; ++i) xs[i] = pow(10, -2.0 + 0.1 * i);
  double rmax = 5;

  for (size_t i = 0; i < xs.size(); ++i) {
    auto affs = RandomTransformGenerator2D(xs[i] * rmax).Generate(samples);
    Res r1, r3, r5, r7;
    for (auto aff : affs) {
      auto src = TransformPoints(tgt, aff);
      // clang-format off
      vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, Eigen::Affine2d::Identity()}};
      vector<pair<vector<Vector2d>, Affine2d>> datas{{src, Eigen::Affine2d::Identity()}};
      // clang-format on

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
      Updates(T1, params1, aff, r1);

      // Symmetric ICP Method
      SICPParameters params3;
      params3.radius = 1.5;
      params3.reject = true;
      params3._usedtime.Start();
      auto tgt3 = MakePoints(datat, params3);
      auto src3 = MakePoints(datas, params3);
      Affine2d T3;
      if (tr)
        T3 = SICPMatch(tgt3, src3, params3);
      else
        T3 = PMatch(tgt3, src3, params3);
      Updates(T3, params3, aff, r3);

      // D2D-NDT Method
      D2DNDTParameters params5;
      params5.reject = true;
      params5.cell_size = cell_size;
      params5.d2 = d2;
      params5._usedtime.Start();
      auto tgt5 = MakeNDT(datat, params5);
      auto src5 = MakeNDT(datas, params5);
      // cout << "NDT" << endl;
      Affine2d T5;
      if (tr)
        T5 = DMatch(tgt5, src5, params5);
      else
        T5 = D2DNDTMatch(tgt5, src5, params5);
      Updates(T5, params5, aff, r5);

      // Symmetric NDT Method
      D2DNDTParameters params7;
      params7.reject = true;
      params7.cell_size = cell_size;
      params7.d2 = d2;
      params7._usedtime.Start();
      auto tgt7 = MakeNDT(datat, params7);
      auto src7 = MakeNDT(datas, params7);
      // cout << "SNDT" << endl;
      Affine2d T7;
      if (tr)
        T7 = SMatch(tgt7, src7, params7);
      else
        T7 = SNDTMatch2(tgt7, src7, params7);
      Updates(T7, params7, aff, r7);
    }
    printf("sr(%.4f): %d, %d, %d, %d\n", xs[i] * rmax, r1.n, r3.n, r5.n, r7.n);
    y1.push_back(Stat(r1.tl_1).rms / rmax);
    y3.push_back(Stat(r3.tl_1).rms / rmax);
    y5.push_back(Stat(r5.tl_1).rms / rmax);
    y7.push_back(Stat(r7.tl_1).rms / rmax);
    r1.Print();
    r3.Print();
    r5.Print();
    r7.Print();
  }

  NPArray("x", xs);
  NPArray("y1", y1);
  NPArray("y3", y3);
  NPArray("y5", y5);
  NPArray("y7", y7);
  printf("converge(x, y1, y3, y5, y7)\n");
}
