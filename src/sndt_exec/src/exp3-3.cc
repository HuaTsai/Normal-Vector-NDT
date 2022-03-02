// Error Before vs. After @ Fake Ellipse Data
#include <metric/metric.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <sndt/matcher.h>
#include <sndt/visuals.h>
#include <sndt_exec/wrapper.h>
#include <tqdm/tqdm.h>
#include <sndt/matcher_pcl.h>

#include <boost/program_options.hpp>

using namespace std;
using namespace Eigen;
using namespace visualization_msgs;
namespace po = boost::program_options;
// HACK
#define USETR

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

vector<Vector2d> EllipseData() {
  double x = 10, y = 6, xc = 0, yc = 0;
  vector<Vector2d> ret;
  for (int i = 0; i < 360; ++i) {
    double th = i * M_PI / 180.;
    ret.push_back(Vector2d(xc + x * cos(th), yc + y * sin(th)));
  }
  return ret;
}

void NPArray(string str, vector<double> data) {
  printf("%s = np.array([", str.c_str());
  for (auto d : data) printf("%f, ", d);
  printf("])\n");
}

bool SuccessMatch(Affine2d aff) {
  auto diff = TransNormRotDegAbsFromAffine2d(aff);
  return diff(0) < 0.2 && diff(1) < 3;
}

void PrintTime(const vector<double> &opts, const vector<double> &its) {
  Stat o(opts), i(its);
  printf("%.2f (%.2f) / %.2f (%.2f) / %.2f (%.2f)\n", o.min, i.min, o.max,
         i.max, o.mean, i.mean);
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
  res.iiit.push_back(0);
  res.opt.push_back(params._usedtime.optimize());
  res.ttl.push_back(params._usedtime.total());
  ++res.n;
}

int main(int argc, char **argv) {
  int samples;
  double cell_size, voxel, d2;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("samples,s", po::value<int>(&samples)->default_value(100), "Transform Samples")
      ("d2", po::value<double>(&d2)->required(), "d2");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  po::notify(vm);

  auto tgt = EllipseData();
  vector<double> y1, y3, y5, y7;
  vector<double> xs(21);
  for (int i = 0; i < 21; ++i) xs[i] = pow(10, -2.0 + 0.1 * i);
  double rmax = 5;
  // double rmax = 1.2;
  // xs.resize(1); xs = {1};

  ros::init(argc, argv, "exp3_3");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<Marker>("marker1", 0, true);
  ros::Publisher pub2 = nh.advertise<Marker>("marker2", 0, true);
  ros::Publisher pub3 = nh.advertise<Marker>("marker3", 0, true);
  ros::Publisher pub4 = nh.advertise<Marker>("marker4", 0, true);

  for (size_t i = 0; i < xs.size(); ++i) {
    auto affs = RandomTransformGenerator2D(xs[i] * rmax).Generate(samples);
    // auto affs = RandomTranslationGenerator(xs[i] * rmax).Generate(samples);
    vector<double> e2tl, e2rot;
    Res r1, r3, r5, r7;
    for (auto aff : affs) {
      auto src = TransformPoints(tgt, aff);
      vector<pair<vector<Vector2d>, Affine2d>> datat{
          {tgt, Eigen::Affine2d::Identity()}};
      vector<pair<vector<Vector2d>, Affine2d>> datas{
          {src, Eigen::Affine2d::Identity()}};

      // ICP Method
      ICPParameters params1;
      params1.reject = true;
      params1._usedtime.Start();
      auto tgt1 = MakePoints(datat, params1);
      auto src1 = MakePoints(datas, params1);
#ifdef USETR
      auto T1 = ICPMatch(tgt1, src1, params1);
#else
      auto T1 = IMatch(tgt1, src1, params1);
#endif
      Updates(T1, params1, aff, r1);

      // PCL-ICP
      auto T2 = PCLNDT(tgt1, src1);
      if (SuccessMatch(T2 * aff)) {
        auto diff = TransNormRotDegAbsFromAffine2d(T2 * aff);
        e2tl.push_back(diff(0));
        e2rot.push_back(diff(1));
      }

      // Symmetric ICP Method
      SICPParameters params3;
      params3.reject = true;
      params3.minimize = ceres::LINE_SEARCH;
      params3._usedtime.Start();
      auto tgt3 = MakePoints(datat, params3);
      auto src3 = MakePoints(datas, params3);
#ifdef USETR
      // BUG: SICP does not work...
      // auto T3 = SICPMatch(tgt3, src3, params3);
      auto T3 = PMatch(tgt3, src3, params3);
#else
      auto T3 = PMatch(tgt3, src3, params3);
#endif
      Updates(T3, params3, aff, r3);

      // D2D-NDT Method
      D2DNDTParameters params5;
      params5.reject = true;
      params5.r_variance = params5.t_variance = 0;
      params5.cell_size = cell_size;
      params5.d2 = d2;
      params5._usedtime.Start();
      auto tgt5 = MakeNDTMap(datat, params5);
      auto src5 = MakeNDTMap(datas, params5);
#ifdef USETR
      auto T5 = DMatch(tgt5, src5, params5);
#else
      auto T5 = D2DNDTMatch(tgt5, src5, params5);
#endif
      Updates(T5, params5, aff, r5);

      // Symmetric NDT Method
      D2DNDTParameters params7;
      params7.reject = true;
      params7.r_variance = params7.t_variance = 0;
      params7.cell_size = cell_size;
      params7.d2 = d2;
      params7._usedtime.Start();
      auto tgt7 = MakeNDTMap(datat, params7);
      auto src7 = MakeNDTMap(datas, params7);
#ifdef USETR
      auto T7 = SMatch(tgt7, src7, params7);
#else
      auto T7 = SNDTMatch2(tgt7, src7, params7);
#endif
      Updates(T7, params7, aff, r7);
    }
    printf("sr(%.4f): %d, %d, %d, %d\n", xs[i] * rmax, r1.n, r3.n, r5.n, r7.n);
    y1.push_back(Stat(r1.tl_1).rms / rmax);
    y3.push_back(Stat(r3.tl_1).rms / rmax);
    y5.push_back(Stat(r5.tl_1).rms / rmax);
    y7.push_back(Stat(r7.tl_1).rms / rmax);
    // printf("%f / %f / %f / %f\n", Stat(e1).mean, Stat(e3).mean,
    // Stat(e5).mean,
    //        Stat(e7).mean);
    // PrintTime(r1.opt, r1.iit);
    // PrintTime(r3.opt, r3.iit);
    // PrintTime(r5.opt, r5.iit);
    // PrintTime(r7.opt, r7.iit);
    // r1.Print();
    // r3.Print();
    // r5.Print();
    // r7.Print();
    printf("icp: %ld, %f / %f\n", e2tl.size(), Stat(e2tl).rms, Stat(e2rot).rms);
  }

  NPArray("x", xs);
  NPArray("y1", y1);
  NPArray("y3", y3);
  NPArray("y5", y5);
  NPArray("y7", y7);
  printf("converge(x, y1, y3, y5, y7)\n");
}
