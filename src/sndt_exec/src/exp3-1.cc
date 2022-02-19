// Generate figure oriented
#include <metric/metric.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <sndt/matcher.h>
#include <sndt_exec/wrapper.h>
#include <tqdm/tqdm.h>

#include <boost/program_options.hpp>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

void XYT(string str, Affine2d aff) {
  printf("%s: %.6f, %.6f, %.6f\n", str.c_str(), aff.translation()(0),
         aff.translation()(1),
         Eigen::Rotation2Dd(aff.rotation()).angle() * 180 / M_PI);
}

void NPArray(string str, vector<double> data) {
  printf("%s = np.array([", str.c_str());
  for (auto d : data) printf("%f, ", d);
  printf("])\n");
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  int n;
  double cell_size, voxel, d2;
  string data;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("n,n", po::value<int>(&n)->default_value(0), "n")
      ("d2", po::value<double>(&d2)->required(), "d2");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  po::notify(vm);

  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath(data), "lidar.ser"), vpc);

  auto tgt = PCMsgTo2D(vpc[n], voxel);
  TransformPointsInPlace(tgt, aff2);

  vector<double> y1, y3, y5, y7;
  vector<double> xs(21);
  for (int i = 0; i < 21; ++i) xs[i] = pow(10, -2.0 + 0.1 * i);
  double rmax = 10;

  tqdm bar;
  for (size_t i = 0; i < xs.size(); ++i) {
    bar.progress(i, xs.size());
    int samples = 100;
    auto affs = RandomTransformGenerator2D(xs[i] * rmax).Generate(samples);
    vector<Affine2d> affs2{affs[25], affs[30], affs[35], affs[40], affs[45]};
    vector<double> e1, e3, e5, e7;
    for (auto aff : affs) {
      auto src = TransformPoints(tgt, aff);
      vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, Eigen::Affine2d::Identity()}};
      vector<pair<vector<Vector2d>, Affine2d>> datas{{src, Eigen::Affine2d::Identity()}};

      // ICP Method
      // ICPParameters params1;
      // params1.reject = false;
      // params1._usedtime.Start();
      // auto tgt1 = MakePoints(datat, params1);
      // auto src1 = MakePoints(datas, params1);
      // auto T1 = ICPMatch(tgt1, src1, params1);
      // if ((T1 * aff).isApprox(Eigen::Affine2d::Identity(), 1e-2))
      //   e1.push_back(
      //     TransNormRotDegAbsFromAffine2d(params1._sols[0].back() * aff)(0));

      // Symmetric ICP Method
      SICPParameters params3;
      params3.reject = true;
      params3._usedtime.Start();
      auto tgt3 = MakePoints(datat, params3);
      auto src3 = MakePoints(datas, params3);
      auto T3 = SICPMatch(tgt3, src3, params3);
      if ((T3 * aff).isApprox(Eigen::Affine2d::Identity(), 1e-2))
        e3.push_back(
          TransNormRotDegAbsFromAffine2d(params3._sols[0].back() * aff)(0));

      // D2D-NDT Method
      // D2DNDTParameters params5;
      // params5.r_variance = params5.t_variance = 0;
      // params5.cell_size = cell_size;
      // params5.d2 = d2;
      // params5.reject = true;
      // params5._usedtime.Start();
      // auto tgt5 = MakeNDTMap(datat, params5);
      // auto src5 = MakeNDTMap(datas, params5);
      // auto T5 = D2DNDTMatch(tgt5, src5, params5);
      // if ((T5 * aff).isApprox(Eigen::Affine2d::Identity(), 1e-2)) ++s5;
      // if ((T5 * aff).isApprox(Eigen::Affine2d::Identity(), 1e-2))
      //   e5.push_back(
      //     TransNormRotDegAbsFromAffine2d(params5._sols[0].back() * aff)(0));

      // Symmetric NDT Method
      // D2DNDTParameters params7;
      // params7.r_variance = params7.t_variance = 0;
      // params7.cell_size = cell_size;
      // params7.d2 = d2;
      // params7.reject = true;
      // params7._usedtime.Start();
      // auto tgt7 = MakeNDTMap(datat, params7);
      // auto src7 = MakeNDTMap(datas, params7);
      // auto T7 = SNDTMatch2(tgt7, src7, params7);
      // if ((T7 * aff).isApprox(Eigen::Affine2d::Identity(), 1e-2)) ++s7;
      // if ((T7 * aff).isApprox(Eigen::Affine2d::Identity(), 1e-2))
      //   e7.push_back(
      //     TransNormRotDegAbsFromAffine2d(params7._sols[0].back() * aff)(0));
    }
    printf("sr: %ld, %ld, %ld, %ld\n", e1.size(), e3.size(), e5.size(), e7.size());
    // y1.push_back(Stat(e1).mean / rmax);
    y3.push_back(Stat(e3).mean / rmax);
    // y5.push_back(Stat(e5).mean / rmax);
    // y7.push_back(Stat(e7).mean / rmax);
  }
  bar.finish();

  NPArray("x", xs);
  // NPArray("y1", y1);
  NPArray("y3", y3);
  // NPArray("y5", y5);
  // NPArray("y7", y7);
  // printf("converge(x, y1, y3, y5, y7)\n");
}
