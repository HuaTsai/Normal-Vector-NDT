// Test Limit oriented
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

  // for (int n = 0; n < 2000; n += 25) {
    auto tgt = PCMsgTo2D(vpc[n], voxel);
    TransformPointsInPlace(tgt, aff2);
    printf("n = %d, ", n);

    for (double r : {1, 2, 4, 6, 8, 10, 12, 14, 16}) {
      int cf = 0;
      int samples = 100;
      auto affs = RandomTransformGenerator2D(r).Generate(samples);
      // vector<Affine2d> affs2{affs[25], affs[30], affs[35], affs[40], affs[45]};
      // for (auto aff : affs2) {
      for (auto aff : affs) {
        auto src = TransformPoints(tgt, aff);
        vector<pair<vector<Vector2d>, Affine2d>> datat{
            {tgt, Eigen::Affine2d::Identity()}};
        vector<pair<vector<Vector2d>, Affine2d>> datas{
            {src, Eigen::Affine2d::Identity()}};

        D2DNDTParameters params5;
        params5.r_variance = params5.t_variance = 0;
        params5.cell_size = cell_size;
        params5.d2 = d2;
        params5.reject = false;
        params5._usedtime.Start();
        auto tgt5 = MakeNDTMap(datat, params5);
        auto src5 = MakeNDTMap(datas, params5);
        auto T5 = D2DNDTMatch(tgt5, src5, params5);
        if (!(T5 * aff).isApprox(Eigen::Affine2d::Identity(), 1e-2)) {
          ++cf;
          // printf("f @ %.2f", r);
          // isf = true;
          // break;
        }
      }
      printf("r = %.2f, fail %d\n", r, cf);
      // if (isf) break;
    }
    printf("\n");
  // }
}

// d2 0.01 v 0: 0, 75, 175, 1525 -> 14
// d2 0.01 v 1: 75, 325, 475, 1525 -> 14
// d2 0.05 v 0: 50
// 175 0.01 m5:
//   r = 8.00, fail 1
//   r = 10.00, fail 4
//   r = 12.00, fail 11
//   r = 14.00, fail 41
//   r = 16.00, fail 65