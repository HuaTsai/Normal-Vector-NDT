#include <common/common.h>
#include <sndt_exec/wrapper.h>

#include <boost/program_options.hpp>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  int n, m;
  double cell_size, voxel, r, radius = 1.5;
  string data;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("n,n", po::value<int>(&n)->default_value(0), "n")
      ("r,r", po::value<double>(&r)->default_value(15), "r");
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

  auto aff = Translation2d(4, 0) * Rotation2Dd(20 * M_PI / 180.);
  Vector3d gt(-4, 0, -20);
  cout << "aff: " << XYTDegreeFromAffine2d(aff).transpose() << endl;
  auto src = TransformPoints(tgt, aff);
  vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, Affine2d::Identity()}};
  vector<pair<vector<Vector2d>, Affine2d>> datas{{src, Affine2d::Identity()}};

  SNDTParameters params6;
  params6.cell_size = cell_size;
  params6.radius = radius;
  params6.r_variance = params6.t_variance = 0;
  auto tgt6 = MakeSNDTMap(datat, params6);
  auto src6 = MakeSNDTMap(datas, params6);
  tgt6.ShowCellDistri();
  src6.ShowCellDistri();
}