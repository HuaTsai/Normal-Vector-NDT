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

bool SF(Affine2d T) {
  auto rt = TransNormRotDegAbsFromAffine2d(T);
  return rt(0) < 1 && rt(1) < 3;
}

void PrintTime(int n, const UsedTime &ut) {
  double nn = n * 1000.;
  std::printf(
      "%d success: nm: %.2f, ndt: %.2f, bud: %.2f, opt: %.2f, oth: %.2f\n", n,
      ut.normal() / nn, ut.ndt() / nn, ut.build() / nn, ut.optimize() / nn,
      ut.others() / nn);
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  int n;
  double cell_size, voxel, radius = 1.5, d2 = -1;
  string data;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("n,n", po::value<int>(&n)->default_value(0), "To where")
      ("d2", po::value<double>(&d2), "d2");
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

  // vector<double> rs{0, 4, 8, 12, 16, 20, 24, 28};
  // vector<double> ts{0, 5, 10, 15, 20, 25, 30, 35, 40};
  vector<double> rs{12};
  vector<double> ts{20};
  // vector<double> rs{24};
  // vector<double> ts{35};
  vector<vector<double>> s(rs.size(), vector<double>(ts.size()));

  for (size_t i = 0; i < rs.size(); ++i) {
    for (size_t j = 0; j < ts.size(); ++j) {
      UsedTime ut5, ut7;
      int cnt5 = 0, cnt7 = 0;
      auto affs = UniformTransformGenerator(rs[i], ts[j]).Generate(50);
      vector<pair<vector<Vector2d>, Affine2d>> datat{
          {tgt, Affine2d::Identity()}};

      for (auto aff : affs) {
        auto src = TransformPoints(tgt, aff);
        vector<pair<vector<Vector2d>, Affine2d>> datas{
            {src, Affine2d::Identity()}};

        D2DNDTParameters params5;
        params5.cell_size = cell_size;
        params5.r_variance = params5.t_variance = 0;
        params5.d2 = d2;
        auto tgt5 = MakeNDTMap(datat, params5);
        auto src5 = MakeNDTMap(datas, params5);
        auto T5 = D2DNDTMatch(tgt5, src5, params5);
        if (SF(T5 * aff)) {
          ut5 = ut5 + params5._usedtime;
          ++cnt5;
        }

        D2DNDTParameters params7;
        params7.cell_size = cell_size;
        params7.r_variance = params7.t_variance = 0;
        params7.d2 = d2;
        auto tgt7 = MakeNDTMap(datat, params7);
        auto src7 = MakeNDTMap(datas, params7);
        auto T7 = SNDTMatch2(tgt7, src7, params7);
        if (SF(T7 * aff)) {
          ut7 = ut7 + params7._usedtime;
          ++cnt7;
        }
      }

      cout << rs[i] << ", " << ts[j] << endl;
      PrintTime(cnt5, ut5);
      PrintTime(cnt7, ut7);
    }
  }
}