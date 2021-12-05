#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <sndt/matcher.h>
#include <sndt/visuals.h>
#include <sndt_exec/wrapper.h>
#include <std_msgs/Int32.h>
#include <tqdm/tqdm.h>

#include <boost/program_options.hpp>

using namespace std;
using namespace Eigen;
using namespace visualization_msgs;
namespace po = boost::program_options;
const auto &Avg = Average;

string filepath = "/tmp/evalrate.txt";
int samples = 10;
int scans = 1;

bool SF(Affine2d T) {
  auto rt = TransNormRotDegAbsFromAffine2d(T);
  return rt(0) < 1 && rt(1) < 3;
}

void Output(int m,
            const vector<double> rs,
            const vector<double> &ts,
            const vector<vector<double>> &data) {
  ofstream fout(filepath);
  fout.setf(ios::fixed);
  fout.precision(2);
  string str;
  if (m == 1) {
    str = "ICP";
  } else if (m == 2) {
    str = "Point-to-Plane ICP";
  } else if (m == 3) {
    str = "Symmetric ICP";
  } else if (m == 4) {
    str = "Point-to-Distribution NDT";
  } else if (m == 42) {
    str = "Point-to-Distribution NDT (MD)";
  } else if (m == 5) {
    str = "Distribution-to-Distribution NDT";
  } else if (m == 52) {
    str = "Distribution-to-Distribution NDT (MD)";
  } else if (m == 6) {
    str = "Symmetric NDT";
  } else if (m == 62) {
    str = "Symmetric NDT (MD)";
  } else if (m == 7) {
    str = "Symmetric NDT 2";
  } else if (m == 72) {
    str = "Symmetric NDT 2 (MD)";
  }
  str += " with " + to_string(samples) + " (tfs) x " + to_string(scans) +
         " (scenes) Trials Each Tile\n";
  fout << str;
  for (size_t i = 0; i < rs.size(); ++i)
    fout << rs[i] << ((i + 1 == rs.size()) ? "\n" : ", ");
  for (size_t i = 0; i < ts.size(); ++i)
    fout << ts[i] << ((i + 1 == ts.size()) ? "\n" : ", ");

  for (auto d : data)
    for (int i = 0; i < d.size(); ++i)
      fout << d[i] * 100 << ((i + 1 == d.size()) ? "\n" : ", ");
  fout.close();
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  int n, m;
  double cell_size, voxel, radius;
  string data;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("radius,a", po::value<double>(&radius)->default_value(1.5), "Radius")
      ("m,m", po::value<int>(&m)->required(), "Method");
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

  vector<double> rs{0, 4, 8, 12, 16, 20, 24, 28};
  vector<double> ts{0, 5, 10, 15, 20, 25, 30, 35, 40};
  vector<vector<double>> s(rs.size(), vector<double>(ts.size()));

  for (int n = 100; n <= scans * 100; ++n) {
    auto tgt = PCMsgTo2D(vpc[n], voxel);
    TransformPointsInPlace(tgt, aff2);
    for (int i = 0; i < rs.size(); ++i) {
      for (int j = 0; j < ts.size(); ++j) {
        auto affs = UniformTransformGenerator(rs[i], ts[j]).Generate(samples);
        vector<pair<vector<Vector2d>, Affine2d>> datat{
            {tgt, Affine2d::Identity()}};
        for (auto aff : affs) {
          auto src = TransformPoints(tgt, aff);
          vector<pair<vector<Vector2d>, Affine2d>> datas{
              {src, Affine2d::Identity()}};

          Affine2d T;

          if (m == 1) {
            ICPParameters params1;
            auto tgt1 = MakePoints(datat, params1);
            auto src1 = MakePoints(datas, params1);
            T = ICPMatch(tgt1, src1, params1);
          } else if (m == 2) {
            Pt2plICPParameters params2;
            auto tgt2 = MakePoints(datat, params2);
            auto src2 = MakePoints(datas, params2);
            T = Pt2plICPMatch(tgt2, src2, params2);      
          } else if (m == 3) {
            SICPParameters params3;
            params3.radius = radius;
            auto tgt3 = MakePoints(datat, params3);
            auto src3 = MakePoints(datas, params3);
            T = SICPMatch(tgt3, src3, params3);
          } else if (m == 4 || m == 42) {
            P2DNDTParameters params4;
            params4.cell_size = cell_size;
            params4.r_variance = params4.t_variance = 0;
            auto tgt4 = MakeNDTMap(datat, params4);
            auto src4 = MakePoints(datas, params4);
            if (m == 4)
              T = P2DNDTMatch(tgt4, src4, params4);
            else
              T = P2DNDTMDMatch(tgt4, src4, params4);
          } else if (m == 5 || m == 52) {
            D2DNDTParameters params5;
            params5.cell_size = cell_size;
            params5.r_variance = params5.t_variance = 0;
            auto tgt5 = MakeNDTMap(datat, params5);
            auto src5 = MakeNDTMap(datas, params5);
            if (m == 5)
              T = D2DNDTMatch(tgt5, src5, params5);
            else
              T = D2DNDTMDMatch(tgt5, src5, params5);
          } else if (m == 6 || m == 62) {
            SNDTParameters params6;
            params6.cell_size = cell_size;
            params6.radius = radius;
            params6.r_variance = params6.t_variance = 0;
            auto tgt6 = MakeSNDTMap(datat, params6);
            auto src6 = MakeSNDTMap(datas, params6);
            T = SNDTMDMatch(tgt6, src6, params6);
          } else if (m == 7 || m == 72) {
            D2DNDTParameters params7;
            params7.cell_size = cell_size;
            params7.r_variance = params7.t_variance = 0;
            auto tgt7 = MakeNDTMap(datat, params7);
            auto src7 = MakeNDTMap(datas, params7);
            T = SNDTMDMatch2(tgt7, src7, params7);
          }

          if (SF(T * aff)) ++s[i][j];
        }
      }
    }
  }
  for (int i = 0; i < rs.size(); ++i)
    for (int j = 0; j < ts.size(); ++j)
      s[i][j] /= (scans * samples);

  Output(m, rs, ts, s);

  std::cerr << "Generate Figure...";
  string python = PYTHONPATH;
  string script = JoinPath(WSPATH, "src/sndt_exec/scripts/evalrate.py");
  string output = JoinPath(
      WSPATH, "src/sndt_exec/output/rate-" + GetCurrentTimeAsString() + ".png");
  std::system((python + " " + script + " " + filepath + " " + output).c_str());
  std::cerr << " Done!" << endl;
}
