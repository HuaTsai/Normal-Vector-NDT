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
const auto &Avg = Average;

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  int n;
  double cell_size, radius, voxel;
  string data;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("radius,r", po::value<double>(&radius)->default_value(1.5), "Radius")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("n,n", po::value<int>(&n)->default_value(0), "n");
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

  vector<double> rmsavgs1, rmsavgs2, rmsavgs3, rmsavgs4, rmsavgs5, rmsavgs6;
  // My rs
  // vector<double> rs(18);
  // for (int i = 0; i < 18; ++i) rs[i] = pow(10, -0.5 + 0.1 * i);

  // Paper rs
  vector<double> rs(21);
  double rmax = pow(10, 1.2);
  for (int i = 0; i < 21; ++i) rs[i] = pow(10, -2.0 + 0.1 * i) * rmax;

  for (auto r : rs) {
    int samples = 200;
    auto affs = RandomTransformGenerator2D(r).Generate(samples);
    vector<double> rms1, rms2, rms3, rms4, rms5, rms6;
    tqdm bar;
    for (size_t i = 0; i < affs.size(); ++i) {
      auto aff = affs[i];
      bar.progress(i, affs.size());
      auto src = TransformPoints(tgt, aff);
      vector<pair<vector<Vector2d>, Affine2d>> datat{
          {tgt, Eigen::Affine2d::Identity()}};
      vector<pair<vector<Vector2d>, Affine2d>> datas{
          {src, Eigen::Affine2d::Identity()}};

      // ICP Method
      ICPParameters params1;
      auto tgt1 = MakePoints(datat, params1);
      auto src1 = MakePoints(datas, params1);
      auto T1 = ICPMatch(tgt1, src1, params1);
      rms1.push_back((params1._sols[0].back() * aff).translation().norm());

      // Pt2pl ICP Method
      Pt2plICPParameters params2;
      auto tgt2 = MakePoints(datat, params2);
      auto src2 = MakePoints(datas, params2);
      auto T2 = Pt2plICPMatch(tgt2, src2, params2);
      rms2.push_back((params2._sols[0].back() * aff).translation().norm());

      // Symmetric ICP Method
      SICPParameters params3;
      auto tgt3 = MakePoints(datat, params3);
      auto src3 = MakePoints(datas, params3);
      auto T3 = SICPMatch(tgt3, src3, params3);
      rms3.push_back((params3._sols[0].back() * aff).translation().norm());

      // P2D-NDT Method
      P2DNDTParameters params4;
      params4.r_variance = params4.t_variance = 0;
      params4.cell_size = cell_size;
      auto tgt4 = MakeNDTMap(datat, params4);
      auto src4 = MakePoints(datas, params4);
      auto T4 = P2DNDTMatch(tgt4, src4, params4);
      rms4.push_back((params4._sols[0].back() * aff).translation().norm());

      // D2D-NDT Method
      D2DNDTParameters params5;
      params5.r_variance = params5.t_variance = 0;
      params5.cell_size = cell_size;
      auto tgt5 = MakeNDTMap(datat, params5);
      auto src5 = MakeNDTMap(datas, params5);
      auto T5 = D2DNDTMatch(tgt5, src5, params5);
      rms5.push_back((params5._sols[0].back() * aff).translation().norm());

      // Symmetric NDT Method
      // SNDTParameters params6;
      D2DNDTParameters params6;
      params6.r_variance = params6.t_variance = 0;
      params6.cell_size = cell_size;
      auto tgt6 = MakeNDTMap(datat, params6);
      auto src6 = MakeNDTMap(datas, params6);
      auto T6 = SNDTMatch2(tgt6, src6, params6);
      rms6.push_back((params6._sols[0].back() * aff).translation().norm());
    }
    bar.finish();
    rmsavgs1.push_back(Avg(rms1));
    rmsavgs2.push_back(Avg(rms2));
    rmsavgs3.push_back(Avg(rms3));
    rmsavgs4.push_back(Avg(rms4));
    rmsavgs5.push_back(Avg(rms5));
    rmsavgs6.push_back(Avg(rms6));

    // Paper rs
    rmsavgs1.back() /= rmax;
    rmsavgs2.back() /= rmax;
    rmsavgs3.back() /= rmax;
    rmsavgs4.back() /= rmax;
    rmsavgs5.back() /= rmax;
    rmsavgs6.back() /= rmax;
  }

  // Paper rs
  for (size_t i = 0; i < rs.size(); ++i) rs[i] = pow(10, -2.0 + 0.1 * i);

  string filepath = "/tmp/evalcon.txt";
  ofstream fout(filepath);
  for (size_t i = 0; i < rs.size(); ++i)
    fout << rs[i] << ((i + 1 == rs.size()) ? "\n" : ", ");
  for (auto rmsavgs :
       {rmsavgs1, rmsavgs2, rmsavgs3, rmsavgs4, rmsavgs5, rmsavgs6})
    for (size_t i = 0; i < rmsavgs.size(); ++i)
      fout << rmsavgs[i] << ((i + 1 == rmsavgs.size()) ? "\n" : ", ");
  fout.close();

  cerr << "Generate Figure...";
  string python = PYTHONPATH;
  string script = JoinPath(WSPATH, "src/sndt_exec/scripts/evalcon.py");
  string output = JoinPath(
      WSPATH, "src/sndt_exec/output/cvg-" + GetCurrentTimeAsString() + ".png");
  system((python + " " + script + " " + filepath + " " + output).c_str());
  cerr << " Done!" << endl;
}