#include <ros/ros.h>
#include <sndt/matcher.h>
#include <boost/program_options.hpp>
#include <sndt_exec/wrapper.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <rosbag/bag.h>
#include <tqdm/tqdm.h>

#define LIDAR

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

template <typename T>
double Avg(const T &c) {
  return accumulate(c.begin(), c.end(), 0.) / c.size();
}

vector<Vector2d> PCMsgTo2D(const sensor_msgs::PointCloud2 &msg, double voxel) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(msg, *pc);
  if (voxel != 0) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(pc);
    vg.setLeafSize(voxel, voxel, voxel);
    vg.filter(*pc);
  }

  vector<Vector2d> ret;
  for (const auto &pt : *pc)
    if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z))
      ret.push_back(Vector2d(pt.x, pt.y));
  return ret;
}

double RMS(const vector<Vector2d> &tgt, const vector<Vector2d> &src,
           const Affine2d &aff = Affine2d::Identity()) {
  vector<Vector2d> src2(src.size());
  transform(src.begin(), src.end(), src2.begin(), [&aff](auto p) { return aff * p; });
  double ret = 0;
  for (size_t i = 0; i < src2.size(); ++i)
    ret += (src2[i] - tgt[i]).squaredNorm();
  return sqrt(ret / src2.size());
}

void ShowRMS(const CommonParameters &params, const vector<Vector2d> &tgt, const vector<Vector2d> &src) {
  cout << "[";
  for (auto sols : params._sols) {
    for (auto T : sols)
      cout << RMS(tgt, src, T) << ", ";
  }
  cout << " ]";
  cout << endl;
}

int main(int argc, char **argv) {
#ifdef LIDAR
  Affine3d aff3 =
      Translation3d(0.943713, 0.000000, 1.840230) *
      Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 =
      Translation2d(aff3.translation()(0), aff3.translation()(1)) *
      Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));
#endif
  int n;
  double cell_size, radius, huber, voxel;
  string data;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("radius,r", po::value<double>(&radius)->default_value(1.5), "Radius")
      ("huber,u", po::value<double>(&huber)->default_value(1.0), "Use Huber loss")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("n,n", po::value<int>(&n)->default_value(0), "n");
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
  transform(tgt.begin(), tgt.end(), tgt.begin(), [&aff2](auto p) { return aff2 * p; });

  vector<double> rmsavgs1, rmsavgs2, rmsavgs3, rmsavgs4, rmsavgs5, rmsavgs6;
  double rmax = 15;
  vector<double> rs(21);
  for (int i = 0; i <= 20; ++i)
    rs[i] = pow(10, -1. + 0.05 * i);
  for (auto r : rs) {
    RandomTransformGenerator2D rtg(r * rmax);
    int samples = 50;
    auto affs = rtg.Generate(samples);
    // double err = 0.01;
    vector<double> rms1, rms2, rms3, rms4, rms5, rms6;
    tqdm bar;
    cerr << "r = " << r * rmax << endl;
    for (size_t i = 0; i < affs.size(); ++i) {
      auto aff = affs[i];
      bar.progress(i, affs.size());
      std::vector<Eigen::Vector2d> src(tgt.size());
      transform(tgt.begin(), tgt.end(), src.begin(), [&aff](auto p) { return aff * p; });
      vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, Eigen::Affine2d::Identity()}};
      vector<pair<vector<Vector2d>, Affine2d>> datas{{src, Eigen::Affine2d::Identity()}};

      // ICP Method
      ICPParameters params1;
      params1.huber = r;
      auto tgt1 = MakePoints(datat, params1);
      auto src1 = MakePoints(datas, params1);
      auto T1 = ICPMatch(tgt1, src1, params1);
      rms1.push_back(RMS(tgt, src, params1._sols[0].back()) / rmax);
      // if ((T1.translation() + aff.translation()).norm() > err)
      //   cout << " icp:  gt -> " << aff.translation().transpose() << endl
      //        << "      res -> " << T1.translation().transpose() << endl;

      // Pt2pl ICP Method
      Pt2plICPParameters params2;
      params2.huber = r;
      auto tgt2 = MakePoints(datat, params2);
      auto src2 = MakePoints(datas, params2);
      auto T2 = Pt2plICPMatch(tgt2, src2, params2);
      rms2.push_back(RMS(tgt, src, params2._sols[0].back()) / rmax);
      // if ((T2.translation() + aff.translation()).norm() > err)
      //   cout << "nicp:  gt -> " << aff.translation().transpose() << endl
      //        << "      res -> " << T2.translation().transpose() << endl;

      // Symmetric ICP Method
      SICPParameters params3;
      params3.huber = r;
      auto tgt3 = MakePoints(datat, params3);
      auto src3 = MakePoints(datas, params3);
      auto T3 = SICPMatch(tgt3, src3, params3);
      rms3.push_back(RMS(tgt, src, params3._sols[0].back()) / rmax);
      // if ((T3.translation() + aff.translation()).norm() > err)
      //   cout << "sicp:  gt -> " << aff.translation().transpose() << endl
      //        << "      res -> " << T3.translation().transpose() << endl;

      // P2D-NDT Method
      P2DNDTParameters params4;
      params4.r_variance = params4.t_variance = 0;
      params4.cell_size = cell_size, params4.huber = r;
      auto tgt4 = MakeNDTMap(datat, params4);
      auto src4 = MakePoints(datas, params4);
      auto T4 = P2DNDTMatch(tgt4, src4, params4);
      rms4.push_back(RMS(tgt, src, params4._sols[0].back()) / rmax);
      // if ((T4.translation() + aff.translation()).norm() > err)
      //   cout << "pndt:  gt -> " << aff.translation().transpose() << endl
      //        << "      res -> " << T4.translation().transpose() << endl;

      // D2D-NDT Method
      D2DNDTParameters params5;
      params5.r_variance = params5.t_variance = 0;
      params5.cell_size = cell_size, params5.huber = r;
      auto tgt5 = MakeNDTMap(datat, params5);
      auto src5 = MakeNDTMap(datas, params5);
      auto T5 = D2DNDTMatch(tgt5, src5, params5);
      rms5.push_back(RMS(tgt, src, params5._sols[0].back()) / rmax);
      // if ((T5.translation() + aff.translation()).norm() > err)
      //   cout << "dndt:  gt -> " << aff.translation().transpose() << endl
      //        << "      res -> " << T5.translation().transpose() << endl;

      // Symmetric NDT Method
      SNDTParameters params6;
      params6.r_variance = params6.t_variance = 0;
      params6.cell_size = cell_size, params6.huber = r;
      auto tgt6 = MakeSNDTMap(datat, params6);
      auto src6 = MakeSNDTMap(datas, params6);
      auto T6 = SNDTMatch(tgt6, src6, params6);
      rms6.push_back(RMS(tgt, src, params6._sols[0].back()) / rmax);
      // if ((T6.translation() + aff.translation()).norm() > err)
      //   cout << "sndt:  gt -> " << aff.translation().transpose() << endl
      //        << "      res -> " << T6.translation().transpose() << endl;
    }
    bar.finish();
    rmsavgs1.push_back(Avg(rms1));
    rmsavgs2.push_back(Avg(rms2));
    rmsavgs3.push_back(Avg(rms3));
    rmsavgs4.push_back(Avg(rms4));
    rmsavgs5.push_back(Avg(rms5));
    rmsavgs6.push_back(Avg(rms6));
  }

  string filepath = "/tmp/evalcon.txt";
  ofstream fout(filepath);
  for (size_t i = 0; i < rs.size(); ++i)
    fout << rs[i] << ((i + 1 == rs.size()) ? "\n" : ", ");
  for (auto rmsavgs : {rmsavgs1, rmsavgs2, rmsavgs3, rmsavgs4, rmsavgs5, rmsavgs6})
    for (size_t i = 0; i < rmsavgs.size(); ++i)
      fout << rmsavgs[i] << ((i + 1 == rmsavgs.size()) ? "\n" : ", ");
  fout.close();

  cerr << "Generate Figure...";
  string python = PYTHONPATH;
  string script = JoinPath(WSPATH, "src/sndt_exec/scripts/evalcon.py");
  string output = JoinPath(WSPATH, "src/sndt_exec/output/cvg-" + GetCurrentTimeAsString() + ".png");
  system((python + " " + script + " " + filepath + " " + output).c_str());
  cerr << " Done!" << endl;
}