#include <ros/ros.h>
#include <sndt/matcher.h>
#include <boost/program_options.hpp>
#include <sndt_exec/wrapper.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <rosbag/bag.h>
#include <tqdm/tqdm.h>
#include <sndt/visuals.h>
#include <std_msgs/Int32.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace std;
using namespace Eigen;
using namespace visualization_msgs;
namespace po = boost::program_options;

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

int main(int argc, char **argv) {
  Affine3d aff3 =
      Translation3d(0.943713, 0.000000, 1.840230) *
      Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 =
      Translation2d(aff3.translation()(0), aff3.translation()(1)) *
      Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  int n;
  double cell_size, huber, voxel, r, radius;
  string data;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("radius,a", po::value<double>(&radius)->default_value(1.5), "radius")
      ("n,n", po::value<int>(&n)->default_value(0), "n")
      ("r,r", po::value<double>(&r)->default_value(15), "r");
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

  vector<vector<int>> ss(6);
  tqdm bar;
  for (int r = 5; r <= 25; ++r) {
    bar.progress(r, 25);
    int samples = 99;
    auto affs = RandomTransformGenerator2D(r).Generate(samples);
    int s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
    tqdm bar;
    for (size_t i = 0; i < affs.size(); ++i) {
      // bar.progress(i, affs.size());
      auto aff = affs[i];
      std::vector<Eigen::Vector2d> src(tgt.size());
      transform(tgt.begin(), tgt.end(), src.begin(), [&aff](auto p) { return aff * p; });
      vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, Eigen::Affine2d::Identity()}};
      vector<pair<vector<Vector2d>, Affine2d>> datas{{src, Eigen::Affine2d::Identity()}};

      ICPParameters params1;
      auto tgt1 = MakePoints(datat, params1);
      auto src1 = MakePoints(datas, params1);
      auto T1 = ICPMatch(tgt1, src1, params1);
      s1 += ((aff * T1).translation().isZero(1) ? 1 : 0);

      Pt2plICPParameters params2;
      auto tgt2 = MakePoints(datat, params2);
      auto src2 = MakePoints(datas, params2);
      auto T2 = Pt2plICPMatch(tgt2, src2, params2);
      s2 += ((aff * T2).translation().isZero(1) ? 1 : 0);

      SICPParameters params3;
      auto tgt3 = MakePoints(datat, params3);
      auto src3 = MakePoints(datas, params3);
      auto T3 = SICPMatch(tgt3, src3, params3);
      s3 += ((aff * T3).translation().isZero(1) ? 1 : 0);

      P2DNDTParameters params4;
      params4.r_variance = params4.t_variance = 0;
      auto tgt4 = MakeNDTMap(datat, params4);
      auto src4 = MakePoints(datas, params4);
      auto T4 = P2DNDTMatch(tgt4, src4, params4);
      s4 += ((aff * T4).translation().isZero(1) ? 1 : 0);

      D2DNDTParameters params5;
      params5.r_variance = params5.t_variance = 0;
      auto tgt5 = MakeNDTMap(datat, params5);
      auto src5 = MakeNDTMap(datas, params5);
      auto T5 = D2DNDTMatch(tgt5, src5, params5);
      s5 += ((aff * T5).translation().isZero(1) ? 1 : 0);

      SNDTParameters params6;
      params6.r_variance = params6.t_variance = 0;
      auto tgt6 = MakeSNDTMap(datat, params6);
      auto src6 = MakeSNDTMap(datas, params6);
      auto T6 = SNDTMatch(tgt6, src6, params6);
      s6 += ((aff * T6).translation().isZero(1) ? 1 : 0);

      ss[0].push_back(s1);
      ss[1].push_back(s2);
      ss[2].push_back(s3);
      ss[3].push_back(s4);
      ss[4].push_back(s5);
      ss[5].push_back(s6);
    }
  }
  bar.finish();
  ::printf("05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25\n");
  copy(ss[0].begin(), ss[0].end(), ostream_iterator<int>(cout, ", "));  cout << endl;
  copy(ss[1].begin(), ss[1].end(), ostream_iterator<int>(cout, ", "));  cout << endl;
  copy(ss[2].begin(), ss[2].end(), ostream_iterator<int>(cout, ", "));  cout << endl;
  copy(ss[3].begin(), ss[3].end(), ostream_iterator<int>(cout, ", "));  cout << endl;
  copy(ss[4].begin(), ss[4].end(), ostream_iterator<int>(cout, ", "));  cout << endl;
  copy(ss[5].begin(), ss[5].end(), ostream_iterator<int>(cout, ", "));  cout << endl;
}