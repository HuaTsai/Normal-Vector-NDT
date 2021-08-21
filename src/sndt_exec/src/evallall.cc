#include <ros/ros.h>
#include <sndt/matcher.h>
#include <sndt/visuals.h>
#include <sensor_msgs/CompressedImage.h>
#include <boost/program_options.hpp>
#include <sndt_exec/wrapper.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <rosbag/bag.h>
#include <tqdm/tqdm.h>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;


nav_msgs::Path gtpath;

template <typename T>
double Avg(const T &c) {
  return accumulate(c.begin(), c.end(), 0.) / c.size();
}

template <typename T>
double Avg2(const T &c) {
  return accumulate(c.begin(), c.end(), 0.) / c.size() / 1000;
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

Affine2d GetBenchMark(const ros::Time &t1, const ros::Time &t2) {
  auto pigt = GetPose(gtpath.poses, t1);
  auto pngt = GetPose(gtpath.poses, t2);
  Affine3d Tigt3, Tngt3;
  tf2::fromMsg(pigt, Tigt3);
  tf2::fromMsg(pngt, Tngt3);
  auto Tingt3 = Tigt3.inverse() * Tngt3;
  Tingt3 = Conserve2DFromAffine3d(Tingt3);
  Affine2d ret =
    Translation2d(Tingt3.translation()(0), Tingt3.translation()(1)) *
    Rotation2Dd(Tingt3.rotation().block<2, 2>(0, 0));
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
      ("n,n", po::value<int>(&n)->default_value(-1), "n");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  po::notify(vm);

  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath(data), "lidar.ser"), vpc);
  SerializationInput(JoinPath(GetDataPath(data), "gt.ser"), gtpath);

  // |  i-1  |   i   |  i+1  | -> actual id
  // |   m   |   i   |   n   | -> symbol id
  //         |- tgt -|- src -|
  //         <<----- T ----->>
  //           Tr(bf)  Tr(af)
  vector<int> it1, it2, it3;
  vector<int> iit1, iit2, iit3;
  vector<int> ndt1, ndt2, ndt3;
  vector<int> nm1, nm2, nm3;
  vector<int> opt1, opt2, opt3;
  vector<int> oth1, oth2, oth3;
  vector<int> bud1, bud2, bud3;
  vector<int> ttl1, ttl2, ttl3;
  vector<double> rerr1, rerr2, rerr3;
  vector<double> terr1, terr2, terr3;
  if (n == -1)
    n = vpc.size() - 1;

  Affine2d Tg1 = Affine2d::Identity();
  Affine2d Tg2 = Affine2d::Identity();
  Affine2d Tg3 = Affine2d::Identity();

  tqdm bar;
  for (int i = 0; i < n; ++i) {
    bar.progress(i, n);

    auto tgt = PCMsgTo2D(vpc[i], voxel);
    auto src = PCMsgTo2D(vpc[i + 1], voxel);
    vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, aff2}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{{src, aff2}};

    auto Tingt = GetBenchMark(vpc[i].header.stamp, vpc[i + 1].header.stamp);

    // SNDT Method
    SNDTParameters params1;
    params1.r_variance = params1.t_variance = 0, params1.cell_size = cell_size, params1.huber = huber;
    auto tgt1 = MakeSNDTMap(datat, params1);
    auto src1 = MakeSNDTMap(datas, params1);
    auto T1 = SNDTMatch(tgt1, src1, params1, Tg1);
    Tg1 = T1;
    ndt1.push_back(params1._usedtime.ndt);
    nm1.push_back(params1._usedtime.normal);
    opt1.push_back(params1._usedtime.optimize);
    bud1.push_back(params1._usedtime.build);
    oth1.push_back(params1._usedtime.others);
    ttl1.push_back(params1._usedtime.total());
    auto err1 = TransNormRotDegAbsFromMatrix3d((Tingt.inverse() * T1).matrix());
    rerr1.push_back(err1(0));
    terr1.push_back(err1(1));
    it1.push_back(params1._iteration);
    iit1.push_back(params1._ceres_iteration);

    // NDT method
    // cout << "NDT: ";
    NDTD2DParameters params2;
    params2.r_variance = params2.t_variance = 0, params2.cell_size = cell_size, params2.huber = huber;
    auto tgt2 = MakeNDTMap(datat, params2);
    auto src2 = MakeNDTMap(datas, params2);
    auto T2 = NDTD2DMatch(tgt2, src2, params2, Tg2);
    Tg2 = T2;
    ndt2.push_back(params2._usedtime.ndt);
    nm2.push_back(params2._usedtime.normal);
    opt2.push_back(params2._usedtime.optimize);
    bud2.push_back(params2._usedtime.build);
    oth2.push_back(params2._usedtime.others);
    ttl2.push_back(params2._usedtime.total());
    auto err2 = TransNormRotDegAbsFromMatrix3d((Tingt.inverse() * T2).matrix());
    rerr2.push_back(err2(0));
    terr2.push_back(err2(1));
    it2.push_back(params2._iteration);
    iit2.push_back(params2._ceres_iteration);

    // SICP method
    // cout << "SICP: ";
    SICPParameters params3;
    params3.huber = huber;
    auto tgt3 = MakePoints(datat, params3);
    auto src3 = MakePoints(datas, params3);
    auto T3 = SICPMatch(tgt3, src3, params3, Tg3);
    Tg3 = T3;
    ndt3.push_back(params3._usedtime.ndt);
    nm3.push_back(params3._usedtime.normal);
    opt3.push_back(params3._usedtime.optimize);
    bud3.push_back(params3._usedtime.build);
    oth3.push_back(params3._usedtime.others);
    ttl3.push_back(params3._usedtime.total());
    auto err3 = TransNormRotDegAbsFromMatrix3d((Tingt.inverse() * T3).matrix());
    rerr3.push_back(err3(0));
    terr3.push_back(err3(1));
    it3.push_back(params3._iteration);
    iit3.push_back(params3._ceres_iteration);
  }
  bar.finish();

  ::printf("\n%s\n", data.c_str());
  ::printf("ndt, nm, bud, opt, oth\n");
  ::printf("SNDT: [%.2f, %.2f, %.2f, %.2f, %.2f], %.2f, terr: %.6f, rerr: %.6f, it: %.2f, iit: %.2f\n",
      Avg2(ndt1), Avg2(nm1), Avg2(bud1), Avg2(opt1), Avg2(oth1), Avg2(ttl1), Avg(terr1), Avg(rerr1), Avg(it1), Avg(iit1));

  ::printf(" NDT: [%.2f, %.2f, %.2f, %.2f, %.2f], %.2f, terr: %.6f, rerr: %.6f, it: %.2f, iit: %.2f\n",
      Avg2(ndt2), Avg2(nm2), Avg2(bud2), Avg2(opt2), Avg2(oth2), Avg2(ttl2), Avg(terr2), Avg(rerr2), Avg(it2), Avg(iit2));

  ::printf("SICP: [%.2f, %.2f, %.2f, %.2f, %.2f], %.2f, terr: %.6f, rerr: %.6f, it: %.2f, iit: %.2f\n",
      Avg2(ndt3), Avg2(nm3), Avg2(bud3), Avg2(opt3), Avg2(oth3), Avg2(ttl3), Avg(terr3), Avg(rerr3), Avg(it3), Avg(iit3));
}