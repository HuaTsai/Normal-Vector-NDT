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

// #define LIDAR

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
  return accumulate(c.begin(), c.end(), 0.) / c.size();
}

template <typename T>
void PrintValues(const T &coll) {
  copy(coll.begin(), coll.end(), ostream_iterator<typename T::value_type>(cout, ", "));
  cout << endl;
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
  Affine3d To, Ti;
  tf2::fromMsg(GetPose(gtpath.poses, t2), To);
  tf2::fromMsg(GetPose(gtpath.poses, t1), Ti);
  Affine3d Tgt3 = Conserve2DFromAffine3d(Ti.inverse() * To);
  Affine2d ret = Translation2d(Tgt3.translation()(0), Tgt3.translation()(1)) *
                 Rotation2Dd(Tgt3.rotation().block<2, 2>(0, 0));
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
  vector<common::EgoPointClouds> vepcs;
  SerializationInput(JoinPath(GetDataPath(data), "lidar.ser"), vpc);
  SerializationInput(JoinPath(GetDataPath(data), "gt.ser"), gtpath);
  SerializationInput(JoinPath(GetDataPath(data), "vepcs.ser"), vepcs);

  // |  i-1  |   i   |  i+1  | -> actual id
  // |   m   |   i   |   n   | -> symbol id
  //         |- tgt -|- src -|
  //         <<----- T ----->>
  //           Tr(bf)  Tr(af)
  vector<int> it1, it2, it3, it4, it5;
  vector<int> iit1, iit2, iit3, iit4, iit5;
  vector<double> ndt1, ndt2, ndt3, ndt4, ndt5;
  vector<double> nm1, nm2, nm3, nm4, nm5;
  vector<double> bud1, bud2, bud3, bud4, bud5;
  vector<double> opt1, opt2, opt3, opt4, opt5;
  vector<double> oth1, oth2, oth3, oth4, oth5;
  vector<double> ttl1, ttl2, ttl3, ttl4, ttl5;
  vector<double> terr1, terr2, terr3, terr4, terr5;
  vector<double> rerr1, rerr2, rerr3, rerr4, rerr5;
  if (n == -1)
    n = vpc.size() - 1;

#ifdef LIDAR
  for (int i = 0; i < n; ++i) {
    auto tgt = PCMsgTo2D(vpc[i], voxel);
    auto src = PCMsgTo2D(vpc[i + 1], voxel);
    vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, aff2}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{{src, aff2}};
    auto Tgt = GetBenchMark(vpc[i].header.stamp, vpc[i + 1].header.stamp);
    Affine2d Tg = Affine2d::Identity();
#else
  int f = 5;
  n = vepcs.size() / f * f;
  // for (int i = 0; i < n - f; i += f) {
  for (int i = 55; i < 56; ++i) {
    Affine2d Tio, Toq;
    vector<Affine2d> Tios, Toqs;
    auto datat = Augment(vepcs, i, i + f - 1, Tio, Tios);
    auto datas = Augment(vepcs, i + f, i + 2 * f - 1, Toq, Toqs);
    auto Tgt = GetBenchMark(vepcs[i].stamp, vepcs[i + f].stamp);
    Affine2d Tg = Tio;
#endif

    // SNDT Method
    SNDTParameters params1;
#ifdef LIDAR
    params1.r_variance = params1.t_variance = 0;
#endif
    params1.verbose = true;
    params1.cell_size = cell_size, params1.huber = huber;
    auto tgt1 = MakeSNDTMap(datat, params1);
    auto src1 = MakeSNDTMap(datas, params1);
    std::cerr << "Match: " << i << std::endl;
    auto T1 = SNDTMatch(tgt1, src1, params1, Tg);
    ndt1.push_back(params1._usedtime.ndt / 1000.);
    nm1.push_back(params1._usedtime.normal / 1000.);
    bud1.push_back(params1._usedtime.build / 1000.);
    opt1.push_back(params1._usedtime.optimize / 1000.);
    oth1.push_back(params1._usedtime.others / 1000.);
    ttl1.push_back(params1._usedtime.total() / 1000.);
    auto err1 = TransNormRotDegAbsFromMatrix3d((Tgt.inverse() * T1).matrix());
    terr1.push_back(err1(0));
    rerr1.push_back(err1(1));
    it1.push_back(params1._iteration);
    iit1.push_back(params1._ceres_iteration);

    // NDT method
    NDTD2DParameters params2;
#ifdef LIDAR
    params2.r_variance = params2.t_variance = 0;
#endif
    params2.cell_size = cell_size, params2.huber = huber;
    auto tgt2 = MakeNDTMap(datat, params2);
    auto src2 = MakeNDTMap(datas, params2);
    auto T2 = NDTD2DMatch(tgt2, src2, params2, Tg);
    ndt2.push_back(params2._usedtime.ndt / 1000.);
    nm2.push_back(params2._usedtime.normal / 1000.);
    bud2.push_back(params2._usedtime.build / 1000.);
    opt2.push_back(params2._usedtime.optimize / 1000.);
    oth2.push_back(params2._usedtime.others / 1000.);
    ttl2.push_back(params2._usedtime.total() / 1000.);
    auto err2 = TransNormRotDegAbsFromMatrix3d((Tgt.inverse() * T2).matrix());
    terr2.push_back(err2(0));
    rerr2.push_back(err2(1));
    it2.push_back(params2._iteration);
    iit2.push_back(params2._ceres_iteration);

    // SICP method
    SICPParameters params3;
    params3.huber = huber;
    auto tgt3 = MakePoints(datat, params3);
    auto src3 = MakePoints(datas, params3);
    auto T3 = SICPMatch(tgt3, src3, params3, Tg);
    ndt3.push_back(params3._usedtime.ndt / 1000.);
    nm3.push_back(params3._usedtime.normal / 1000.);
    bud3.push_back(params3._usedtime.build / 1000.);
    opt3.push_back(params3._usedtime.optimize / 1000.);
    oth3.push_back(params3._usedtime.others / 1000.);
    ttl3.push_back(params3._usedtime.total() / 1000.);
    auto err3 = TransNormRotDegAbsFromMatrix3d((Tgt.inverse() * T3).matrix());
    terr3.push_back(err3(0));
    rerr3.push_back(err3(1));
    it3.push_back(params3._iteration);
    iit3.push_back(params3._ceres_iteration);

    // ICP method
    ICPParameters params4;
    params4.huber = huber;
    auto tgt4 = MakePoints(datat, params4);
    auto src4 = MakePoints(datas, params4);
    auto T4 = ICPMatch(tgt4, src4, params4, Tg);
    ndt4.push_back(params4._usedtime.ndt / 1000.);
    nm4.push_back(params4._usedtime.normal / 1000.);
    bud4.push_back(params4._usedtime.build / 1000.);
    opt4.push_back(params4._usedtime.optimize / 1000.);
    oth4.push_back(params4._usedtime.others / 1000.);
    ttl4.push_back(params4._usedtime.total() / 1000.);
    auto err4 = TransNormRotDegAbsFromMatrix3d((Tgt.inverse() * T4).matrix());
    terr4.push_back(err4(0));
    rerr4.push_back(err4(1));
    it4.push_back(params4._iteration);
    iit4.push_back(params4._ceres_iteration);

    // Point-to-Plane ICP method
    Pt2plICPParameters params5;
    params5.huber = huber;
    auto tgt5 = MakePoints(datat, params5);
    auto src5 = MakePoints(datas, params5);
    auto T5 = Pt2plICPMatch(tgt5, src5, params5, Tg);
    ndt5.push_back(params5._usedtime.ndt / 1000.);
    nm5.push_back(params5._usedtime.normal / 1000.);
    bud5.push_back(params5._usedtime.build / 1000.);
    opt5.push_back(params5._usedtime.optimize / 1000.);
    oth5.push_back(params5._usedtime.others / 1000.);
    ttl5.push_back(params5._usedtime.total() / 1000.);
    auto err5 = TransNormRotDegAbsFromMatrix3d((Tgt.inverse() * T5).matrix());
    terr5.push_back(err5(0));
    rerr5.push_back(err5(1));
    it5.push_back(params5._iteration);
    iit5.push_back(params5._ceres_iteration);
  }
  // bar.finish();

  PrintValues(ndt1);
  PrintValues(nm1);
  PrintValues(bud1);
  PrintValues(opt1);
  PrintValues(ttl1);
  PrintValues(terr1);
  PrintValues(rerr1);
  PrintValues(it1);
  PrintValues(iit1);

  PrintValues(ndt2);
  PrintValues(nm2);
  PrintValues(bud2);
  PrintValues(opt2);
  PrintValues(ttl2);
  PrintValues(terr2);
  PrintValues(rerr2);
  PrintValues(it2);
  PrintValues(iit2);

  PrintValues(ndt3);
  PrintValues(nm3);
  PrintValues(bud3);
  PrintValues(opt3);
  PrintValues(ttl3);
  PrintValues(terr3);
  PrintValues(rerr3);
  PrintValues(it3);
  PrintValues(iit3);

  PrintValues(ndt4);
  PrintValues(nm4);
  PrintValues(bud4);
  PrintValues(opt4);
  PrintValues(ttl4);
  PrintValues(terr4);
  PrintValues(rerr4);
  PrintValues(it4);
  PrintValues(iit4);

  PrintValues(ndt5);
  PrintValues(nm5);
  PrintValues(bud5);
  PrintValues(opt5);
  PrintValues(ttl5);
  PrintValues(terr5);
  PrintValues(rerr5);
  PrintValues(it5);
  PrintValues(iit5);

  // ::printf("SNDT: [%.2f, %.2f, %.2f, %.2f, %.2f], %.2f, terr: %.6f, rerr: %.6f, it: %.2f, iit: %.2f\n",
  //     Avg2(ndt1), Avg2(nm1), Avg2(bud1), Avg2(opt1), Avg2(oth1), Avg2(ttl1), Avg(terr1), Avg(rerr1), Avg(it1), Avg(iit1));

  // ::printf(" NDT: [%.2f, %.2f, %.2f, %.2f, %.2f], %.2f, terr: %.6f, rerr: %.6f, it: %.2f, iit: %.2f\n",
  //     Avg2(ndt2), Avg2(nm2), Avg2(bud2), Avg2(opt2), Avg2(oth2), Avg2(ttl2), Avg(terr2), Avg(rerr2), Avg(it2), Avg(iit2));

  // ::printf("SICP: [%.2f, %.2f, %.2f, %.2f, %.2f], %.2f, terr: %.6f, rerr: %.6f, it: %.2f, iit: %.2f\n",
  //     Avg2(ndt3), Avg2(nm3), Avg2(bud3), Avg2(opt3), Avg2(oth3), Avg2(ttl3), Avg(terr3), Avg(rerr3), Avg(it3), Avg(iit3));
}