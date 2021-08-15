#include <common/EgoPointClouds.h>
#include <common/common.h>
#include <geometry_msgs/Vector3.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sndt/matcher.h>
#include <sndt/visuals.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int32.h>

#include <boost/program_options.hpp>
#include <sndt_exec/wrapper.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;
using namespace Eigen;
using namespace visualization_msgs;
namespace po = boost::program_options;

vector<sensor_msgs::CompressedImage> imb, imbl, imbr, imf, imfl, imfr;
vector<common::EgoPointClouds> vepcs;
vector<sensor_msgs::PointCloud2> vpc;
ros::Publisher pub1, pub2, pub3, pub4, pub5, pub6, pub7, pubd, pube;
ros::Publisher pb1, pb2, pb3, pb4, pb5, pb6, pbs, pbt;
double cell_size, radius, huber;
bool verbose;
nav_msgs::Path gtpath;

vector<Vector2d> PCMsgTo2D(const sensor_msgs::PointCloud2 &msg) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(msg, *pc);
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr fpc(new pcl::PointCloud<pcl::PointXYZ>);
  vg.setInputCloud(pc);
  vg.setLeafSize(1, 1, 1);
  vg.filter(*fpc);

  vector<Vector2d> ret;
  for (const auto &pt : *fpc)
    if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z) && pt.z > -0.9)
      ret.push_back(Vector2d(pt.x, pt.y));
  return ret;
}

Affine2d GetDiffT(const vector<common::EgoPointClouds> &vepcs,
                   const ros::Time &t1, const ros::Time &t2) {
  if (t1 <= vepcs[0].stamp || t2 >= vepcs[vepcs.size() - 1].stamp)
    return Affine2d::Identity();
  auto it1 = lower_bound(vepcs.begin(), vepcs.end(), t1, [](auto a, auto b) { return a.stamp < b; });
  auto it2 = lower_bound(vepcs.begin(), vepcs.end(), t2, [](auto a, auto b) { return a.stamp < b; });
  Vector3d xyt = Vector3d::Zero();
  if (it1 == it2)
    xyt += Vector3d(it1->vxyt[0], it1->vxyt[1], it1->vxyt[2]) * (t2 - t1).toSec();
  for (auto it = it1; it != it2; ++it) {
    Vector3d v(it->vxyt[0], it->vxyt[1], it->vxyt[2]);
    double dt;
    if (it == it1)
      dt = ((it + 1)->stamp - t1).toSec();
    else if (it == it2)
      dt = (t2 - it->stamp).toSec();
    else
      dt = ((it + 1)->stamp - it->stamp).toSec();
    xyt += v * dt;
  }
  return Rotation2Dd(xyt[2]) * Translation2d(xyt[0], xyt[1]);
}

void cb(const std_msgs::Int32 &num) {
  Affine3d aff3 =
      Translation3d(0.943713, 0.000000, 1.840230) *
      Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 =
      Translation2d(aff3.translation()(0), aff3.translation()(1)) *
      Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  int i = num.data;
  SNDTParameters params;
  Affine2d Tio, Toq;
  vector<Affine2d> Tios, Toqs;
  auto tgt = PCMsgTo2D(vpc[i]);
  auto src = PCMsgTo2D(vpc[i + 1]);
  vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, aff2}};
  vector<pair<vector<Vector2d>, Affine2d>> datas{{src, aff2}};

  // auto Tg = GetDiffT(vepcs, vpc[i].header.stamp, vpc[i + 1].header.stamp);
  auto Tg = Affine2d::Identity();

  // SNDT Method
  auto t1 = GetTime();
  SNDTParameters params1;
  params1.r_variance = params1.t_variance = 0;
  params1.verbose = verbose;
  auto mapt1 = MakeSNDTMap(datat, params1);
  auto maps1 = MakeSNDTMap(datas, params1);
  auto T1 = SNDTMatch(mapt1, maps1, params1, Tg);

  // NDTD2D method
  NDTD2DParameters params2;
  params2.r_variance = params2.t_variance = 0;
  params2.verbose = verbose;
  auto mapt2 = MakeNDTMap(datat, params2);
  auto maps2 = MakeNDTMap(datas, params2);
  auto T2 = NDTD2DMatch(mapt2, maps2, params2, Tg);

  // SICP method
  vector<Vector2d> tgt2(tgt.size()), src2(src.size());
  transform(tgt.begin(), tgt.end(), tgt2.begin(), [&aff2](auto p) { return aff2 * p; });
  transform(src.begin(), src.end(), src2.begin(), [&aff2](auto p) { return aff2 * p; });
  SICPParameters params3;
  params3.verbose = verbose;
  auto T3 = SICPMatch(tgt2, src2, params3, Tg);
  auto t2 = GetTime();
  cout << GetDiffTime(t1, t2) << " ms" << endl;

  /********* Compute Ground Truth *********/
  Affine3d To, Ti;
  tf2::fromMsg(GetPose(gtpath.poses, vpc[i + 1].header.stamp), To);
  tf2::fromMsg(GetPose(gtpath.poses, vpc[i].header.stamp), Ti);
  Affine3d Tgt3 = Conserve2DFromAffine3d(Ti.inverse() * To);
  Affine2d Tgt = Translation2d(Tgt3.translation()(0), Tgt3.translation()(1)) *
                 Rotation2Dd(Tgt3.rotation().block<2, 2>(0, 0));
  /********* Compute End Here     *********/

  cout << "sndt: " << TransNormRotDegAbsFromMatrix3d((T1.inverse() * Tgt).matrix()).transpose() << endl;
  cout << " ndt: " << TransNormRotDegAbsFromMatrix3d((T2.inverse() * Tgt).matrix()).transpose() << endl;
  cout << "sicp: " << TransNormRotDegAbsFromMatrix3d((T3.inverse() * Tgt).matrix()).transpose() << endl;

  pub1.publish(JoinMarkers({MarkerOfPoints(tgt2, 0.5, Color::kRed)}));
  pub2.publish(JoinMarkers({MarkerOfPoints(src2)}));

  auto mps1 = maps1.GetPoints();
  transform(mps1.begin(), mps1.end(), mps1.begin(), [&T1](auto p) { return T1 * p; });
  pub3.publish(JoinMarkers({MarkerOfPoints(mps1)}));  // SNDT

  auto mps2 = maps2.GetPoints();
  transform(mps2.begin(), mps2.end(), mps2.begin(), [&T2](auto p) { return T2 * p; });
  pub4.publish(JoinMarkers({MarkerOfPoints(mps2)}));  // NDTD2D

  transform(src2.begin(), src2.end(), src2.begin(), [&T3](auto p) { return T3 * p; });
  pub5.publish(JoinMarkers({MarkerOfPoints(src2)}));  // SICP

  auto stmp = vepcs[i].stamp;
  auto cmp = [](sensor_msgs::CompressedImage a, ros::Time b) {
    return a.header.stamp < b;
  };
  pb1.publish(*lower_bound(imb.begin(), imb.end(), stmp, cmp));
  pb2.publish(*lower_bound(imbl.begin(), imbl.end(), stmp, cmp));
  pb3.publish(*lower_bound(imbr.begin(), imbr.end(), stmp, cmp));
  pb4.publish(*lower_bound(imf.begin(), imf.end(), stmp, cmp));
  pb5.publish(*lower_bound(imfl.begin(), imfl.end(), stmp, cmp));
  pb6.publish(*lower_bound(imfr.begin(), imfr.end(), stmp, cmp));
}

void GetFiles(string data) {
  auto base = GetDataPath(data);
  SerializationInput(JoinPath(base, "lidar.ser"), vpc);
  SerializationInput(JoinPath(base, "vepcs.ser"), vepcs);
  SerializationInput(JoinPath(base, "gt.ser"), gtpath);
  SerializationInput(JoinPath(base, "back.ser"), imb);
  SerializationInput(JoinPath(base, "back_left.ser"), imbl);
  SerializationInput(JoinPath(base, "back_right.ser"), imbr);
  SerializationInput(JoinPath(base, "front.ser"), imf);
  SerializationInput(JoinPath(base, "front_left.ser"), imfl);
  SerializationInput(JoinPath(base, "front_right.ser"), imfr);
}

int main(int argc, char **argv) {
  string data;
  bool run;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("radius,r", po::value<double>(&radius)->default_value(1.5), "Radius")
      ("huber,u", po::value<double>(&huber)->default_value(1), "Huber")
      ("verbose,v", po::value<bool>(&verbose)->default_value(false)->implicit_value(true), "Verbose")
      ("run", po::value<bool>(&run)->default_value(false)->implicit_value(true), "Run");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  ros::init(argc, argv, "testl_match");
  ros::NodeHandle nh;

  GetFiles(data);

  ros::Subscriber sub = nh.subscribe("idx", 0, cb);

  pub1 = nh.advertise<MarkerArray>("markers1", 0, true);
  pub2 = nh.advertise<MarkerArray>("markers2", 0, true);
  pub3 = nh.advertise<MarkerArray>("markers3", 0, true);
  pub4 = nh.advertise<MarkerArray>("markers4", 0, true);
  pub5 = nh.advertise<MarkerArray>("markers5", 0, true);
  pub6 = nh.advertise<MarkerArray>("markers6", 0, true);
  pub7 = nh.advertise<MarkerArray>("markers7", 0, true);
  pubd = nh.advertise<Marker>("markerd", 0, true);
  pbs = nh.advertise<Marker>("marker1", 0, true);
  pbt = nh.advertise<Marker>("marker2", 0, true);
  pube = nh.advertise<geometry_msgs::Vector3>("err", 0, true);
  pb1 = nh.advertise<sensor_msgs::CompressedImage>("back/compressed", 0, true);
  pb2 = nh.advertise<sensor_msgs::CompressedImage>("back_left/compressed", 0, true);
  pb3 = nh.advertise<sensor_msgs::CompressedImage>("back_right/compressed", 0, true);
  pb4 = nh.advertise<sensor_msgs::CompressedImage>("front/compressed", 0, true);
  pb5 = nh.advertise<sensor_msgs::CompressedImage>("front_left/compressed", 0, true);
  pb6 = nh.advertise<sensor_msgs::CompressedImage>("front_right/compressed", 0, true);

  if (run) {
    for (int i = 0; i < (int)vpc.size() - 2; ++i) {
      std_msgs::Int32 num;
      num.data = i;
      cb(num);
    }
  }

  cout << "Ready!" << endl;

  ros::spin();
}