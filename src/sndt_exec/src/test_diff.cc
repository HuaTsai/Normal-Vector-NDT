/**
 * @file test_diff.cc
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2021-08-01
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include <common/EgoPointClouds.h>
#include <common/common.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sndt/matcher.h>
#include <sndt/visuals.h>
#include <std_msgs/Int32.h>
#include <nav_msgs/Path.h>

#include <boost/program_options.hpp>
#include <sndt_exec/wrapper.hpp>

using namespace std;
using namespace Eigen;
using namespace visualization_msgs;
namespace po = boost::program_options;

vector<sensor_msgs::CompressedImage> imb, imbl, imbr, imf, imfl, imfr;
vector<common::EgoPointClouds> vepcs;  
ros::Publisher pub1, pub2, pub3, pub4, pub5, pub6, pub7, pub8, pube;
ros::Publisher pb1, pb2, pb3, pb4, pb5, pb6, pbs, pbt;
int frames;
double cell_size, radius, rvar, tvar;
double huber;
nav_msgs::Path gtpath;
std_msgs::Float64MultiArray errs;

pair<int, double> FindMax(double a, double b, double c) {
  if (a > b && a > c)
    return {0, a};
  if (b > a && b > c)
    return {1, b};
  return {2, c};
}

pair<int, double> FindMin(double a, double b, double c) {
  if (a < b && a < c)
    return {0, a};
  if (b < a && b < c)
    return {1, b};
  return {2, c};
}

// start, ..., end, end+1
// <<------- T -------->>
vector<pair<MatrixXd, Affine2d>> Augment(
    const vector<common::EgoPointClouds> &vepcs, int start, int end,
    Affine2d &T, vector<Affine2d> &allT) {
  vector<pair<MatrixXd, Affine2d>> ret;
  double dx = 0, dy = 0, dth = 0;
  for (int i = start; i <= end; ++i) {
    Affine2d T0i = Rotation2Dd(dth) * Translation2d(dx, dy);
    allT.push_back(T0i);
    for (const auto &pc : vepcs[i].pcs) {
      MatrixXd fi(2, pc.points.size());
      for (int i = 0; i < fi.cols(); ++i)
        fi.col(i) = Vector2d(pc.points[i].x, pc.points[i].y);
      Affine3d aff;
      tf2::fromMsg(pc.origin, aff);
      Matrix3d mtx = Matrix3d::Identity();
      mtx.block<2, 2>(0, 0) = aff.matrix().block<2, 2>(0, 0);
      mtx.block<2, 1>(0, 2) = aff.matrix().block<2, 1>(0, 3);
      ret.push_back({fi, T0i * Affine2d(mtx)});
    }
    double dt = (vepcs[i + 1].stamp - vepcs[i].stamp).toSec();
    dx += vepcs[i].vxyt[0] * dt;
    dy += vepcs[i].vxyt[1] * dt;
    dth += vepcs[i].vxyt[2] * dt;
  }
  T = Rotation2Dd(dth) * Translation2d(dx, dy);
  return ret;
}

// i-f, ..., i-1 | i, i+1, ..., i+f-1 | i+f, ..., i+2f-1 | i+2f -> actual id
// ..., ...,  m  | i, ..., ...,   n   |  o , ...,   p    |  q   -> symbol id
// -- frames  -- | ----- target ----- | ---- source ---- |
//                 <<---- Computed T ---->>
//                Tr (before iter)      Tr (after iter)
void cb(const std_msgs::Int32 &num) {
  static int idx = 0;
  int i = num.data;
  int f = frames;
  Affine2d Tio, Toq;
  vector<Affine2d> Tios, Toqs;

  auto datat = Augment(vepcs, i, i + f - 1, Tio, Tios);
  auto datas = Augment(vepcs, i + f, i + 2 * f - 1, Toq, Toqs);

  // Matching by SICP
  auto t1 = GetTime();
  auto tgt = MakeMatrix(datat);
  auto src = MakeMatrix(datas);
  SICPParameters sicpparams;
  sicpparams.huber = huber;
  auto sicpT = SICPMatch(tgt, src, sicpparams, Tio);
  auto t2 = GetTime();
  auto t12 = GetDiffTime(t1, t2);
  std::printf("sicp -> optimize: %d ms(%d%%), total: %d ms\n",
              sicpparams.usedtime.optimize, sicpparams.usedtime.optimize * 100 / t12, t12);

  // Matching by NDTD2D
  auto t3 = GetTime();
  auto mapt = MakeNDTMap(datat, {rvar, tvar}, {cell_size});
  auto maps = MakeNDTMap(datas, {rvar, tvar}, {cell_size});
  NDTD2DParameters ndtparams;
  ndtparams.huber = huber;
  auto ndtT = NDTD2DMatch(mapt, maps, ndtparams, Tio);
  auto t4 = GetTime();
  auto t34 = GetDiffTime(t3, t4);
  std::printf("ndt -> optimize: %d ms(%d%%), total: %d ms\n",
              ndtparams.usedtime.optimize, ndtparams.usedtime.optimize * 100 / t34, t34);

  // Matching by SNDT
  auto t5 = GetTime();
  auto smapt = MakeSNDTMap(datat, {rvar, tvar}, {cell_size, radius});
  auto smaps = MakeSNDTMap(datas, {rvar, tvar}, {cell_size, radius});
  SNDTParameters sndtparams;
  sndtparams.huber = huber;
  auto sndtT = SNDTMatch(smapt, smaps, sndtparams, Tio);
  auto t6 = GetTime();
  auto t56 = GetDiffTime(t5, t6);
  std::printf("sndt -> optimize: %d ms(%d%%), total: %d ms\n",
              sndtparams.usedtime.optimize, sndtparams.usedtime.optimize * 100 / t56, t56);

  // Compute Ground Truth
  Affine3d To, Ti;
  tf2::fromMsg(common::GetPose(gtpath.poses, vepcs[i + f].stamp), To);
  tf2::fromMsg(common::GetPose(gtpath.poses, vepcs[i].stamp), Ti);
  Affine3d gtTio3 = common::Conserve2DFromAffine3d(Ti.inverse() * To);
  Affine2d gtTio = Translation2d(gtTio3.translation()(0), gtTio3.translation()(1)) *
                   Rotation2Dd(gtTio3.rotation().block<2, 2>(0, 0));

  // Compute error
  auto err0 = common::TransNormRotDegAbsFromMatrix3d((sicpT * gtTio.inverse()).matrix());
  auto err1 = common::TransNormRotDegAbsFromMatrix3d((ndtT * gtTio.inverse()).matrix());
  auto err2 = common::TransNormRotDegAbsFromMatrix3d((sndtT * gtTio.inverse()).matrix());

  errs.data.push_back(err0(1));
  errs.data.push_back(err1(1));
  errs.data.push_back(err2(1));
  errs.layout.data_offset = ++idx;
  pube.publish(errs);

  auto rmax = FindMax(err0(0), err1(0), err2(0));
  auto rmin = FindMin(err0(0), err1(0), err2(0));
  auto tmax = FindMax(err0(1), err1(1), err2(1));
  auto tmin = FindMin(err0(1), err1(1), err2(1));

  std::printf("%d   sicp   ndt  sndt:\n"
              "rerr %.2f, %.2f, %.2f -> best: %d, diff, %.2f\n"
              "terr %.2f, %.2f, %.2f -> best: %d, diff, %.2f\n\n", i,
              err0(0), err1(0), err2(0), rmin.first, rmax.second - rmin.second,
              err0(1), err1(1), err2(1), tmin.first, tmax.second - tmin.second);
  std::fflush(stdout);
}

void GetFiles(string data) {
  string base = "/home/ee904/Desktop/HuaTsai/NormalNDT/Analysis/1Data/" + data;
  SerializationInput(base + "/vepcs.ser", vepcs);
  SerializationInput(base + "/gt.ser", gtpath);
  SerializationInput(base + "/back.ser", imb);
  SerializationInput(base + "/back_left.ser", imbl);
  SerializationInput(base + "/back_right.ser", imbr);
  SerializationInput(base + "/front.ser", imf);
  SerializationInput(base + "/front_left.ser", imfl);
  SerializationInput(base + "/front_right.ser", imfr);
}

int main(int argc, char **argv) {
  string data;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data Path")
      ("frames,f", po::value<int>(&frames)->default_value(5), "Frames")
      ("rvar", po::value<double>(&rvar)->default_value(0.0625), "Intrinsic radius variance")
      ("tvar", po::value<double>(&tvar)->default_value(0.0001), "Intrinsic theta variance")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("radius,r", po::value<double>(&radius)->default_value(1.5), "Radius")
      ("huber,u", po::value<double>(&huber)->default_value(0), "Huber");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  ros::init(argc, argv, "test_diff");
  ros::NodeHandle nh;

  GetFiles(data);

  ros::Subscriber sub = nh.subscribe("idx", 0, cb);

  pub1 = nh.advertise<MarkerArray>("markers1", 0, true);  // target
  pub2 = nh.advertise<MarkerArray>("markers2", 0, true);  // source
  pub3 = nh.advertise<MarkerArray>("markers3", 0, true);  // aligned source
  pub4 = nh.advertise<MarkerArray>("markers4", 0, true);
  pub5 = nh.advertise<MarkerArray>("markers5", 0, true);
  pub6 = nh.advertise<MarkerArray>("markers6", 0, true);
  pub7 = nh.advertise<MarkerArray>("markers7", 0, true);
  pub8 = nh.advertise<MarkerArray>("markers8", 0, true);
  pube = nh.advertise<std_msgs::Float64MultiArray>("error", 0, true);
  pb1 = nh.advertise<sensor_msgs::CompressedImage>("back/compressed", 0, true);
  pb2 = nh.advertise<sensor_msgs::CompressedImage>("back_left/compressed", 0, true);
  pb3 = nh.advertise<sensor_msgs::CompressedImage>("back_right/compressed", 0, true);
  pb4 = nh.advertise<sensor_msgs::CompressedImage>("front/compressed", 0, true);
  pb5 = nh.advertise<sensor_msgs::CompressedImage>("front_left/compressed", 0, true);
  pb6 = nh.advertise<sensor_msgs::CompressedImage>("front_right/compressed", 0, true);

  for (int i = 0; i < (int)vepcs.size() - 15; i += 5) {
    std_msgs::Int32 num;
    num.data = i;
    cb(num);
  }
  ros::spin();
}
