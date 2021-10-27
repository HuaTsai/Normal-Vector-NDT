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
vector<int> corr0, corr1, corr2, opt0, opt1, opt2;

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

  // Matching by SICP
  auto t1 = GetTime();
  SICPParameters sicpparams;
  auto tgt = AugmentPoints(vepcs, i, i + f - 1, Tio, Tios);
  auto src = AugmentPoints(vepcs, i + f, i + 2 * f - 1, Toq, Toqs);
  auto sicpT = SICPMatch(tgt, src, sicpparams, Tio);
  // corr0.push_back(sicpparams._corres[0]);
  auto t2 = GetTime();
  auto t12 = GetDiffTime(t1, t2);
  opt0.push_back(t12);
  std::printf("sicp -> optimize: %d ms(%d%%), total: %d ms\n",
              sicpparams._usedtime.optimize, sicpparams._usedtime.optimize * 100 / t12, t12);

  // Matching by NDTD2D
  auto t3 = GetTime();
  D2DNDTParameters ndtparams;
  auto datat0 = Augment(vepcs, i, i + f - 1, Tio, Tios);
  auto datas0 = Augment(vepcs, i + f, i + 2 * f - 1, Toq, Toqs);
  auto mapt = MakeNDTMap(datat0, ndtparams);
  auto maps = MakeNDTMap(datas0, ndtparams);
  auto ndtT = D2DNDTMatch(mapt, maps, ndtparams, Tio);
  // corr1.push_back(ndtparams._corres[0]);
  auto t4 = GetTime();
  auto t34 = GetDiffTime(t3, t4);
  opt1.push_back(t34);
  std::printf("ndt -> optimize: %d ms(%d%%), total: %d ms\n",
              ndtparams._usedtime.optimize, ndtparams._usedtime.optimize * 100 / t34, t34);

  // Matching by SNDT
  auto t5 = GetTime();
  SNDTParameters sndtparams;
  auto datat1 = Augment(vepcs, i, i + f - 1, Tio, Tios);
  auto datas1 = Augment(vepcs, i + f, i + 2 * f - 1, Toq, Toqs);
  auto smapt = MakeSNDTMap(datat1, sndtparams);
  auto smaps = MakeSNDTMap(datas1, sndtparams);
  auto sndtT = SNDTMatch(smapt, smaps, sndtparams, Tio);
  // corr2.push_back(sndtparams._corres[0]);
  auto t6 = GetTime();
  auto t56 = GetDiffTime(t5, t6);
  opt2.push_back(t56);
  std::printf("sndt -> optimize: %d ms(%d%%), total: %d ms\n",
              sndtparams._usedtime.optimize, sndtparams._usedtime.optimize * 100 / t56, t56);

  // Compute Ground Truth
  Affine3d To, Ti;
  tf2::fromMsg(GetPose(gtpath.poses, vepcs[i + f].stamp), To);
  tf2::fromMsg(GetPose(gtpath.poses, vepcs[i].stamp), Ti);
  Affine3d gtTio3 = Conserve2DFromAffine3d(Ti.inverse() * To);
  Affine2d gtTio = Translation2d(gtTio3.translation()(0), gtTio3.translation()(1)) *
                   Rotation2Dd(gtTio3.rotation().block<2, 2>(0, 0));

  // Compute error
  auto err0 = TransNormRotDegAbsFromAffine2d(sicpT * gtTio.inverse());
  auto err1 = TransNormRotDegAbsFromAffine2d(ndtT * gtTio.inverse());
  auto err2 = TransNormRotDegAbsFromAffine2d(sndtT * gtTio.inverse());

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

  /******* Publishers *******/
  pub1.publish(JoinMarkers({MarkerOfPoints(tgt, 0.5, Color::kRed)}));

  vector<Eigen::Vector2d> gue;
  transform(src.begin(), src.end(), back_inserter(gue),
                 [&Tio](auto p) { return Tio * p; });
  pub2.publish(JoinMarkers({MarkerOfPoints(gue)}));

  vector<Eigen::Vector2d> res;
  transform(src.begin(), src.end(), back_inserter(res),
                 [&sicpT](auto p) { return sicpT * p; });
  pub3.publish(JoinMarkers({MarkerOfPoints(res)}));

  // NDTD2D
  vector<Marker> m4;
  vector<Vector2d> m4pts;
  for (const auto &cell : mapt) {
    if (cell->HasGaussian())
      m4.push_back(MarkerOfBoundary(cell->GetCenter(), cell->GetSize(),
                                    cell->GetSkewRad(), Color::kRed, 0.5));
    for (auto pt : cell->GetPoints())
      m4pts.push_back(pt);
  }
  m4.push_back(MarkerOfPoints(m4pts, 0.5, Color::kRed));
  pub4.publish(JoinMarkers(m4));

  auto nexts = maps.PseudoTransformCells(ndtT, true);
  vector<Marker> m5;
  vector<Vector2d> m5pts;
  for (const auto &cell : nexts) {
    if (cell->HasGaussian())
      m5.push_back(MarkerOfBoundary(cell->GetCenter(), cell->GetSize(),
                                    cell->GetSkewRad(), Color::kFuchsia, 0.5));
    for (auto pt : cell->GetPoints())
      m5pts.push_back(pt);
  }
  m5.push_back(MarkerOfPoints(m5pts, 0.5, Color::kFuchsia));
  pub5.publish(JoinMarkers(m5));

  // SNDT
  vector<Marker> m6;
  vector<Vector2d> m6pts;
  for (const auto &cell : smapt) {
    if (cell->HasGaussian())
      m6.push_back(MarkerOfBoundary(cell->GetCenter(), cell->GetSize(),
                                    cell->GetSkewRad(), Color::kRed, 0.5));
    for (auto pt : cell->GetPoints())
      m6pts.push_back(pt);
  }
  m6.push_back(MarkerOfPoints(m6pts, 0.5, Color::kRed));
  pub6.publish(JoinMarkers(m6));

  vector<Marker> m7;
  vector<Vector2d> m7pts;
  auto snexts = smaps.PseudoTransformCells(sndtT, true);
  for (const auto &cell : snexts) {
    if (cell->HasGaussian()) {
      m7.push_back(MarkerOfBoundary(cell->GetCenter(), cell->GetSize(),
                                     cell->GetSkewRad(), Color::kBlue));
    }
    for (auto pt : cell->GetPoints())
      m7pts.push_back(pt);
  }
  m7.push_back(MarkerOfPoints(m7pts, 0.5, Color::kBlue));
  pub7.publish(JoinMarkers(m7));
  pub8.publish(MarkerArrayOfSNDTMap(snexts));

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
  bool run;
  string data;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (log24, log35-1, log62-1, log62-2)")
      ("frames,f", po::value<int>(&frames)->default_value(5), "Frames")
      ("rvar", po::value<double>(&rvar)->default_value(0.0625), "Intrinsic radius variance")
      ("tvar", po::value<double>(&tvar)->default_value(0.0001), "Intrinsic theta variance")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("radius,r", po::value<double>(&radius)->default_value(1.5), "Radius")
      ("huber,u", po::value<double>(&huber)->default_value(1), "Huber")
      ("run", po::value<bool>(&run)->default_value(false)->implicit_value(true), "Run");
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
  pub3 = nh.advertise<MarkerArray>("markers3", 0, true);  // sicp
  pub4 = nh.advertise<MarkerArray>("markers4", 0, true);  // ndtd2d
  pub5 = nh.advertise<MarkerArray>("markers5", 0, true);  // sndt
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

  if (run) {
    int n = vepcs.size() / 5 * 5;
    for (int i = 0; i < n - 5; i += 5) {
      cout << "(" << i << "/" << n - 10 << ")" << endl;
      std_msgs::Int32 num;
      num.data = i;
      cb(num);
    }
  }
  // cout << "SICP corres:\n";
  // copy(corr0.begin(), corr0.end(), ostream_iterator<int>(cout, ", "));
  // cout << "\nNDTD2D corres:\n";
  // copy(corr1.begin(), corr1.end(), ostream_iterator<int>(cout, ", "));
  // cout << "\nSNDT corres:\n";
  // copy(corr2.begin(), corr2.end(), ostream_iterator<int>(cout, ", "));

  cout << "\n\nSICP time:\n";
  copy(opt0.begin(), opt0.end(), ostream_iterator<int>(cout, ", "));
  cout << "\nNDTD2D time:\n";
  copy(opt1.begin(), opt1.end(), ostream_iterator<int>(cout, ", "));
  cout << "\nSNDT time:\n";
  copy(opt2.begin(), opt2.end(), ostream_iterator<int>(cout, ", "));

  cout << "\n\nReady" << endl;
  ros::spin();
}
