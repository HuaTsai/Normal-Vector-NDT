/**
 * @file test_match.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Test Matching
 * @version 0.1
 * @date 2021-07-13
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include <bits/stdc++.h>
#include <sndt_exec/wrapper.hpp>
#include <boost/program_options.hpp>
#include <common/EgoPointClouds.h>
#include <sndt/ndt_visualizations.h>
#include <std_msgs/Int32.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/Float64MultiArray.h>
#include <nav_msgs/Path.h>

using namespace std;
using namespace visualization_msgs;
namespace po = boost::program_options;

vector<common::EgoPointClouds> vepcs;  
ros::Publisher pub1, pub2, pub3, pub4, pub5, pub6, pub7, pubd, pube;
vector<Marker> allcircs;
vector<pair<MarkerArray, MarkerArray>> vmas;
int frames;
double cell_size, radius, rvar, tvar;
double huber;
nav_msgs::Path gtpath;

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
      ret.push_back(make_pair(fi, T0i * Affine2d(mtx)));
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
  int i = num.data;
  int f = frames;
  Affine2d Tio, Toq;
  vector<Affine2d> Tios, Toqs;
  auto datat = Augment(vepcs, i, i + f - 1, Tio, Tios);
  auto mapt = MakeMap(datat, {rvar, tvar}, {cell_size, radius});
  // int tvc = count_if(mapt.begin(), mapt.end(), [](NDTCell *cell) { return cell->BothHasGaussian(); });

  auto datas = Augment(vepcs, i + f, i + 2 * f - 1, Toq, Toqs);
  auto maps = MakeMap(datas, {rvar, tvar}, {cell_size, radius});
  // int svc = count_if(maps.begin(), maps.end(), [](NDTCell *cell) { return cell->BothHasGaussian(); });
  // cout << "valid target cells: " << tvc << endl;
  // cout << "valid source cells: " << svc << endl;

  /********* Compute Ground Truth *********/
  Affine3d To, Ti;
  tf2::fromMsg(common::GetPose(gtpath.poses, vepcs[i + f].stamp), To);
  tf2::fromMsg(common::GetPose(gtpath.poses, vepcs[i].stamp), Ti);
  Affine3d gtTio3 = common::Conserve2DFromAffine3d(Ti.inverse() * To);
  Affine2d gtTio = Translation2d(gtTio3.translation()(0), gtTio3.translation()(1)) *
                   Rotation2Dd(gtTio3.rotation().block<2, 2>(0, 0));
  /********* Compute End Here     *********/

  NDTMatcher matcher;
  matcher.SetStrategy(NDTMatcher::kUSE_CELLS_GREATER_THAN_TWO_POINTS);
  matcher.huber = huber;
  cout << "start frame: " << i;
  // matcher.verbose = true;
  auto res = matcher.CeresMatch(mapt, maps, Tio);
  vmas = matcher.vmas;

  // cout << "guess: " << common::XYTDegreeFromMatrix3d(Tio.matrix()).transpose() << endl;
  // cout << "result: " << common::XYTDegreeFromMatrix3d(res.matrix()).transpose() << endl;
  // cout << "grount truth: " << common::XYTDegreeFromMatrix3d(gtTio.matrix()).transpose() << endl;
  auto err = common::TransNormRotDegAbsFromMatrix3d(res.inverse() * gtTio.matrix());
  cout << "err: " << err.transpose() << endl;
  auto maps2 = maps.PseudoTransformCells(res, true);
  auto mapsg = maps.PseudoTransformCells(gtTio, true);

  pub1.publish(MarkerArrayOfNDTMap(mapt, true));
  pub2.publish(MarkerArrayOfNDTMap(maps));   // source map
  pub3.publish(MarkerArrayOfNDTMap(maps2));  // source map after T0
  pub4.publish(MarkerArrayOfNDTMap(mapsg));  // source map after Tgt
  // pub5 ?
  pub6.publish(vmas[0].first);
  pub7.publish(vmas[0].second);
  geometry_msgs::Vector3 errmsg;
  errmsg.x = err(0), errmsg.y = err(1);
  pube.publish(errmsg);
}

void cb2(const std_msgs::Int32 &num) {
  int n = num.data;
  if (n >= 0 && n < (int)vmas.size()) {
    pub6.publish(vmas[n].first);
    pub7.publish(vmas[n].second);
  }
}

int main(int argc, char **argv) {
  string datafile, gtfile;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("gtfile,g", po::value<string>(&gtfile)->required(), "Groudtruth Path")
      ("datafile,d", po::value<string>(&datafile)->required(), "Data Path")
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

  ros::init(argc, argv, "test_match");
  ros::NodeHandle nh;

  common::SerializationInput(datafile, vepcs);
  common::SerializationInput(gtfile, gtpath);

  ros::Subscriber sub = nh.subscribe("idx", 0, cb);
  ros::Subscriber sub2 = nh.subscribe("iter", 0, cb2);

  pub1 = nh.advertise<MarkerArray>("markers1", 0, true);  // target
  pub2 = nh.advertise<MarkerArray>("markers2", 0, true);  // source
  pub3 = nh.advertise<MarkerArray>("markers3", 0, true);  // aligned source
  pub4 = nh.advertise<MarkerArray>("markers4", 0, true);
  pub5 = nh.advertise<MarkerArray>("markers5", 0, true);
  pub6 = nh.advertise<MarkerArray>("markers6", 0, true);
  pub7 = nh.advertise<MarkerArray>("markers7", 0, true);
  pubd = nh.advertise<Marker>("markerd", 0, true);  // iterations
  pube = nh.advertise<geometry_msgs::Vector3>("err", 0, true);

  // for (int i = 0; i < (int)vepcs.size() - 15; ++i) {
  //   std_msgs::Int32 i32;
  //   i32.data = i;
  //   cb(i32);
  // }

  ros::spin();
}
