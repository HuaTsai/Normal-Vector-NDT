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

using namespace std;
using namespace visualization_msgs;
namespace po = boost::program_options;

vector<common::EgoPointClouds> vepcs;  
ros::Publisher pub1, pub2, pub3, pub4, pub5, pub6;
vector<Marker> allcircs;
vector<MarkerArray> vmas;
int frames;
double cell_size, radius;

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

void cb(const std_msgs::Int32 &num) {
  int start_frame = num.data;
  vector<Affine2d> T611s, T1116s;
  Affine2d T611, T1116;
  auto datat = Augment(vepcs, start_frame, start_frame + frames - 1, T611, T611s);
  auto mapt = MakeMap(datat, {0.0625, 0.0001}, {cell_size, radius});
  int tvc = count_if(mapt.begin(), mapt.end(), [](NDTCell *cell) { return cell->BothHasGaussian(); });
  // cout << mapt.ToString() << endl;

  auto datas = Augment(vepcs, start_frame + frames, start_frame + 2 * frames - 1, T1116, T1116s);
  auto maps = MakeMap(datas, {0.0625, 0.0001}, {cell_size, radius});
  int svc = count_if(maps.begin(), maps.end(), [](NDTCell *cell) { return cell->BothHasGaussian(); });
  cout << "valid target cells: " << tvc << endl;
  cout << "valid source cells: " << svc << endl;
  // cerr << maps.ToString() << endl;

  NDTMatcher matcher;
  matcher.SetStrategy(NDTMatcher::kUSE_CELLS_GREATER_THAN_TWO_POINTS);
  cout << "start frame: " << start_frame << endl;
  auto res = matcher.CeresMatch(mapt, maps, T611);
  vmas = matcher.vmas;
  cout << "guess: " << common::XYTDegreeFromMatrix3d(T611.matrix()).transpose() << endl;
  cout << "result: " << common::XYTDegreeFromMatrix3d(res.matrix()).transpose() << endl;
  auto maps2 = maps.PseudoTransformCells(res, true);
  cout << "vis mapt" << endl;

  pub1.publish(MarkerArrayOfNDTMap(mapt, true));
  pub2.publish(MarkerArrayOfNDTMap(maps));
  pub3.publish(MarkerArrayOfNDTMap(maps2));
  vector<Affine2d> affs, afft;
  transform(datas.begin(), datas.end(), back_inserter(affs), [&T611](auto a) { return T611 * a.second; });
  transform(datat.begin(), datat.end(), back_inserter(afft), [](auto a) { return a.second; });
  vector<Vector2d> cars, cart;
  transform(T1116s.begin(), T1116s.end(), back_inserter(cars), [&T611](auto a) { return T611 * a.translation(); });
  transform(T611s.begin(), T611s.end(), back_inserter(cart), [](auto a) { return a.translation(); });
  pub4.publish(JoinMarkerArraysAndMarkers(
      {MarkerArrayOfSensor(afft)},
      {MarkerOfLinesByEndPoints(cart, Color::kLime, 1.0),
       MarkerOfPoints(cart, 0.1, Color::kBlack)}));
  pub5.publish(JoinMarkerArraysAndMarkers(
      {MarkerArrayOfSensor(affs)},
      {MarkerOfLinesByEndPoints(cars, Color::kRed, 1.0),
       MarkerOfPoints(cars, 0.1, Color::kBlack)}));
}

void cb2(const std_msgs::Int32 &num) {
  int n = num.data;
  if (n >= 0 && n < (int)vmas.size())
    pub6.publish(vmas[n]);
}

int main(int argc, char **argv) {
  string path;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("datapath,p", po::value<string>(&path)->required(), "Data Path")
      ("frames,f", po::value<int>(&frames)->default_value(5), "Frames")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("radius,r", po::value<double>(&radius)->default_value(1.5), "Radius");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  ros::init(argc, argv, "test_match");
  ros::NodeHandle nh;

  common::SerializationInput(path, vepcs);

  ros::Subscriber sub = nh.subscribe("idx", 0, cb);
  ros::Subscriber sub2 = nh.subscribe("iter", 0, cb2);

  pub1 = nh.advertise<MarkerArray>("markers1", 0, true);  // target
  pub2 = nh.advertise<MarkerArray>("markers2", 0, true);  // source
  pub3 = nh.advertise<MarkerArray>("markers3", 0, true);  // aligned source
  pub4 = nh.advertise<MarkerArray>("markers4", 0, true);  // target sensor
  pub5 = nh.advertise<MarkerArray>("markers5", 0, true);  // source sensor
  pub6 = nh.advertise<MarkerArray>("markers6", 0, true);  // iterations

  ros::spin();
}
