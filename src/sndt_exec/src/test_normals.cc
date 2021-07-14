/**
 * @file test_normals.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief check individual points radius normal
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

ros::Publisher pb1;
vector<Marker> allcircs;

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
  int n = num.data;
  pb1.publish(allcircs[n]);
}

int main(int argc, char **argv) {
  string path;
  int start_frame, frames;
  double cell_size, radius;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("datapath,p", po::value<string>(&path)->required(), "Data Path")
      ("startframe,s", po::value<int>(&start_frame)->default_value(6), "Start Frame")
      ("frames,f", po::value<int>(&frames)->default_value(5), "Frames")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1), "Cell Size")
      ("radius,r", po::value<double>(&radius)->default_value(2.5), "Radius");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  ros::init(argc, argv, "test_normals");
  ros::NodeHandle nh;

  vector<common::EgoPointClouds> vepcs;  
  common::SerializationInput(path, vepcs);

  vector<Affine2d> T611s;
  Affine2d T611;
  auto data = Augment(vepcs, start_frame, start_frame + frames - 1, T611, T611s);
  auto map = MakeMap(data, {0.0625, 0.0001}, {cell_size, radius});

  vector<Vector2d> allpts, nmstarts, nmends;
  vector<Marker> mall;
  int j = 0;
  for (const auto &cell : map) {
    int n = cell->GetN();
    cout << "[" << j << " ~ " << j + n - 1 << "]"<< endl;
    j += n;
    for (int i = 0; i < n; ++i) {
      allpts.push_back(cell->GetPoints()[i]);
      nmstarts.push_back(cell->GetPoints()[i]);
      nmends.push_back(cell->GetPoints()[i] + cell->GetNormals()[i]);
      allcircs.push_back(MarkerOfCircle(cell->GetPoints()[i], 2.5, Color::kGray));
    }
    mall.push_back(MarkerOfBoundary(cell->GetCenter(), cell->GetSize()));
    cout << cell->ToString();
  }
  mall.push_back(MarkerOfPoints(allpts));
  auto mnm = MarkerArrayOfArrow(nmstarts, nmends);
  cout << allcircs.size() << endl;

  ros::Subscriber sub = nh.subscribe("idx", 0, cb);

  ros::Publisher pub1 = nh.advertise<MarkerArray>("markers1", 0, true);  // all arrows and all points
  pub1.publish(JoinMarkerArraysAndMarkers({mnm}, mall));

  pb1 = nh.advertise<Marker>("marker1", 0, true);  // single point, arrow and circle

  ros::spin();
}
