#include <bits/stdc++.h>
#include "sndt/wrapper.hpp"
#include "dbg/dbg.h"
#include "common/common.h"
#include "common/EgoPointClouds.h"
#include "sndt/ndt_conversions.hpp"

using namespace std;

pair<MatrixXd, Affine2d> ToPair(const common::PointCloudSensor &pcs) {
  MatrixXd fi(2, pcs.points.size());
  for (int i = 0; i < fi.cols(); ++i)
    fi.col(i) = Vector2d(pcs.points[i].x, pcs.points[i].y);
  Affine3d aff;
  tf2::fromMsg(pcs.origin, aff);
  Matrix3d mtx = Matrix3d::Identity();
  mtx.block<2, 2>(0, 0) = aff.matrix().block<2, 2>(0, 0);
  mtx.block<2, 1>(0, 2) = aff.matrix().block<2, 1>(0, 3);
  return make_pair(fi, Affine2d(mtx));
}

int main(int argc, char **argv) {
  vector<common::EgoPointClouds> vepcs;  
  common::SerializationInput(argv[1], vepcs);
  cout << vepcs.size() << endl;
  vector<pair<MatrixXd, Affine2d>> data;
  for (int i = 6; i <= 10; ++i)
    for (int j = 0; j <= 4; ++j)
      data.push_back(ToPair(vepcs[i].pcs[j]));
  auto map = MakeMap(data, Vector2d(0.0625, 0.0001), {1, 2.5});
  cout << map.ToString() << endl;
  ros::init(argc, argv, "test_covmap");
  ros::NodeHandle nh;

  // 1. by NDTMapMsg
  // ros::Publisher pub = nh.advertise<sndt::NDTMapMsg>("map", 0, true);
  // auto msg = ToMessage(map, "map");
  // pub.publish(msg);
  // ros::spin();

  // 2. by markerarray
  ros::Publisher pub = nh.advertise<visualization_msgs::MarkerArray>("marker", 0, true);
  visualization_msgs::MarkerArray ma;
  int id = -1;
  for (auto cell : map) {
    auto ima = MarkerArrayFromNDTCell(cell, id);
    ma.markers.insert(ma.markers.end(), ima.markers.begin(), ima.markers.end());
    id = ima.markers.back().id;
  }
  pub.publish(ma);
  ros::spin();
}
