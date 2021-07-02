/**
 * @file test_0dis.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Evaluate Distribution of SICP and SNDT
 * @version 0.1
 * @date 2021-07-01
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include <bits/stdc++.h>
#include <sndt/wrapper.hpp>
#include <boost/program_options.hpp>
#include <common/EgoPointClouds.h>
#include <sndt/ndt_conversions.hpp>

using namespace std;
namespace po = boost::program_options;

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

// start, ..., end, end+1
// <<------- T -------->>
vector<pair<MatrixXd, Affine2d>> Augment(
    const vector<common::EgoPointClouds> &vepcs, int start, int end,
    Affine2d &T) {
  vector<pair<MatrixXd, Affine2d>> ret;
  T = Affine2d::Identity();
  double dx = 0, dy = 0, dth = 0;
  for (int i = start; i <= end; ++i) {
    double dt = (vepcs[i + 1].stamp - vepcs[i].stamp).toSec();
    dx += vepcs[i].vxyt[0] * dt;
    dy += vepcs[i].vxyt[1] * dt;
    dth += vepcs[i].vxyt[2] * dt;
    Affine2d T0i = Rotation2Dd(dth) * Translation2d(dx, dy);
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
  }
  return ret;
}

int main(int argc, char **argv) {
  string path;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("datapath,p", po::value<string>(&path)->required(), "Data Path");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  ros::init(argc, argv, "test_0dis");
  ros::NodeHandle nh;

  vector<common::EgoPointClouds> vepcs;  
  common::SerializationInput(path, vepcs);
  cout << vepcs.size() << endl;

  Affine2d T611, T1116;
  auto datat = Augment(vepcs, 6, 10, T611);
  auto mapt = MakeMap(datat, {0.0625, 0.0001}, {1, 2.5});

  auto datas = Augment(vepcs, 11, 15, T1116);
  auto maps = MakeMap(datas, {0.0625, 0.0001}, {1, 2.5});

  auto kd = MakeKDTree(mapt);
  int i = 0;
  vector<visualization_msgs::MarkerArray> vps, vqs;
  vector<Vector2d> lines;
  auto maps2 = maps.PseudoTransformCells(T611);
  for (auto cellp : maps2) {
    if (!cellp->BothHasGaussian()) { continue; }
    vector<int> idx(1);
    vector<float> dist2(1);
    pcl::PointXYZ pt(cellp->GetPointMean()(0), cellp->GetPointMean()(1), 0);
    int found = kd.nearestKSearch(pt, 1, idx, dist2);
    if (!found) { continue; }
    Vector2d npt((*kd.getInputCloud())[idx[0]].x, (*kd.getInputCloud())[idx[0]].y);
    auto cellq = mapt.GetCellForPoint(npt);
    if (!cellq || !cellq->BothHasGaussian()) { continue; }
    // cout << "Match " << ++i << endl;
    // cout << "cell p: " << cellp->ToString() << endl;
    // cout << "cell q: " << cellq->ToString() << endl;
    vps.push_back(MarkerArrayOfNDTCell(cellp.get()));
    vqs.push_back(MarkerArrayOfNDTCell(cellq));
    lines.push_back(cellp->GetPointMean());
    lines.push_back(cellq->GetPointMean());
  }

  ros::Publisher pub = nh.advertise<visualization_msgs::MarkerArray>("marker", 0, true);
  ros::Publisher pub2 = nh.advertise<visualization_msgs::MarkerArray>("marker2", 0, true);
  ros::Publisher pub3 = nh.advertise<visualization_msgs::MarkerArray>("marker3", 0, true);
  ros::Publisher pub4 = nh.advertise<visualization_msgs::MarkerArray>("marker4", 0, true);
  ros::Publisher pub5 = nh.advertise<visualization_msgs::Marker>("marker5", 0, true);
  pub.publish(MarkerArrayOfNDTMap(maps2));
  pub2.publish(MarkerArrayOfNDTMap(mapt));
  pub3.publish(JoinMarkerArraysAndMarkers(vps));
  pub4.publish(JoinMarkerArraysAndMarkers(vqs));
  pub5.publish(MarkerOfLines(lines, common::Color::kRed, 1.0));
  ros::spin();
}
