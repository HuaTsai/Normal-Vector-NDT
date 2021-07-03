#include <bits/stdc++.h>
#include "sndt/wrapper.hpp"
#include "dbg/dbg.h"
#include "common/common.h"
#include "common/EgoPointClouds.h"
#include "sndt/ndt_conversions.hpp"
#include <boost/program_options.hpp>

using namespace std;
namespace po = boost::program_options;

MatrixXd RemoveInfiniteColumn(const MatrixXd &mtx) {
  MatrixXd ret(mtx.rows(), mtx.cols());
  int n = 0;
  for (int i = 0; i < mtx.cols(); ++i) {
    if (mtx.col(i).allFinite())
      ret.col(n++) = mtx.col(i);
  }
  ret.resize(NoChange, n);
  return ret;
}

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
  double cellsize, radius;
  Vector2d intrinsic;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("datapath,p", po::value<string>(&path)->required(), "Data Path")
      ("cellsize,s", po::value<double>(&cellsize)->default_value(1), "Cell Size")
      ("radius,r", po::value<double>(&radius)->default_value(2.5), "Normal Radius")
      ("r2cov,c", po::value<double>(&intrinsic(0))->default_value(0.0625), "Intrinsic sigma_r^2")
      ("thcov,t", po::value<double>(&intrinsic(1))->default_value(0.0001), "Intrinsic sigma_theta^2");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  vector<common::EgoPointClouds> vepcs;  
  common::SerializationInput(path, vepcs);
  cout << vepcs.size() << endl;
  Affine2d T;
  auto data = Augment(vepcs, 6, 10, T);
  auto map = MakeMap(data, intrinsic, {cellsize, radius});
  cout << map.ToString() << endl;
  ros::init(argc, argv, "test_covmap");
  ros::NodeHandle nh;

  ros::Publisher pub1_pt = nh.advertise<visualization_msgs::MarkerArray>("marker1", 0, true);
  ros::Publisher pub2_ptic = nh.advertise<visualization_msgs::MarkerArray>("marker2", 0, true);
  ros::Publisher pub3_ptc = nh.advertise<visualization_msgs::MarkerArray>("marker3", 0, true);
  ros::Publisher pub4_nm = nh.advertise<visualization_msgs::MarkerArray>("marker4", 0, true);
  ros::Publisher pub5_nmc = nh.advertise<visualization_msgs::MarkerArray>("marker5", 0, true);
  ros::Publisher pub6_tg = nh.advertise<visualization_msgs::MarkerArray>("marker6", 0, true);
  ros::Publisher pub7_bd = nh.advertise<visualization_msgs::MarkerArray>("marker7", 0, true);
  ros::Publisher pub8_ss = nh.advertise<visualization_msgs::MarkerArray>("marker8", 0, true);

  visualization_msgs::MarkerArray ma1_pt, ma2_ptic, ma3_ptc, ma4_nm, ma5_nmc, ma6_tg, ma7_bd;
  for (auto cell : map) {
    UpdateMarkerArray(ma1_pt, MarkerOfPoints(cell->GetPointsMatrix()));
    // TODO: ptic
    // if (n++ != 292) { continue; }
    if (cell->GetPHasGaussian()) {
      UpdateMarkerArray(ma3_ptc, MarkerOfEclipse(cell->GetPointMean(), cell->GetPointCov()));
    }
    auto ma4_nm_ = MarkerArrayOfArrow(cell->GetPointsMatrix(), cell->GetPointsMatrix() + cell->GetNormalsMatrix());
    ma4_nm = JoinMarkerArraysAndMarkers({ma4_nm, ma4_nm_});
    if (cell->GetNHasGaussian()) {
      auto neclipse = MarkerOfEclipse(cell->GetPointMean() + cell->GetNormalMean(), cell->GetNormalCov(), common::Color::kGray);
      UpdateMarkerArray(ma5_nmc, neclipse);
      auto points = FindTangentPoints(neclipse, cell->GetPointMean());
      UpdateMarkerArray(ma6_tg, MarkerOfLines({points[0], cell->GetPointMean(),
                                               cell->GetPointMean(), points[1]}));
    }
    UpdateMarkerArray(ma7_bd, MarkerOfBoundary(cell->GetCenter(), cell->GetSize()(0), cell->GetSkewRad()));
  }

  visualization_msgs::MarkerArray ma8_ss;
  for (size_t i = 0; i < data.size(); ++i) {
    visualization_msgs::Marker m;
    m.header.frame_id = "map";
    m.header.stamp = ros::Time::now();
    m.id = i;
    m.type = visualization_msgs::Marker::CUBE;
    m.action = visualization_msgs::Marker::ADD;
    m.pose = tf2::toMsg(common::Affine3dFromAffine2d(data[i].second));
    m.color = common::MakeColorRGBA(common::Color::kGray);
    m.scale.x = 0.3;
    m.scale.y = 1.0;
    m.scale.z = 0.5;
    ma8_ss.markers.push_back(m);
  }

  pub1_pt.publish(ma1_pt);
  // pub2_ptic.publish();
  pub3_ptc.publish(ma3_ptc);
  pub4_nm.publish(ma4_nm);
  pub5_nmc.publish(ma5_nmc);
  pub6_tg.publish(ma6_tg);
  pub7_bd.publish(ma7_bd);
  pub8_ss.publish(ma8_ss);
  ros::spin();
}
