#include <bits/stdc++.h>
#include <common/angle_utils.h>
#include <common/eigen_utils.h>
#include <nav_msgs/OccupancyGrid.h>
#include <ros/ros.h>
#include <tf2_eigen/tf2_eigen.h>
#include <visualization_msgs/MarkerArray.h>

using namespace std;
using namespace Eigen;
using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

void ComputeEvalEvec(const Eigen::Matrix2d &covariance,
                     Eigen::Vector2d &evals,
                     Eigen::Matrix2d &evecs) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> evd;
  evd.computeDirect(covariance);
  evals = evd.eigenvalues();
  evecs = evd.eigenvectors();
}

Marker MarkerOfEllipse(const Eigen::Vector2d &mean,
                       const Eigen::Matrix2d &covariance) {
  Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = ros::Time::now();
  ret.id = 0;
  ret.type = Marker::SPHERE;
  ret.action = Marker::ADD;
  ret.color.a = 0.7;
  ret.color.g = 1.0;
  Eigen::Vector2d evals;
  Eigen::Matrix2d evecs;
  ComputeEvalEvec(covariance, evals, evecs);
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  R.block<2, 2>(0, 0) << evecs(0, 0), -evecs(1, 0), evecs(1, 0), evecs(0, 0);
  Eigen::Quaterniond q(R);
  ret.scale.x = 2 * sqrt(evals(0));  // +- 1σ
  ret.scale.y = 2 * sqrt(evals(1));  // +- 1σ
  ret.scale.z = 0.1;
  ret.pose.position.x = mean(0);
  ret.pose.position.y = mean(1);
  ret.pose.position.z = 0;
  ret.pose.orientation = tf2::toMsg(q);
  return ret;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "occtest");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<nav_msgs::OccupancyGrid>("occ", 0, true);
  ros::Publisher pub2 = nh.advertise<Marker>("marker1", 0, true);

  Vector2d up(2, 1.5), uq(1, 1);
  Matrix2d cp, cq;
  cp << 0.08956964, -0.0624375, -0.0624375, 0.16318393;
  cq << 0.13287857, -0.12345357, -0.12345357, 0.19375536;
  Vector2d cen(1.5, 1.5);
  double cs = 2;
  double res = 0.1;
  double skew = Deg2Rad(40);
  int n = (int)cs / res;
  double ares = cs / n;
  Vector2d bl = cen + Eigen::Rotation2Dd(skew) * Vector2d(-cs / 2, -cs / 2);
  // Vector2d tl = cen + Eigen::Rotation2Dd(skew) * Vector2d(-cs / 2, cs / 2);

  nav_msgs::MapMetaData mmd;
  mmd.map_load_time = ros::Time(1, 0);
  mmd.resolution = ares;
  mmd.width = mmd.height = n;
  Affine3d aff =
      Translation3d(bl(0), bl(1), 0) * AngleAxisd(skew, Vector3d::UnitZ());
  mmd.origin = tf2::toMsg(aff);
  nav_msgs::OccupancyGrid occ;
  occ.header.stamp = ros::Time::now();
  occ.header.frame_id = "map";
  occ.info = mmd;
  std::vector<double> vals;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      Vector2d offset((j + 0.5) * ares, (i + 0.5) * ares);
      Vector2d x = bl + Eigen::Rotation2Dd(skew) * offset;
      vals.push_back(Mvn(uq, cq).pdf(x));
    }
  }
  // double scale = 2000. / accumulate(vals.begin(), vals.end(), 0.);
  double scale = 80;
  std::transform(vals.begin(), vals.end(), back_inserter(occ.data),
                 [&scale](double v) { return clamp(int(scale * v), 10, 100); });
  pub.publish(occ);
  pub2.publish(MarkerOfEllipse(uq, cq));

  ros::spin();
}
