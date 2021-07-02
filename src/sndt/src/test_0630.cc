/**
 * @file test_0630.cc
 * @author HuaTsai
 * @brief Fake Data for pipeline usage
 * @version 0.1
 * @date 2021-06-30
 *
 * @copyright Copyright (c) 2021
 *
 */

#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <sndt/ndt_cell_2d.hpp>
#include <sndt/ndt_conversions.hpp>

using namespace std;
using namespace Eigen;

visualization_msgs::MarkerArray MakePointsNormalMarker(const MatrixXd &pts, const MatrixXd &nms) {
  Expects(pts.cols() == nms.cols());
  int n = pts.cols();
  visualization_msgs::MarkerArray ret;
  auto now = ros::Time::now();
  visualization_msgs::Marker points;
  points.header.frame_id = "map";
  points.header.stamp = now;
  points.id = 0;
  points.type = visualization_msgs::Marker::SPHERE_LIST;
  points.action = visualization_msgs::Marker::ADD;
  points.scale.x = 0.1;
  points.scale.y = 0.1;
  points.scale.z = 0.1;
  points.pose = tf2::toMsg(Affine3d::Identity());
  points.color = common::MakeColorRGBA(common::Color::kLime);
  for (int i = 0; i < n; ++i) {
    geometry_msgs::Point pt;
    pt.x = pts(0, i), pt.y = pts(1, i), pt.z = 0;
    points.points.push_back(pt);
  }
  ret.markers.push_back(points);

  visualization_msgs::Marker normal;
  normal.header.frame_id = "map";
  normal.id = 0;
  normal.header.stamp = now;
  normal.type = visualization_msgs::Marker::ARROW;
  normal.action = visualization_msgs::Marker::ADD;
  normal.scale.x = 0.05;
  normal.scale.y = 0.2;
  normal.pose = tf2::toMsg(Affine3d::Identity());
  normal.color = common::MakeColorRGBA(common::Color::kRed);
  normal.points.resize(2);
  for (int i = 0; i < n; ++i) {
    ++normal.id;
    geometry_msgs::Point pt;
    pt.x = pts(0, i), pt.y = pts(1, i), pt.z = 0;
    normal.points[0] = pt;
    pt.x = pt.x + nms(0, i), pt.y = pt.y + nms(1, i);
    normal.points[1] = pt;
    ret.markers.push_back(normal);
  }
  return ret;
}

NDTCell MakeCell(const Vector2d &center,
                 const MatrixXd &pts,
                 const MatrixXd &normals,
                 const Affine2d &T,
                 const Vector2d &intrinsic,
                 double cell_size) {
  NDTCell ret;
  ret.SetCenter(center);
  ret.SetSize(Vector2d(cell_size, cell_size));
  for (int i = 0; i < pts.cols(); ++i) {
    Vector2d pt = pts.col(i);
    double r2 = pt.squaredNorm();
    double theta = atan2(pt(1), pt(0));
    Matrix2d J = Rotation2Dd(theta).matrix();
    Matrix2d S = Vector2d(intrinsic(0), r2 * intrinsic(1)).asDiagonal();
    ret.AddPointWithCovariance(T * pt, J * S * J.transpose());
    ret.AddNormal(normals.col(i));
  }
  ret.ComputeGaussianWithCovariances();
  return ret;
}

int main(int argc, char **argv) {
  // namespace po = boost::program_options;
  // po::options_description desc("Allowed options");
  // desc.add_options()
  //     ("help,h", "Produce help message")
  //     ("var1", po::value<var1_t>(&var1)->required(), "Var1 Description")
  //     ("var2", po::value<var2_t>(&var2)->default_value(""), "Var2 Description")
  //     ("var3", po::value<var3_t>(&var3)->implicit_value(""), "Var3 Description");
  // po::variables_map vm;
  // po::store(po::parse_command_line(argc, argv, desc), vm);
  // po::notify(vm);
  // if (vm.count("help")) {
  //   cout << desc << endl;
  //   return 1;
  // }

  Vector2d center(3.5, 1.5);
  MatrixXd pts(2, 6), nms(2, 6);
  pts << 3.77, 3.13, 3.72, 3.81, 3.25, 3.46,
         1.86, 1.50, 1.46, 1.26, 1.88, 1.26;
  nms << 0.20, 0.05, 0.20, 0.20, 0.13, 0.13,
         0.98, 1.00, 0.98, 0.98, 0.99, 0.99;
  auto cell = MakeCell(center, pts, nms, Affine2d::Identity(), Vector2d(0.0625, 0.0001), 1);
  ros::init(argc, argv, "test_0630");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<visualization_msgs::MarkerArray>("marker", 0, true);
  ros::Publisher pub2 = nh.advertise<visualization_msgs::MarkerArray>("marker2", 0, true);
  pub.publish(MarkerArrayOfNDTCell(&cell));
  pub2.publish(MakePointsNormalMarker(pts, nms));
  cout << cell.ToString() << endl;
  ros::spin();
}

