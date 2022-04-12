// Snake Test
#include <common/common.h>
#include <ndt/costs.h>
#include <ndt/matcher2d.h>
#include <ndt/visuals.h>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

using namespace std;
using namespace Eigen;
using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

Affine2d Match(NDTMatcher2D &m, const Affine2d &offset) {
  auto res = m.Align();
  auto diff = TransNormRotDegAbsFromAffine2d(res * offset);
  printf("etl: %.4f, erot: %.4f, iter: %d, opt: %.2f, ttl: %.2f\n", diff(0),
         diff(1), m.iteration(), m.timer().optimize() / 1000.,
         m.timer().total() / 1000.);
  return res;
}

Marker MText(string text, const Vector2d &position) {
  Marker ret;
  ret.header.frame_id = "map";
  ret.id = 0;
  ret.type = Marker::TEXT_VIEW_FACING;
  ret.text = text;
  ret.pose.position.x = position(0);
  ret.pose.position.y = position(1);
  ret.pose.position.z = 0.;
  ret.pose.orientation.w = 1;
  ret.scale.z = 0.5;
  ret.color.r = ret.color.g = ret.color.b = 1;
  ret.color.a = 1;
  return ret;
}

Marker MMM(const std::vector<Eigen::Vector2d> &points, bool red) {
  Marker ret;
  ret.header.frame_id = "map";
  ret.pose.orientation.w = 1.;
  ret.header.stamp = ros::Time::now();
  ret.type = Marker::SPHERE_LIST;
  ret.scale.x = ret.scale.y = ret.scale.z = 0.1;
  ret.color.a = 0.7, ret.color.r = red ? 1 : 0, ret.color.g = red ? 0 : 1;
  for (const auto &point : points) {
    geometry_msgs::Point pt;
    pt.x = point(0), pt.y = point(1), pt.z = 0.;
    ret.points.push_back(pt);
  }
  return ret;
}

Marker Corre(const NMap2D &src,
             const NMap2D &tgt,
             const Affine2d &T = Affine2d::Identity()) {
  Marker ret;
  ret.header.frame_id = "map";
  ret.pose.orientation.w = 1.;
  ret.header.stamp = ros::Time::now();
  ret.type = Marker::LINE_LIST;
  ret.scale.x = 0.05;
  ret.color.a = 1., ret.color.b = 1., ret.color.g = 1.;
  auto next = src.TransformCells(T);

  for (auto p : next) {
    if (!p.GetHasGaussian()) continue;
    auto q = tgt.SearchNearestCell(p.GetMean());
    if (!q.GetHasGaussian()) continue;
    geometry_msgs::Point pt;
    pt.x = p.GetMean()(0), pt.y = p.GetMean()(1), pt.z = 0.;
    ret.points.push_back(pt);
    pt.x = q.GetMean()(0), pt.y = q.GetMean()(1), pt.z = 0.;
    ret.points.push_back(pt);
  }
  return ret;
}

MarkerArray Arrows(const NMap2D &src,
                   const NMap2D &tgt,
                   const Affine2d &T = Affine2d::Identity()) {
  MarkerArray ret;
  int id = 0;
  auto InitMarker = [&id]() {
    Marker ret;
    ret.id = id++;
    ret.header.frame_id = "map";
    ret.header.stamp = ros::Time::now();
    ret.type = Marker::ARROW;
    ret.scale.x = 0.05, ret.scale.y = 0.2;
    ret.pose.orientation.w = 1;
    ret.color.a = 1;
    return ret;
  };

  for (const auto &elem : tgt) {
    const Cell2D &cell = elem.second;
    Marker ar = InitMarker();
    ar.color.r = 1;

    geometry_msgs::Point pt;
    pt.x = cell.GetMean()(0), pt.y = cell.GetMean()(1), pt.z = 0.;
    ar.points.push_back(pt);
    auto nm = cell.GetNormal();
    if (nm.dot(Vector2d(-3, 4)) < 0) nm *= -1;
    pt.x += nm(0) * 0.7, pt.y += nm(1) * 0.7;

    ar.points.push_back(pt);
    ret.markers.push_back(ar);
  }

  auto next = src.TransformCells(T);
  for (const auto &cell : next) {
    Marker ar = InitMarker();
    ar.color.g = 1;

    geometry_msgs::Point pt;
    pt.x = cell.GetMean()(0), pt.y = cell.GetMean()(1), pt.z = 0.;
    ar.points.push_back(pt);
    auto nm = cell.GetNormal();
    if (nm.dot(Vector2d(-3, 4)) < 0) nm *= -1;
    pt.x += nm(0) * 0.7, pt.y += nm(1) * 0.7;

    ar.points.push_back(pt);
    ret.markers.push_back(ar);
  }
  return ret;
}

int main(int argc, char **argv) {
  vector<Vector2d> target;
  for (double x = -3; x < 3; x += 0.02) {
    double y = 0.3 * (x + 4) * sin(x + 4) + 0.1;
    target.push_back(Vector2d(x, y));
  }

  double x = atof(argv[1]);
  double y = atof(argv[2]);
  double t = atof(argv[3]);
  Affine2d offset = Translation2d(x, y) * Rotation2Dd(Deg2Rad(t));
  auto source = TransformPoints(target, offset);

  auto m1 = NDTMatcher2D::GetBasic({kNDT, k1to1, kNoReject, kPointCov},
                                   0.5, 0.05);
  m1.SetSource(source);
  m1.SetTarget(target);
  Match(m1, offset);

  auto m2 = NDTMatcher2D::GetBasic({kNNDT, k1to1, kNoReject, kPointCov},
                                   0.5, 0.05);
  m2.SetSource(source);
  m2.SetTarget(target);
  Match(m2, offset);

  ros::init(argc, argv, "exp9");
  ros::NodeHandle nh;
  ros::Publisher pu1 = nh.advertise<Marker>("marker1", 0, true);
  ros::Publisher pu2 = nh.advertise<Marker>("marker2", 0, true);
  ros::Publisher pu3 = nh.advertise<Marker>("marker3", 0, true);
  ros::Publisher pu4 = nh.advertise<Marker>("marker4", 0, true);
  ros::Publisher pub1 = nh.advertise<MarkerArray>("markers1", 0, true);
  ros::Publisher pub2 = nh.advertise<MarkerArray>("markers2", 0, true);
  ros::Publisher pub3 = nh.advertise<MarkerArray>("markers3", 0, true);
  ros::Publisher pub4 = nh.advertise<MarkerArray>("markers4", 0, true);
  pub1.publish(MarkerOfNDT(m1.tmap(), {kRed, kCell, kCov}));
  int i;
  cout << "Index: ";
  while (cin >> i) {
    if (i >= 0 && i < (int)m1.tfs().size()) {
      pu1.publish(Corre(*m1.smap(), *m1.tmap(), m1.tfs()[i]));
      pu2.publish(MText("NDT: " + to_string(i), Vector2d(-2, -1)));
      pub2.publish(MarkerOfNDT(m1.smap(), {kGreen, kCell, kCov}, m1.tfs()[i]));
    }
    if (i >= 0 && i < (int)m2.tfs().size()) {
      pu3.publish(Corre(*m2.smap(), *m2.tmap(), m2.tfs()[i]));
      pu4.publish(MText("Normal NDT: " + to_string(i), Vector2d(-2, -1)));
      pub3.publish(MarkerOfNDT(m2.smap(), {kGreen, kCell, kCov}, m2.tfs()[i]));
      pub4.publish(Arrows(*m2.smap(), *m2.tmap(), m2.tfs()[i]));
    }
    cout << "Index: ";
  }
}
