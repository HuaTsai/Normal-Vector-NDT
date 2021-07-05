#pragma once

#include <bits/stdc++.h>
#include <ros/ros.h>

#include <Eigen/Eigen>

#include <common/common.h>
#include "sndt/NDTCellMsg.h"
#include "sndt/NDTMapMsg.h"
#include "sndt/ndt_map_2d.hpp"
#include <pcl_ros/point_cloud.h>

using namespace std;
using namespace Eigen;

sndt::NDTMapMsg ToMessage(const NDTMap &map, string frame_name) {
  sndt::NDTMapMsg msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = frame_name;
  msg.map_size = {map.map_size()(0), map.map_size()(1)};
  msg.map_center = {map.map_center()(0), map.map_center()(1)};
  msg.cell_size = {map.cell_size()(0), map.cell_size()(1)};
  for (auto cell : map) {
    sndt::NDTCellMsg cellmsg;
    cellmsg.N = cell->GetN();
    cellmsg.phasgaussian = cell->GetPHasGaussian();
    cellmsg.nhasgaussian = cell->GetNHasGaussian();
    cellmsg.center = {cell->GetCenter()(0), cell->GetCenter()(1)};
    cellmsg.pmean = {cell->GetPointMean()(0), cell->GetPointMean()(1)};
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        cellmsg.pcov.push_back(cell->GetPointCov()(i, j));
    cellmsg.nmean = {cell->GetNormalMean()(0), cell->GetNormalMean()(1)};
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        cellmsg.ncov.push_back(cell->GetNormalCov()(i, j));
    msg.cells.push_back(cellmsg);
  }
  return msg;
}

vector<Vector2d> FindTangentPoints(const visualization_msgs::Marker &eclipse, const Vector2d &point) {
  Affine3d aff3;
  tf2::fromMsg(eclipse.pose, aff3);
  Matrix3d mtx = Matrix3d::Identity();
  mtx.block<2, 2>(0, 0) = aff3.rotation().block<2, 2>(0, 0);
  mtx.block<2, 1>(0, 2) = aff3.translation().block<2, 1>(0, 0);
  Affine2d aff2(mtx);
  auto point2 = aff2.inverse() * point;
  auto rx2 = (eclipse.scale.x / 2) * (eclipse.scale.x / 2);
  auto ry2 = (eclipse.scale.y / 2) * (eclipse.scale.y / 2);
  auto x0 = point2(0), x02 = x0 * x0;
  auto y0 = point2(1), y02 = y0 * y0;
  vector<Vector2d> sols(2);
  if (x02 == rx2) {
    auto msol = (-rx2 * ry2 * ry2 + rx2 * ry2 * y02) / (2 * rx2 * ry2 * x0 * y0);
    sols[0](0) = (msol * rx2 * (msol * x0 - y0)) / (msol * msol * rx2 + ry2);
    sols[0](1) = y0 + msol * (sols[0](0) - x0);
    sols[1](0) = x0;
    sols[1](1) = 0;
  } else {
    auto msol1 = (-x0 * y0 + sqrt(-rx2 * ry2 + rx2 * y02 + ry2 * x02)) / (rx2 - x02);
    sols[0](0) = (msol1 * rx2 * (msol1 * x0 - y0)) / (msol1 * msol1 * rx2 + ry2);
    sols[0](1) = y0 + msol1 * (sols[0](0) - x0);
    auto msol2 = (-x0 * y0 - sqrt(-rx2 * ry2 + rx2 * y02 + ry2 * x02)) / (rx2 - x02);
    sols[1](0) = (msol2 * rx2 * (msol2 * x0 - y0)) / (msol2 * msol2 * rx2 + ry2);
    sols[1](1) = y0 + msol2 * (sols[1](0) - x0);
  }
  for (auto &sol : sols)
    sol = aff2 * sol;
  return sols;
}

void UpdateMarkerArray(visualization_msgs::MarkerArray &markerarray,
                       visualization_msgs::Marker marker) {
  if (!markerarray.markers.size()) {
    markerarray.markers.push_back(marker);
    return;
  }
  marker.header.frame_id = "map";
  marker.header.stamp = markerarray.markers.back().header.stamp;
  marker.id = markerarray.markers.back().id + 1;
  markerarray.markers.push_back(marker);
}

visualization_msgs::MarkerArray JoinMarkers(
    const vector<visualization_msgs::Marker> &ms) {
  visualization_msgs::MarkerArray ret;
  int id = 0;
  auto now = ros::Time::now();
  for (const auto &m : ms) {
    ret.markers.push_back(m);
    ret.markers.back().header.stamp = now;
    ret.markers.back().id = id++;
  }
  return ret;
}

visualization_msgs::MarkerArray JoinMarkerArraysAndMarkers(
    const vector<visualization_msgs::MarkerArray> &mas,
    const vector<visualization_msgs::Marker> &ms = {}) {
  visualization_msgs::MarkerArray ret;
  int id = 0;
  auto now = ros::Time::now();
  for (const auto &ma : mas) {
    for (const auto &m : ma.markers) {
      ret.markers.push_back(m);
      ret.markers.back().header.stamp = now;
      ret.markers.back().id = id++;
    }
  }
  for (const auto &m : ms) {
    ret.markers.push_back(m);
    ret.markers.back().header.stamp = now;
    ret.markers.back().id = id++;
  }
  return ret;
}

visualization_msgs::Marker MarkerOfBoundary(
    const Vector2d &center, double size, double skew_rad = 0,
    const common::Color &color = common::Color::kBlack) {
  visualization_msgs::Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = ros::Time::now();
  ret.id = 0;
  ret.type = visualization_msgs::Marker::LINE_LIST;
  ret.action = visualization_msgs::Marker::ADD;
  ret.pose = tf2::toMsg(Affine3d::Identity());
  ret.scale.x = 0.02;
  ret.color = common::MakeColorRGBA(color);
  double r = size / 2, t = skew_rad;
  auto R = Rotation2Dd(t);
  vector<Vector2d> dxy{Vector2d(r, r), Vector2d(r, -r), Vector2d(-r, -r), Vector2d(-r, r), Vector2d(r, r)};
  transform(dxy.begin(), dxy.end(), dxy.begin(), [&R](auto a) { return R * a; });
  for (int i = 0; i < 4; ++i) {
    geometry_msgs::Point pt;
    pt.x = center(0) + dxy[i](0);
    pt.y = center(1) + dxy[i](1);
    pt.z = 0;
    ret.points.push_back(pt);
    pt.x = center(0) + dxy[i + 1](0);
    pt.y = center(1) + dxy[i + 1](1);
    ret.points.push_back(pt);
  }
  return ret;
}

visualization_msgs::Marker MarkerOfEclipse(
    const Vector2d &mean, const Matrix2d &covariance,
    const common::Color &color = common::Color::kLime, double alpha = 0.6) {
  visualization_msgs::Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = ros::Time::now();
  ret.id = 0;
  ret.type = visualization_msgs::Marker::SPHERE;
  ret.action = visualization_msgs::Marker::ADD;
  ret.color = common::MakeColorRGBA(color, alpha);
  EigenSolver<Matrix2d> es(covariance);
  /** Note: "pseudo" computes complex eigen value if no real solution
   * However, covariance is always a real symmetric matrix, which means
   *   1) it is Hermitia, all its eigenvalues are real
   *   2) the decomposed matrix is an orthogonal matrix, i.e., rotaion matrix
   */
  Matrix2d eval = es.pseudoEigenvalueMatrix().cwiseSqrt();
  Matrix2d evec = es.pseudoEigenvectors();
  Matrix3d R = Matrix3d::Identity();
  R.block<2, 2>(0, 0) = evec;
  Quaterniond q(R);
  ret.scale.x = 3 * eval(0, 0);
  ret.scale.y = 3 * eval(1, 1);
  ret.scale.z = 0.1;
  ret.pose.position.x = mean(0);
  ret.pose.position.y = mean(1);
  ret.pose.position.z = 0;
  ret.pose.orientation = tf2::toMsg(q);
  return ret;
}

visualization_msgs::Marker MarkerOfLines(
    const vector<Vector2d> &points,
    const common::Color &color = common::Color::kGray, double alpha = 0.6) {
  Expects(points.size() % 2 == 0);
  visualization_msgs::Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = ros::Time::now();
  ret.id = 0;
  ret.type = visualization_msgs::Marker::LINE_LIST;
  ret.action = visualization_msgs::Marker::ADD;
  ret.scale.x = 0.05;
  ret.pose = tf2::toMsg(Affine3d::Identity());
  ret.color = common::MakeColorRGBA(color, alpha);
  for (const auto &p : points) {
    geometry_msgs::Point pt;
    pt.x = p(0), pt.y = p(1), pt.z = 0;
    ret.points.push_back(pt);
  }
  return ret;
}

MatrixXd PointMatrixXdOfNDTMap(const NDTMap &map) {
  vector<Vector2d> pts;
  for (auto cell : map)
    for (auto pt : cell->GetPoints())
      if (pt.allFinite())
        pts.push_back(pt);
  MatrixXd ret(2, pts.size());
  for (int i = 0; i < ret.cols(); ++i)
    ret.col(i) = pts[i];
  return ret;
}

MatrixXd PointMatrixXdOfNDTMap(const vector<shared_ptr<NDTCell>> &map) {
  vector<Vector2d> pts;
  for (auto cell : map)
    for (auto pt : cell->GetPoints())
      if (pt.allFinite())
        pts.push_back(pt);
  MatrixXd ret(2, pts.size());
  for (int i = 0; i < ret.cols(); ++i)
    ret.col(i) = pts[i];
  return ret;
}

visualization_msgs::Marker MarkerOfPoints(
    const MatrixXd &points, double size = 0.1,
    const common::Color &color = common::Color::kLime, double alpha = 1.0) {
  auto now = ros::Time::now();
  visualization_msgs::Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = now;
  ret.id = 0;
  ret.type = visualization_msgs::Marker::SPHERE_LIST;
  ret.action = visualization_msgs::Marker::ADD;
  ret.scale.x = ret.scale.y = ret.scale.z = size;
  ret.pose = tf2::toMsg(Affine3d::Identity());
  ret.color = common::MakeColorRGBA(color);
  for (int i = 0; i < points.cols(); ++i) {
    geometry_msgs::Point pt;
    pt.x = points(0, i), pt.y = points(1, i), pt.z = 0;
    ret.points.push_back(pt);
  }
  return ret;
}

visualization_msgs::MarkerArray MarkerArrayOfArrow(
    const MatrixXd &start, const MatrixXd &end,
    const common::Color &color = common::Color::kRed, double alpha = 1.0) {
  Expects(start.cols() == end.cols() &&
          start.rows() == end.rows() &&
          start.rows() == 2);
  visualization_msgs::MarkerArray ret;
  visualization_msgs::Marker arrow;
  arrow.header.frame_id = "map";
  arrow.header.stamp = ros::Time::now();
  arrow.id = -1;
  arrow.type = visualization_msgs::Marker::ARROW;
  arrow.action = visualization_msgs::Marker::ADD;
  arrow.scale.x = 0.05;
  arrow.scale.y = 0.2;
  arrow.pose = tf2::toMsg(Affine3d::Identity());
  arrow.color = common::MakeColorRGBA(color, alpha);
  arrow.points.resize(2);
  for (int i = 0; i < start.cols(); ++i) {
    if (!start.col(i).allFinite() || !end.col(i).allFinite())
      continue;
    ++arrow.id;
    geometry_msgs::Point pt;
    pt.x = start(0, i), pt.y = start(1, i), pt.z = 0;
    arrow.points[0] = pt;
    pt.x = end(0, i), pt.y = end(1, i);
    arrow.points[1] = pt;
    ret.markers.push_back(arrow);
  }
  return ret;
}

// Color set for general usage
visualization_msgs::MarkerArray MarkerArrayOfNDTCell(const NDTCell *cell) {
  visualization_msgs::MarkerArray ret;
  auto boundary = MarkerOfBoundary(cell->GetCenter(), cell->GetSize()(0), cell->GetSkewRad());
  auto p_eclipse = MarkerOfEclipse(cell->GetPointMean(), cell->GetPointCov());
  if (cell->GetNHasGaussian()) {
    auto n_eclipse =
        MarkerOfEclipse(cell->GetPointMean() + cell->GetNormalMean(),
                        cell->GetNormalCov(), common::Color::kGray, 0.6);
    auto points = FindTangentPoints(n_eclipse, cell->GetPointMean());
    auto lines = MarkerOfLines({points[0], cell->GetPointMean(), cell->GetPointMean(), points[1]});
    ret = JoinMarkerArraysAndMarkers({}, {boundary, p_eclipse, n_eclipse, lines});
  } else {
    ret = JoinMarkerArraysAndMarkers({}, {boundary, p_eclipse});
  }
  return ret;
}

// Color set for target point cloud
visualization_msgs::MarkerArray MarkerArrayOfNDTCell2(const NDTCell *cell) {
  visualization_msgs::MarkerArray ret;
  auto boundary = MarkerOfBoundary(cell->GetCenter(), cell->GetSize()(0), cell->GetSkewRad(), common::Color::kRed);
  auto p_eclipse = MarkerOfEclipse(cell->GetPointMean(), cell->GetPointCov(), common::Color::kRed);
  if (cell->GetNHasGaussian()) {
    auto n_eclipse =
        MarkerOfEclipse(cell->GetPointMean() + cell->GetNormalMean(),
                        cell->GetNormalCov(), common::Color::kGray);
    auto points = FindTangentPoints(n_eclipse, cell->GetPointMean());
    auto lines = MarkerOfLines({points[0], cell->GetPointMean(), cell->GetPointMean(), points[1]}, common::Color::kGray);
    ret = JoinMarkerArraysAndMarkers({}, {boundary, p_eclipse, n_eclipse, lines});
  } else {
    ret = JoinMarkerArraysAndMarkers({}, {boundary, p_eclipse});
  }
  return ret;
}

visualization_msgs::MarkerArray MarkerArrayOfNDTMap(const NDTMap &map, bool is_target_color = false) {
  vector<visualization_msgs::MarkerArray> vma;
  for (auto cell : map) {
    if (is_target_color)
      vma.push_back(MarkerArrayOfNDTCell2(cell));
    else
      vma.push_back(MarkerArrayOfNDTCell(cell));
  }
  return JoinMarkerArraysAndMarkers(vma);
}

visualization_msgs::MarkerArray MarkerArrayOfNDTMap(const vector<shared_ptr<NDTCell>> &map, bool is_target_color = false) {
  vector<visualization_msgs::MarkerArray> vma;
  for (auto cell : map) {
    if (is_target_color)
      vma.push_back(MarkerArrayOfNDTCell2(cell.get()));
    else
      vma.push_back(MarkerArrayOfNDTCell(cell.get()));
  }
  return JoinMarkerArraysAndMarkers(vma);
}

visualization_msgs::MarkerArray MarkerArrayOfSensor(const vector<Affine2d> &affs) {
  visualization_msgs::MarkerArray ret;
  auto now = ros::Time::now();
  int i = 0;
  for (const auto &aff : affs) {
    visualization_msgs::Marker m;
    m.header.frame_id = "map";
    m.header.stamp = ros::Time::now();
    m.id = i++;
    m.type = visualization_msgs::Marker::CUBE;
    m.action = visualization_msgs::Marker::ADD;
    m.pose = tf2::toMsg(common::Affine3dFromAffine2d(aff));
    m.color = common::MakeColorRGBA(common::Color::kGray);
    m.scale.x = 0.3;
    m.scale.y = 1.0;
    m.scale.z = 0.5;
    ret.markers.push_back(m);
  }
  return ret;
}

// TODO: followings are unused yet match_vis.cc
/** A -> B -> C -> D -> A
 * y D-----A
 *   |.....|
 *   |..X..|
 *   |.....|
 *   C-----B
 * 0       x
 */
visualization_msgs::MarkerArray GridMarkerArrayFromNDTMapMsg(const sndt::NDTMapMsg &msg, common::Color color) {
  visualization_msgs::MarkerArray ret;
  auto now = ros::Time::now();
  int idx = -1;
  double xsize = msg.cell_size[0], ysize = msg.cell_size[1];
  for (const auto &cell : msg.cells) {
    if (cell.phasgaussian && cell.nhasgaussian) {
      visualization_msgs::Marker m;
      m.header.frame_id = msg.header.frame_id;
      m.header.stamp = now;
      m.id = ++idx;
      m.type = visualization_msgs::Marker::LINE_STRIP;
      m.action = visualization_msgs::Marker::ADD;
      vector<double> dxs = {xsize / 2, xsize / 2, -xsize / 2, -xsize / 2, xsize / 2};
      vector<double> dys = {ysize / 2, -ysize / 2, -ysize / 2, ysize / 2, ysize / 2};
      for (int i = 0; i < 5; ++i) {
        geometry_msgs::Point pt;
        pt.x = cell.center[0] + dxs[i];
        pt.y = cell.center[1] + dys[i];
        pt.z = 0;
        m.points.push_back(pt);
      }
      m.scale.x = 0.1;
      m.color = common::MakeColorRGBA(color);
      m.pose.orientation.w = 1;
      ret.markers.push_back(m);
    }
  }
  return ret;
}

visualization_msgs::MarkerArray NormalMarkerArrayFromNDTMapMsg(const sndt::NDTMapMsg &msg, common::Color color) {
  visualization_msgs::MarkerArray ret;
  auto now = ros::Time::now();
  int idx = -1;
  for (const auto &cell : msg.cells) {
    if (cell.phasgaussian && cell.nhasgaussian) {
      visualization_msgs::Marker m;
      m.header.frame_id = msg.header.frame_id;
      m.header.stamp = now;
      m.id = ++idx;
      m.type = visualization_msgs::Marker::ARROW;
      m.action = visualization_msgs::Marker::ADD;
      geometry_msgs::Point pt;
      pt.x = cell.pmean[0];
      pt.y = cell.pmean[1];
      pt.z = 0;
      m.points.push_back(pt);
      pt.x += cell.nmean[0];
      pt.y += cell.nmean[1];
      m.points.push_back(pt);
      m.scale.x = 0.05;
      m.scale.y = 0.2;
      m.color = common::MakeColorRGBA(color);
      m.pose.orientation.w = 1;
      ret.markers.push_back(m);
    }
  }
  return ret;
}

visualization_msgs::MarkerArray NormalCovMarkerArrayFromNDTMapMsg(const sndt::NDTMapMsg &msg, common::Color color) {
  visualization_msgs::MarkerArray ret;
  auto now = ros::Time::now();
  int idx = -1;
  for (const auto &cell : msg.cells) {
    if (cell.phasgaussian && cell.nhasgaussian) {
      visualization_msgs::Marker m;
      m.header.frame_id = msg.header.frame_id;
      m.header.stamp = now;
      m.id = ++idx;
      m.type = visualization_msgs::Marker::SPHERE;
      m.action = visualization_msgs::Marker::ADD;
      m.pose.position.x = cell.pmean[0] + cell.nmean[0];
      m.pose.position.y = cell.pmean[1] + cell.nmean[1];
      m.pose.position.z = 0;
      Matrix3d cov = Matrix3d::Identity();
      for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
          cov(i, j) = cell.pcov[2 * i + j];
      EigenSolver<Matrix3d> es(cov);
      Matrix3d eval = es.pseudoEigenvalueMatrix().cwiseSqrt();
      Matrix3d evec = es.pseudoEigenvectors();
      Quaterniond q(evec);
      m.scale.x = 3 * eval(0, 0);
      m.scale.y = 3 * eval(1, 1);
      m.scale.z = 0.1;
      m.pose.orientation.w = q.w();
      m.pose.orientation.x = q.x();
      m.pose.orientation.y = q.y();
      m.pose.orientation.z = q.z();
      m.color = common::MakeColorRGBA(color);
      m.color.a = 0.3;
      ret.markers.push_back(m);
    }
  }
  return ret;
}
