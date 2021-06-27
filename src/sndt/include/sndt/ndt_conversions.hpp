#pragma once

#include <bits/stdc++.h>
#include <ros/ros.h>

#include <Eigen/Eigen>

#include <common/common.h>
#include "sndt/NDTCellMsg.h"
#include "sndt/NDTMapMsg.h"
#include "sndt/ndt_map_2d.hpp"

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

// This function updates type, action, pose and scale
void UpdateMarkerByMeanAndCov(visualization_msgs::Marker &marker,
                              const Vector2d &mean,
                              const Matrix2d &covariance) {
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;
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
  marker.scale.x = 3 * eval(0, 0);
  marker.scale.y = 3 * eval(1, 1);
  marker.scale.z = 0.1;
  marker.pose.position.x = mean(0);
  marker.pose.position.y = mean(1);
  marker.pose.position.z = 0;
  marker.pose.orientation = tf2::toMsg(q);
}

// This function updates type, action, pose and scale
void UpdateMarkerByCellCenterAndCellSize(visualization_msgs::Marker &marker,
                                         const Vector2d &cell_center,
                                         const Vector2d &cell_size) {
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;
  double halfsize = cell_size(0) / 2;
  vector<double> dxy = {halfsize, halfsize, -halfsize, -halfsize};
  for (int i = 0; i < 5; ++i) {
    geometry_msgs::Point pt;
    pt.x = cell_center(0) + dxy[i % 4];
    pt.y = cell_center(1) + dxy[(i + 1) % 4];
    pt.z = 0;
    marker.points.push_back(pt);
  }
  marker.scale.x = 0.1;
  marker.pose = tf2::toMsg(Affine3d::Identity());
}

// This function updates type, action, pose and scale
void UpdateMarkerByEndPoints(visualization_msgs::Marker &marker, vector<Vector2d> points) {
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;
  for (auto point : points) {
    geometry_msgs::Point pt;
    pt.x = point(0), pt.y = point(1), pt.z = 0;
    marker.points.push_back(pt);
  }
  marker.scale.x = 0.1;
  marker.pose = tf2::toMsg(Affine3d::Identity());
}

vector<Vector2d> FindTangentPoints(visualization_msgs::Marker eclipse, Vector2d point) {
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

visualization_msgs::MarkerArray MarkerArrayFromNDTCell(const NDTCell *cell) {
  visualization_msgs::MarkerArray ret;
  auto now = ros::Time::now();
  visualization_msgs::Marker m;
  m.header.frame_id = "map";
  m.header.stamp = now;
  m.id = -1;

  // 1. Grid Boundary
  // ++m.id;
  // UpdateMarkerByCellCenterAndCellSize(m, cell->GetCenter(), cell->GetSize());
  // m.color = common::MakeColorRGBA(common::Color::kBlack);
  // ret.markers.push_back(m);

  // 2. Point Covariance
  if (cell->GetPHasGaussian()) {
    ++m.id;
    UpdateMarkerByMeanAndCov(m, cell->GetPointMean(), cell->GetPointCov());
    m.color = common::MakeColorRGBA(common::Color::kLime, 0.7);
    ret.markers.push_back(m);
  }

  // 3. Normal Covariance + 4. Lines from PointMean to NormalCov
  if (cell->GetNHasGaussian()) {
    ++m.id;
    UpdateMarkerByMeanAndCov(m, cell->GetPointMean() + cell->GetNormalMean(), cell->GetNormalCov());
    m.color = common::MakeColorRGBA(common::Color::kGray, 0.7);
    ret.markers.push_back(m);

    auto points = FindTangentPoints(m, cell->GetPointMean());
    for (int i = 0; i < 2; ++i) {
      ++m.id;
      UpdateMarkerByEndPoints(m, {cell->GetPointMean(), points[i]});
      m.color = common::MakeColorRGBA(common::Color::kGray, 0.7);
      ret.markers.push_back(m);
    }
  }

  return ret;
}

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
