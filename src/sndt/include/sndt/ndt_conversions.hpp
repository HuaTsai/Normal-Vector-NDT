#pragma once

#include <bits/stdc++.h>
#include <ros/ros.h>

#include <Eigen/Eigen>

#include <common/common.h>
#include "sndt/NDTCellMsg.h"
#include "sndt/NDTMapMsg.h"
#include "sndt/ndt_map_2d.hpp"

using namespace std;

sndt::NDTMapMsg ToMessage(const NDTMap &map, string frame_name) {
  sndt::NDTMapMsg msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = frame_name;
  msg.x_size = map.map_size()(0);
  msg.y_size = map.map_size()(1);
  msg.x_cen = map.map_center()(0);
  msg.y_cen = map.map_center()(1);
  msg.x_cell_size = map.cell_size()(0);
  msg.y_cell_size = map.cell_size()(1);
  for (auto cell : map) {
    sndt::NDTCellMsg cellmsg;
    cellmsg.N = cell->GetN();
    cellmsg.phasGaussian_ = cell->GetPHasGaussian();
    cellmsg.nhasGaussian_ = cell->GetNHasGaussian();
    cellmsg.center_x = cell->GetCenter()(0);
    cellmsg.center_y = cell->GetCenter()(1);
    cellmsg.pmean_x = cell->GetPointMean()(0);
    cellmsg.pmean_y = cell->GetPointMean()(1);
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        cellmsg.pcov.push_back(cell->GetPointCov()(i, j));
    cellmsg.nmean_x = cell->GetNormalMean()(0);
    cellmsg.nmean_y = cell->GetNormalMean()(1);
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        cellmsg.ncov.push_back(cell->GetNormalCov()(i, j));
    msg.cells.push_back(cellmsg);
  }
  return msg;
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
  double xsize = msg.x_cell_size;
  double ysize = msg.y_cell_size;
  for (const auto &cell : msg.cells) {
    if (cell.phasGaussian_ && cell.nhasGaussian_) {
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
        pt.x = cell.center_x + dxs[i];
        pt.y = cell.center_y + dys[i];
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
    if (cell.phasGaussian_ && cell.nhasGaussian_) {
      visualization_msgs::Marker m;
      m.header.frame_id = msg.header.frame_id;
      m.header.stamp = now;
      m.id = ++idx;
      m.type = visualization_msgs::Marker::ARROW;
      m.action = visualization_msgs::Marker::ADD;
      geometry_msgs::Point pt;
      pt.x = cell.pmean_x;
      pt.y = cell.pmean_y;
      pt.z = 0;
      m.points.push_back(pt);
      pt.x += cell.nmean_x;
      pt.y += cell.nmean_y;
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
    if (cell.phasGaussian_ && cell.nhasGaussian_) {
      visualization_msgs::Marker m;
      m.header.frame_id = msg.header.frame_id;
      m.header.stamp = now;
      m.id = ++idx;
      m.type = visualization_msgs::Marker::SPHERE;
      m.action = visualization_msgs::Marker::ADD;
      m.pose.position.x = cell.pmean_x + cell.nmean_x;
      m.pose.position.y = cell.pmean_y + cell.nmean_y;
      m.pose.position.z = 0;
      Eigen::Matrix3d cov;
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          cov(i, j) = cell.pcov.at(2 * i + j);
        }
      }
      Eigen::EigenSolver<Eigen::Matrix3d> es(cov);
      Eigen::Matrix3d eval = es.pseudoEigenvalueMatrix().cwiseSqrt();
      Eigen::Matrix3d evec = es.pseudoEigenvectors();
      Eigen::Quaternion<double> q(evec);
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
