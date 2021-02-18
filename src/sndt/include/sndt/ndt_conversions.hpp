#pragma once

#include <bits/stdc++.h>
#include <ros/ros.h>

#include <Eigen/Eigen>

#include "sndt/NDTCellMsg.h"
#include "sndt/NDTMapMsg.h"
#include "sndt/ndt_map_2d.hpp"

using namespace std;

inline bool toMessage(NDTMap *map, sndt::NDTMapMsg &msg, string frame_name) {
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = frame_name;
  if (!map->getGridSizeInMeters(msg.x_size, msg.y_size)) {
    return false;
  }
  if (!map->getCentroid(msg.x_cen, msg.y_cen)) {
    return false;
  }
  if (!map->getCellSizeInMeters(msg.x_cell_size, msg.y_cell_size)) {
    return false;
  }
  auto cells = map->getAllCells();
  for (size_t i = 0; i < cells.size(); ++i) {
    if (cells.at(i) != NULL) {
      sndt::NDTCellMsg cell;
      cell.phasGaussian_ = cells.at(i)->phasGaussian_;
      cell.nhasGaussian_ = cells.at(i)->nhasGaussian_;
      cell.center_x = cells.at(i)->getCenter().x;
      cell.center_y = cells.at(i)->getCenter().y;
      if (cells.at(i)->phasGaussian_) {
        Eigen::Vector2d mean = cells.at(i)->getPointMean();
        cell.pmean_x = mean(0);
        cell.pmean_y = mean(1);
        Eigen::Matrix2d cov = cells.at(i)->getPointCov();
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 2; j++) {
            cell.pcov.push_back(cov(i, j));
          }
        }
        cell.N = cells.at(i)->getN();
      }
      if (cells.at(i)->nhasGaussian_) {
        Eigen::Vector2d mean = cells.at(i)->getNormalMean();
        cell.nmean_x = mean(0);
        cell.nmean_y = mean(1);
        Eigen::Matrix2d cov = cells.at(i)->getNormalCov();
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 2; j++) {
            cell.ncov.push_back(cov(i, j));
          }
        }
      }
      msg.cells.push_back(cell);
    }
  }
  return true;
}
