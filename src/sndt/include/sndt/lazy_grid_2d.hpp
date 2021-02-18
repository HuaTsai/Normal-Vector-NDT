#pragma once

#include <bits/stdc++.h>
#include "common/common.h"
#include "sndt/ndt_cell_2d.hpp"

using namespace std;

class LazyGrid2D {
 public:
  typedef vector<NDTCell *> CellPtrVector;
  typedef typename CellPtrVector::iterator CellVectorItr;
  typedef typename CellPtrVector::const_iterator CellVectorConstItr;

  LazyGrid2D(double cellSize) {
    centerIsSet = sizeIsSet = initialized = false;
    cellSizeX = cellSizeY = cellSize;
  }

  ~LazyGrid2D() {
    if (initialized) {
      for (size_t i = 0; i < activeCells.size(); ++i) {
        if (activeCells[i]) {
          delete activeCells[i];
        }
      }
      for (int i = 0; i < sizeX; ++i) {
        delete[] dataArray[i];
      }
      delete[] dataArray;
    }
  }

  CellVectorItr begin() { return activeCells.begin(); }

  CellVectorConstItr begin() const { return activeCells.begin(); }

  CellVectorItr end() { return activeCells.end(); }

  CellVectorConstItr end() const { return activeCells.end(); }

  int size() { return activeCells.size(); }

  NDTCell *getCellForPoint(const pcl::PointXY &point) {
    int indX, indY;
    this->getIndexForPoint(point, indX, indY);
    if (indX >= sizeX || indY >= sizeY || indX < 0 || indY < 0) return NULL;
    if (!initialized) return NULL;
    if (dataArray == NULL) return NULL;
    if (dataArray[indX] == NULL) return NULL;
    return dataArray[indX][indY];
  }

  NDTCell *addPoint(const pcl::PointXY &point) {
    NDTCell *cell;
    this->getCellAtAllocate(point, cell);
    if (cell != NULL) cell->addPoint(point);
    return cell;
  }

  void getNeighbors(const pcl::PointXY &point, double radius, vector<NDTCell *> &cells) {
    int indX, indY;
    this->getIndexForPoint(point, indX, indY);
    if (indX >= sizeX || indY >= sizeY) {
      cells.clear();
      return;
    }
    for (int x = indX - radius / cellSizeX; x <= indX + radius / cellSizeX; x++) {
      if (x < 0 || x >= sizeX) continue;
      for (int y = indY - radius / cellSizeY; y <= indY + radius / cellSizeY; y++) {
        if (y < 0 || y >= sizeY) continue;
        if (dataArray[x][y] == NULL) continue;
        cells.push_back(dataArray[x][y]);
      }
    }
  }

  void setCenter(const double &cx, const double &cy) {
    centerX = cx;
    centerY = cy;
    centerIsSet = true;
    if (sizeIsSet) {
      initialize();
    }
  }

  void setSize(const double &sx, const double &sy) {
    sizeXmeters = sx;
    sizeYmeters = sy;
    sizeX = abs(ceil(sizeXmeters / cellSizeX));
    sizeY = abs(ceil(sizeYmeters / cellSizeY));
    sizeIsSet = true;
    if (centerIsSet) {
      initialize();
    }
  }

  bool insertCell(NDTCell cell) {
    int indX, indY;
    this->getIndexForPoint(cell.getCenter(), indX, indY);
    if (indX < 0 || indX >= sizeX) return false;
    if (indY < 0 || indY >= sizeY) return false;
    if (!initialized) return false;
    if (dataArray == NULL) return false;
    if (dataArray[indX] == NULL) return false;
    if (dataArray[indX][indY] == NULL) {
      dataArray[indX][indY] = new NDTCell();
      activeCells.push_back(dataArray[indX][indY]);
    }
    NDTCell *ret = dataArray[indX][indY];
    double xs, ys;
    cell.getDimensions(xs, ys);
    ret->setDimensions(xs, ys);
    ret->setCenter(cell.getCenter());
    ret->setPointMean(cell.getPointMean());
    ret->setPointCov(cell.getPointCov());
    ret->setNormalMean(cell.getNormalMean());
    ret->setNormalCov(cell.getNormalCov());
    ret->setN(cell.getN());
    ret->phasGaussian_ = cell.phasGaussian_;
    ret->nhasGaussian_ = cell.nhasGaussian_;
    return true;
  }

  NDTCell *getClosestNDTCell(const pcl::PointXY &point,
                             int max_neighbors,
                             bool checkForGaussian) {
    int indXn, indYn, indX, indY;
    this->getIndexForPoint(point, indX, indY);
    NDTCell *ret = NULL;
    vector<NDTCell *> cells;

    if (!checkForGaussian) {
      if (checkCellforNDT(indX, indY, checkForGaussian)) {
        ret = (dataArray[indX][indY]);
      }
      return ret;
    }

    for (int x = 1; x < 2 * max_neighbors + 2; x++) {
      indXn = (x % 2 == 0) ? indX + x / 2 : indX - x / 2;
      for (int y = 1; y < 2 * max_neighbors + 2; y++) {
        indYn = (y % 2 == 0) ? indY + y / 2 : indY - y / 2;
        if (checkCellforNDT(indXn, indYn, true)) {
          ret = dataArray[indXn][indYn];
          cells.push_back(ret);
        }
      }
    }

    double minDist = INT_MAX;
    Eigen::Vector2d tmean;
    pcl::PointXY pt = point;
    for (size_t i = 0; i < cells.size(); i++) {
      tmean = cells[i]->getPointMean() - Eigen::Vector2d(pt.x, pt.y);
      double d = tmean.norm();
      if (d < minDist) {
        minDist = d;
        ret = cells[i];
      }
    }
    cells.clear();
    return ret;
  }

  vector<NDTCell *> getClosestNDTCells(const pcl::PointXY &pt,
                                            int n_neigh,
                                            bool checkForGaussian) {
    int indXn, indYn, indX, indY;
    this->getIndexForPoint(pt, indX, indY);
    vector<NDTCell *> cells;
    int i = n_neigh;
    for (int x = 1; x < 2 * i + 2; x++) {
      indXn = (x % 2 == 0) ? indX + x / 2 : indX - x / 2;
      for (int y = 1; y < 2 * i + 2; y++) {
        indYn = (y % 2 == 0) ? indY + y / 2 : indY - y / 2;
        if (checkCellforNDT(indXn, indYn, checkForGaussian)) {
          cells.push_back(dataArray[indXn][indYn]);
        }
      }
    }
    return cells;
  }

  vector<NDTCell *> getClosestCells(const pcl::PointXY &pt) {
    return getClosestNDTCells(pt, 2, true);
  }

  inline void getCellAt(int indX, int indY, NDTCell *&cell) {
    if (indX < sizeX && indY < sizeY && indX >= 0 && indY >= 0) {
      cell = dataArray[indX][indY];
    } else {
      cell = NULL;
    }
  }

  inline void getCellAt(const pcl::PointXY &pt, NDTCell *&cell) {
    int indX, indY;
    this->getIndexForPoint(pt, indX, indY);
    this->getCellAt(indX, indY, cell);
  }

  void getCellAtAllocate(const pcl::PointXY &pt, NDTCell *&cell) {
    cell = NULL;
    pcl::PointXY point = pt;
    if (isnan(point.x) || isnan(point.y)) {
      return;
    }
    int indX, indY;
    this->getIndexForPoint(point, indX, indY);

    if (indX >= sizeX || indY >= sizeY || indX < 0 || indY < 0) {
      return;
    }

    if (!initialized) return;
    if (dataArray == NULL) return;
    if (dataArray[indX] == NULL) return;
    if (dataArray[indX][indY] == NULL) {
      dataArray[indX][indY] = new NDTCell();
      int idcX, idcY;
      pcl::PointXY center, centerCell;
      center.x = centerX;
      center.y = centerY;
      this->getIndexForPoint(center, idcX, idcY);
      centerCell.x = centerX + (indX - idcX) * cellSizeX;
      centerCell.y = centerY + (indY - idcY) * cellSizeY;
      dataArray[indX][indY]->setCenter(centerCell);
      dataArray[indX][indY]->setDimensions(cellSizeX, cellSizeY);
      activeCells.push_back(dataArray[indX][indY]);
    }
    cell = dataArray[indX][indY];
  }

  double getCellSize() {
    return cellSizeX;
  }

  void getCellSize(double &cx, double &cy) {
    cx = cellSizeX;
    cy = cellSizeY;
  }

  void getCenter(double &cx, double &cy) {
    cx = centerX;
    cy = centerY;
  }

  void getGridSize(int &cx, int &cy) {
    cx = sizeX;
    cy = sizeY;
  }

  void getGridSizeInMeters(double &cx, double &cy) {
    cx = sizeXmeters;
    cy = sizeYmeters;
  }

  void getIndexForPoint(const pcl::PointXY &pt, int &idx, int &idy) {
    idx = floor((pt.x - centerX) / cellSizeX + 0.5) + sizeX / 2.0;
    idy = floor((pt.y - centerY) / cellSizeY + 0.5) + sizeY / 2.0;
  }

  void initialize() {
    if (initialized) return;
    dataArray = new NDTCell **[sizeX];
    for (int i = 0; i < sizeX; i++) {
      dataArray[i] = new NDTCell *[sizeY];
      memset(dataArray[i], 0, sizeY * sizeof(NDTCell *));
    }
    initialized = true;
  }

  void initializeAll() {
    if (!initialized) {
      this->initialize();
    }

    int idcX, idcY;
    pcl::PointXY center;
    center.x = centerX;
    center.y = centerY;
    this->getIndexForPoint(center, idcX, idcY);

    pcl::PointXY centerCell;
    for (int i = 0; i < sizeX; i++) {
      for (int j = 0; j < sizeY; j++) {
        dataArray[i][j] = new NDTCell();
        dataArray[i][j]->setDimensions(cellSizeX, cellSizeY);

        centerCell.x = centerX + (i - idcX) * cellSizeX;
        centerCell.y = centerY + (j - idcY) * cellSizeY;

        dataArray[i][j]->setCenter(centerCell);
        activeCells.push_back(dataArray[i][j]);
      }
    }
  };

  NDTCell ***getDataArrayPtr() { return dataArray; }

  bool isInside(const pcl::PointXY &pt) {
    int indX, indY;
    this->getIndexForPoint(pt, indX, indY);
    return (indX < sizeX && indY < sizeY && indX >= 0 && indY >= 0);
  }

  void addNDTCell(NDTCell *cell) {
    activeCells.push_back(cell);
  }

  void ToString() {
    dprintf("sizeIsSet: %s\n", (sizeIsSet ? "true" : "false"));
    dprintf("centerIsSet: %s\n", (centerIsSet ? "true" : "false"));
    dprintf("initialized: %s\n", (initialized ? "true" : "false"));
    dprintf("sizeXYmeters: (%.2f, %.2f)\n", sizeXmeters, sizeYmeters);
    dprintf("cellSizeXY: (%.2f, %.2f)\n", cellSizeX, cellSizeY);
    dprintf("centerXY: (%.2f, %.2f)\n", centerX, centerY);
    dprintf("sizeXY: (%.2f, %.2f)\n", sizeX, sizeY);
    for (size_t i = 0; i < activeCells.size(); ++i) {
      dprintf("# [%ld] ", i);
      activeCells.at(i)->ToString();
    }
  }

 private:
  bool initialized, sizeIsSet, centerIsSet;
  NDTCell ***dataArray;
  vector<NDTCell *> activeCells;

  double sizeXmeters, sizeYmeters;
  double cellSizeX, cellSizeY;
  double centerX, centerY;
  int sizeX, sizeY;

  bool checkCellforNDT(int indX, int indY, bool checkForGaussian) {
    if (indX < sizeX && indY < sizeY && indX >= 0 && indY >= 0) {
      if (dataArray[indX][indY] != NULL) {
        if ((dataArray[indX][indY]->phasGaussian_ &&
             dataArray[indX][indY]->nhasGaussian_) || (!checkForGaussian)) {
          return true;
        }
      }
    }
    return false;
  }

  LazyGrid2D() { InitializeDefaultValues(); }

  void InitializeDefaultValues() {
    activeCells.clear();
    centerIsSet = sizeIsSet = initialized = false;
    sizeXmeters = sizeYmeters = 0;
    cellSizeX = cellSizeY  = 0;
    centerX = centerY = 0;
    sizeX = sizeY = 0;
  }

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
