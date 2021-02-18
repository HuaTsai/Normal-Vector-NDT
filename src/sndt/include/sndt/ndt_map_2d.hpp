#pragma once

#include <bits/stdc++.h>

#include "sndt/lazy_grid_2d.hpp"

using namespace std;

pcl::PointCloud<pcl::PointXY> ToPCLXY(const pcl::PointCloud<pcl::PointXYZ> &pc) {
  pcl::PointCloud<pcl::PointXY> ret;
  for (const auto &ptxyz : pc.points) {
    pcl::PointXY pt;
    pt.x = ptxyz.x;
    pt.y = ptxyz.y;
    ret.points.push_back(pt);
  }
  return ret;
}

class NDTMap {
 public:
  NDTMap(LazyGrid2D *idx = NULL) {
    index_ = idx;
    map_sizex = map_sizey = -1.0;
    centerx = centery = 0.0;
    guess_size_ = true;
    isFirstLoad_ = true;
  }

  /** This function is usually not to be called. */
  void initialize(double cenx, double ceny, double sizex, double sizey) {
    index_->setCenter(cenx, ceny);
    index_->setSize(sizex, sizey);
    map_sizex = sizex;
    map_sizey = sizey;
    centerx = cenx;
    centery = ceny;
    index_->initialize();
    guess_size_ = false;
    isFirstLoad_ = false;
  }

  ~NDTMap() {
    if (index_ != NULL && !isFirstLoad_) {
      delete index_;
      index_ = NULL;
    }
  }

  void setMapSize(double sx, double sy) {
    map_sizex = sx;
    map_sizey = sy;
  }

  void setNormalRadiusSearch(int radius) {
    radius_ = radius;
  }

  // TODO: NOT IMPLEMENT YET!!!
  void addPointCloud(const pcl::PointCloud<pcl::PointXYZ> &pc, double maxz) {
    ;
  }

  void loadPointCloud(const pcl::PointCloud<pcl::PointXYZ> &pc,
                      const pcl::PointCloud<pcl::PointXYZ> &normals,
                      double range_limit = -1) {
    loadPointCloud(ToPCLXY(pc), ToPCLXY(normals), range_limit);
  }

  /**
   * 1. with guess: ctor -> loadPointCloud
   * 2. without guess: ctor -> setGuessSize -> loadPointCloud
   */
  void loadPointCloud(const pcl::PointCloud<pcl::PointXY> &pc,
                      const pcl::PointCloud<pcl::PointXY> &normals,
                      double range_limit = -1) {
    Expects(normals.points.size() == pc.points.size());

    if (index_ == NULL) {
      return;
    }

    if (!isFirstLoad_) {
      LazyGrid2D *si = new LazyGrid2D(index_->getCellSize());
      if (si == NULL) {
        return;
      }
      delete index_;
      index_ = si;
      isFirstLoad_ = false;
    }

    if (guess_size_) {
      Eigen::Vector2d center;
      double mapsize;
      guessMapSize(pc, range_limit, center, mapsize);
      centerx = center(0);
      centery = center(1);
      index_->setCenter(centerx, centery);
      index_->setSize(mapsize, mapsize);
    } else {
      dprintf("Set Map Size: center(%.2f, %.2f) & mapsize (%.2f x %.2f)\n",
              centerx, centery, map_sizex, map_sizey);
      index_->setCenter(centerx, centery);
      if (map_sizex > 0 && map_sizey > 0) {
        index_->setSize(map_sizex, map_sizey);
      } else {
        dprintf("mapsize should always >= 0\n");
        exit(1);
      }
    }

    for (size_t i = 0; i < pc.points.size(); ++i) {
      pcl::PointXY pt = pc.points.at(i);
      if (isnan(pt.x) || isnan(pt.y)) {
        dprintf("exist NaN!!\n");
        exit(1);
      }
      if (range_limit > 0) {
        Eigen::Vector2d d(pt.x, pt.y);
        if (d.norm() > range_limit) {
          continue;
        }
      }
      NDTCell *cell = index_->addPoint(pt);
      cell->addNormal(normals.points.at(i));
      // TODO: add covariance
      // index_->getCellForPoint(pt)->addPCov(pcovs.at(i));
      NDTCell *ptCell;
      index_->getCellAt(pt, ptCell);
      if (ptCell != NULL) {
        update_set.insert(ptCell);
      }
    }
    isFirstLoad_ = false;
  }

  void computeNDTCells() {
    for (const auto &cell : update_set) {
      if (cell != NULL) {
        cell->computeGaussian();
      }
    }
    update_set.clear();
  }

  inline LazyGrid2D *getMyIndex() const { return index_; }

  // double getLikelihoodForPoint(pcl::PointXY pt);

  bool getCellAtPoint(const pcl::PointXY &refPoint, NDTCell *&cell) {
    index_->getCellAt(refPoint, cell);
    return (cell != NULL);
  }

  bool getCellAtPoint(const pcl::PointXY &refPoint, NDTCell *&cell) const {
    index_->getCellAt(refPoint, cell);
    return (cell != NULL);
  }

  bool getCellAtAllocate(const pcl::PointXY &refPoint, NDTCell *&cell) {
    index_->getCellAtAllocate(refPoint, cell);
    return (cell != NULL);
  }

  bool getCellAtAllocate(const pcl::PointXY &refPoint, NDTCell *&cell) const {
    index_->getCellAtAllocate(refPoint, cell);
    return (cell != NULL);
  }

  bool getCellForPoint(const pcl::PointXY &pt, NDTCell *&cell,
                       int max_neighbors, bool checkForGaussian) const {
    cell = index_->getClosestNDTCell(pt, max_neighbors, checkForGaussian);
    return (cell != NULL);
  }

  vector<NDTCell *> getCellsForPoint(const pcl::PointXY pt, int n_neighbors,
                                     bool checkForGaussian) const {
    return index_->getClosestNDTCells(pt, n_neighbors, checkForGaussian);
  }

  vector<NDTCell *> getInitializedCellsForPoint(const pcl::PointXY pt) const {
    return index_->getClosestCells(pt);
  }

  NDTMap *pseudoTransformNDTMap(const Eigen::Matrix3d &T) {
    NDTMap *map = new NDTMap(new LazyGrid2D(index_->getCellSize()));
    typename LazyGrid2D::CellVectorConstItr it = index_->begin();
    auto idx = map->getMyIndex();
    while (it != index_->end()) {
      if (*it != NULL) {
        if (((*it)->phasGaussian_) && ((*it)->nhasGaussian_)) {
          NDTCell *nd = new NDTCell();
          Eigen::Matrix2d R = T.block<2, 2>(0, 0);
          Eigen::Vector2d t = T.block<2, 1>(0, 2);
          Eigen::Vector2d mean = (*it)->getPointMean();
          Eigen::Matrix2d cov = (*it)->getPointCov();
          mean = R * mean + t;
          cov = R * cov * R.transpose();
          double xs, ys;
          (*it)->getDimensions(xs, ys);
          nd->setDimensions(xs, ys);
          nd->setCenter((*it)->getCenter());
          nd->setN((*it)->getN());
          nd->phasGaussian_ = true;
          nd->setPointMean(mean);
          nd->setPointCov(cov);

          mean = (*it)->getNormalMean();
          cov = (*it)->getNormalCov();
          mean = R * mean;
          cov = R * cov * R.transpose();
          nd->nhasGaussian_ = true;
          nd->setNormalMean(mean);
          nd->setNormalCov(cov);
          idx->addNDTCell(nd);
          nd->ToString();
        }
      }
      ++it;
    }
    return map;
  }

  vector<NDTCell *> pseudoTransformNDT(const Eigen::Matrix3d &T) const {
    vector<NDTCell *> ret;
    typename LazyGrid2D::CellVectorConstItr it = index_->begin();
    while (it != index_->end()) {
      if (*it != NULL) {
        if ((*it)->phasGaussian_ && (*it)->nhasGaussian_) {
          NDTCell *nd = new NDTCell();
          Eigen::Matrix2d R = T.block<2, 2>(0, 0);
          Eigen::Vector2d t = T.block<2, 1>(0, 2);
          Eigen::Vector2d mean = (*it)->getPointMean();
          Eigen::Matrix2d cov = (*it)->getPointCov();
          mean = R * mean + t;
          cov = R * cov * R.transpose();
          double xs, ys;
          (*it)->getDimensions(xs, ys);
          nd->setDimensions(xs, ys);
          nd->setCenter((*it)->getCenter());
          nd->setN((*it)->getN());
          nd->phasGaussian_ = true;
          nd->setPointMean(mean);
          nd->setPointCov(cov);
          
          mean = (*it)->getNormalMean();
          cov = (*it)->getNormalCov();
          mean = R * mean;
          cov = R * cov * R.transpose();
          nd->nhasGaussian_ = true;
          nd->setNormalMean(mean);
          nd->setNormalCov(cov);
          ret.push_back(nd);
        }
      }
      ++it;
    }
    return ret;
  }

  vector<boost::shared_ptr<NDTCell>> getAllCells() const {
    vector<boost::shared_ptr<NDTCell>> ret;
    for (auto cell = index_->begin(); cell != index_->end(); ++cell) {
      if ((*cell)->phasGaussian_) {
        boost::shared_ptr<NDTCell> sp(*cell);
        ret.push_back(sp);
      }
    }
    return ret;
  }

  int numberOfActiveCells() {
    int ret = 0;
    if (index_ == NULL) return ret;
    for (auto it = index_->begin(); it != index_->end(); ++it) {
      if ((*it)->phasGaussian_) {
        ++ret;
      }
    }
    return ret;
  }

  int numberOfActiveCells() const {
    int ret = 0;
    if (index_ == NULL) return ret;
    for (auto it = index_->begin(); it != index_->end(); ++it) {
      if ((*it)->phasGaussian_) {
        ++ret;
      }
    }
    return ret;
  }

  bool getCentroid(double &cx, double &cy) {
    if (index_ == NULL) return false;
    index_->getCenter(cx, cy);
    return true;
  }

  bool getGridSize(int &cx, int &cy) {
    if (index_ == NULL) return false;
    index_->getGridSize(cx, cy);
    return true;
  }

  bool getGridSizeInMeters(double &cx, double &cy) {
    if (index_ == NULL) return false;
    index_->getGridSizeInMeters(cx, cy);
    return true;
  }

  bool getGridSizeInMeters(double &cx, double &cy) const {
    if (index_ == NULL) return false;
    index_->getGridSizeInMeters(cx, cy);
    return true;
  }

  bool getCellSizeInMeters(double &cx, double &cy) {
    if (index_ == NULL) return false;
    index_->getCellSize(cx, cy);
    return true;
  }

  bool getCellSizeInMeters(double &cx, double &cy) const {
    if (index_ == NULL) return false;
    index_->getCellSize(cx, cy);
    return true;
  }

  double getSmallestCellSizeInMeters() const {
    double cx, cy;
    if (index_ == NULL) return false;
    index_->getCellSize(cx, cy);
    return min(cx, cy);
  }

  void setGuessSize(double cenx, double ceny, double sizex, double sizey) {
    guess_size_ = false;
    centerx = cenx;
    centery = ceny;
    map_sizex = sizex;
    map_sizey = sizey;
  }

  NDTCell *getCellAtID(int x, int y) const {
    NDTCell *cell = NULL;
    index_->getCellAt(x, y, cell);
    return cell;
  }

  void ToString() {
    index_->ToString();
  }

 private:
  LazyGrid2D *index_;
  bool isFirstLoad_;
  double map_sizex;
  double map_sizey;
  double centerx, centery;
  bool guess_size_;
  set<NDTCell *> update_set;
  int radius_;

  void guessMapSize(const pcl::PointCloud<pcl::PointXY> &pc, double range_limit,
                    Eigen::Vector2d &center, double &mapsize) {
    int npts = 0;
    center = Eigen::Vector2d::Zero();
    for (const auto &pt : pc.points) {
      if (isnan(pt.x) || isnan(pt.y)) continue;
      Eigen::Vector2d d(pt.x, pt.y);
      if (range_limit > 0 && d.norm() > range_limit) continue;
      center += d;
      ++npts;
    }
    center /= npts;

    double maxDist = 0;
    for (const auto &pt : pc.points) {
      if (isnan(pt.x) || isnan(pt.y)) continue;
      Eigen::Vector2d d(pt.x, pt.y);
      if (range_limit > 0 && d.norm() > range_limit) continue;
      double dist = (d - center).norm();
      maxDist = (dist > maxDist) ? dist : maxDist;
    }
    mapsize = maxDist * 4;

    dprintf("Guess Map Size: center(%.2f, %.2f) & mapsize (%.2f x %.2f)\n",
            center(0), center(1), mapsize, mapsize);
  }

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
