#pragma once

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <gsl/gsl>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include "sndt/lazy_grid_2d.hpp"
#include "normal2d/Normal2dEstimation.h"

using namespace std;
using namespace Eigen;
using pcl::search::KdTree;
using pcl::PointXYZ;
using pcl::PointXY;
typedef pcl::PointCloud<pcl::PointXYZ> PCXYZ;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PCXYZPtr;
typedef pcl::PointCloud<pcl::PointXY> PCXY;
typedef pcl::PointCloud<pcl::PointXY>::Ptr PCXYPtr;

PCXY ToPCLXY(const PCXYZ &pc) {
  PCXY ret;
  std::transform(pc.begin(), pc.end(), back_inserter(ret), [](auto p) {
    PointXY pt;
    pt.x = p.x;
    pt.y = p.y;
    return pt;
  });
  return ret;
}

PCXYZPtr ComputeNormals(PCXYZPtr pc, double radius) {
  PCXYZPtr ret(new PCXYZ);
  KdTree<PointXYZ>::Ptr tree(new KdTree<PointXYZ>);
  Normal2dEstimation ne;
  ne.setInputCloud(pc);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(radius);
  ne.compute(ret);
  return ret;
}

class NDTMap {
 public:
  explicit NDTMap(double cell_size) {
    index_ = new LazyGrid2D(cell_size);
    cell_size_ << cell_size, cell_size;
    map_center_.setZero();
    map_size_.setZero();
    guess_map_size_ = true;
    is_initialized_ = false;
  }

  ~NDTMap() {
    delete index_;
  }

  // Three main functions
  // TODO: NOT IMPLEMENT YET!!!
  void AddPointCloud(const PCXYZ &pc) { ; }

  void LoadPointCloud(const PCXYZ &pc, const PCXYZ &normals,
                      double range_limit = -1) {
    LoadPointCloud(ToPCLXY(pc), ToPCLXY(normals), range_limit);
  }

  void LoadPointCloud(const PCXY &pc, const PCXY &normals,
                      double range_limit = -1) {
    Expects(pc.size() == normals.size());
    if (is_initialized_) {
      LazyGrid2D *idx = new LazyGrid2D(cell_size_(0));
      delete index_;
      index_ = idx;
    }
    if (guess_map_size_) {
      GuessMapSize(pc, range_limit);
    }
    index_->SetGridCenter(map_center_);
    index_->SetGridSize(map_size_);

    vector<NDTCell *> update_cells;
    for (size_t i = 1; i < pc.size(); ++i) {
      Vector2d point(pc.at(i).x, pc.at(i).y);
      Vector2d normal(normals.at(i).x, normals.at(i).y);
      if (range_limit > 0 && point.norm() > range_limit)
        continue;
      NDTCell *cell = index_->AddPointAndNormal(point, normal);
      if (cell)
        update_cells.push_back(cell);
    }
    for (auto cell : update_cells)
      cell->ComputeGaussian();
    update_cells.clear();
    is_initialized_ = true;
  }

  vector<std::shared_ptr<NDTCell>> PseudoTransformCells(const Matrix3d &T) {
    vector<std::shared_ptr<NDTCell>> ret;
    Matrix2d R = T.block<2, 2>(0, 0);
    Vector2d t = T.block<2, 1>(0, 2);
    for (auto it = begin(); it != end(); ++it) {
      if ((*it)->BothHasGaussian()) {
        auto cell = std::make_shared<NDTCell>();
        cell->SetN((*it)->GetN());
        cell->SetPHasGaussian(true);
        cell->SetNHasGaussian(true);
        cell->SetSize((*it)->GetSize());
        cell->SetCenter((*it)->GetCenter());
        cell->SetN((*it)->GetN());
        cell->SetPointMean(R * (*it)->GetPointMean() + t);
        cell->SetPointCov(R * (*it)->GetPointCov() * R.transpose());
        cell->SetNormalMean(R * (*it)->GetNormalMean() + t);
        cell->SetNormalCov(R * (*it)->GetNormalCov() * R.transpose());
        ret.push_back(cell);
      }
    }
    return ret;
  }

  // Inherited methods
  NDTCell *GetCellForPoint(const Vector2d &point) {
    return index_->GetCellForPoint(point);
  }

  NDTCell *GetClosestCellForPoint(const Vector2d &point, int maxdist_of_cells, bool include_locate = true) {
    return index_->GetClosestCellForPoint(point, maxdist_of_cells, include_locate);
  }

  vector<NDTCell *> GetClosestCellsForPoint(const Vector2d &point, int maxdist_of_cells, bool include_locate = false) {
    return index_->GetClosestCellsForPoint(point, maxdist_of_cells, include_locate);
  }

  string ToString() {
    return index_->ToString();
  }

  // Size and Iterators
  int size() { return index_->size(); }
  vector<NDTCell *>::iterator begin() { return index_->begin(); }
  vector<NDTCell *>::const_iterator begin() const { return index_->begin(); }
  vector<NDTCell *>::iterator end() { return index_->end(); }
  vector<NDTCell *>::const_iterator end() const { return index_->end(); } 

  // Variables
  void SetGuessMapSize(const Vector2d &map_center, const Vector2d &map_size) {
    guess_map_size_ = false;
    map_center_ = map_center;
    map_size_ = map_size;
  }
  LazyGrid2D *index() const { return index_; }
  Vector2d map_size() const { return map_size_; }
  Vector2d map_center() const { return map_center_; }
  Vector2d cell_size() const { return cell_size_; }

 private:
  LazyGrid2D *index_;
  bool is_initialized_;
  bool guess_map_size_;
  Vector2d map_size_;
  Vector2d map_center_;
  Vector2d cell_size_;

  void GuessMapSize(const PCXY &pc, double range_limit = -1) {
    int n = 0;
    map_center_.setZero();
    for (const auto &pt : pc) {
      if (isnan(pt.x) || isnan(pt.y)) continue;
      Vector2d d(pt.x, pt.y);
      if (range_limit > 0 && d.norm() > range_limit) continue;
      map_center_ += d;
      ++n;
    }
    map_center_ /= n;

    double maxdist = 0;
    for (const auto &pt : pc) {
      if (isnan(pt.x) || isnan(pt.y)) continue;
      Vector2d d(pt.x, pt.y);
      if (range_limit > 0 && d.norm() > range_limit) continue;
      maxdist = max(maxdist, (d - map_center_).norm());
    }
    map_size_ << maxdist * 4, maxdist * 4;
  }
};
