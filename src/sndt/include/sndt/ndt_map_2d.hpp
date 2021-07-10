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
typedef pcl::PointCloud<pcl::PointXYZL> PCXYZL;
typedef pcl::PointCloud<pcl::PointXYZL>::Ptr PCXYZLPtr;
typedef pcl::PointCloud<pcl::PointXYZ> PCXYZ;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PCXYZPtr;
typedef pcl::PointCloud<pcl::PointXY> PCXY;
typedef pcl::PointCloud<pcl::PointXY>::Ptr PCXYPtr;

PCXY ToPCLXY(const PCXYZ &pc) {
  PCXY ret;
  transform(pc.begin(), pc.end(), back_inserter(ret), [](auto p) {
    PointXY pt;
    pt.x = p.x, pt.y = p.y;
    return pt;
  });
  return ret;
}

PCXYZ ToPCLXYZ(const PCXY &pc) {
  PCXYZ ret;
  transform(pc.begin(), pc.end(), back_inserter(ret), [](auto p) {
    PointXYZ pt;
    pt.x = p.x;
    pt.y = p.y;
    pt.z = 0;
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

MatrixXd ComputeNormals2(const MatrixXd &pc, double radius) {
  Expects(pc.rows() == 2);
  PCXYZPtr pclpc(new PCXYZ), normals(new PCXYZ);
  int n = pc.cols();
  for (int i = 0; i < n; ++i)
    pclpc->push_back(PointXYZ(pc(0, i), pc(1, i), 0));
  KdTree<PointXYZ>::Ptr tree(new KdTree<PointXYZ>);
  Normal2dEstimation ne;
  ne.setInputCloud(pclpc);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(radius);
  ne.compute(normals);
  Expects((int)normals->size() == n);
  MatrixXd ret(2, n);
  for (int i = 0; i < n; ++i)
    ret.block<2, 1>(0, i) = Vector2d((*normals)[i].x, (*normals)[i].y);
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

  void LoadPointCloud(PCXYZPtr pc, double radius, double range_limit = -1) {
    auto normals = ComputeNormals(pc, radius);
    LoadPointCloud(*pc, *normals, range_limit);
  }

  void LoadPointCloud(const PCXYZ &pc, const PCXYZ &normals,
                      double range_limit = -1) {
    LoadPointCloud(ToPCLXY(pc), ToPCLXY(normals), range_limit);
  }

  void LoadPointCloud(const PCXY &pc, const PCXY &normals,
                      double range_limit = -1) {
    Expects(pc.size() == normals.size());
    if (is_initialized_) {
      delete index_;
      index_ = new LazyGrid2D(cell_size_(0));
    }
    if (guess_map_size_)
      GuessMapSize(pc, range_limit);
    index_->SetGridCenter(map_center_);
    index_->SetGridSize(map_size_);

    vector<NDTCell *> update_cells;
    for (size_t i = 0; i < pc.size(); ++i) {
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

  // TODO
  // each batch of points has the same sensor origin
  // get r, theta -> compute cov by sensor origin
  // then points, point_covs get
  void LoadPointCloudWithCovariances(const MatrixXd &points,
                                     const MatrixXd &normals,
                                     const MatrixXd &point_covs,
                                     double range_limit = -1) {
    Expects(points.rows() == 2 &&
            normals.rows() == 2 &&
            point_covs.rows() == 2);
    Expects(points.cols() == normals.cols() &&
            points.cols() == point_covs.cols() / 2);

    if (is_initialized_) {
      delete index_;
      index_ = new LazyGrid2D(cell_size_(0));
    }
    if (guess_map_size_)
      GuessMapSize(points, range_limit);
    index_->SetGridCenter(map_center_);
    index_->SetGridSize(map_size_);

    vector<NDTCell *> update_cells;
    for (int i = 0; i < points.cols(); ++i) {
      Vector2d point = points.col(i);
      Vector2d normal = normals.col(i);
      Matrix2d point_cov = point_covs.block<2, 2>(0, 2 * i);
      if (range_limit > 0 && point.norm() > range_limit)
        continue;
      NDTCell *cell = index_->AddPointAndNormalWithCovariance(point, point_cov, normal);
      if (cell)
        update_cells.push_back(cell);
    }
    for (auto cell : update_cells)
      cell->ComputeGaussianWithCovariances();
    update_cells.clear();
    is_initialized_ = true;
  }

  vector<shared_ptr<NDTCell>> PseudoTransformCells(const Affine2d &T, bool include_data = false) {
    return PseudoTransformCells(T.matrix(), include_data);
  }

  // TODO: Complete the cell construction
  vector<shared_ptr<NDTCell>> PseudoTransformCells(const Matrix3d &T, bool include_data = false) {
    vector<shared_ptr<NDTCell>> ret;
    Matrix2d R = T.block<2, 2>(0, 0);
    Vector2d t = T.block<2, 1>(0, 2);
    double skew_rad = Rotation2Dd(R).angle();
    for (auto it = begin(); it != end(); ++it) {
      auto cell = make_shared<NDTCell>();
      cell->SetN((*it)->GetN());
      cell->SetPHasGaussian((*it)->GetPHasGaussian());
      cell->SetNHasGaussian((*it)->GetNHasGaussian());
      cell->SetSkewRad(skew_rad);
      cell->SetCenter(R * (*it)->GetCenter() + t);
      cell->SetSize((*it)->GetSize());
      cell->SetPointMean(R * (*it)->GetPointMean() + t);
      if ((*it)->GetPHasGaussian())
        cell->SetPointCov(R * (*it)->GetPointCov() * R.transpose());
      cell->SetNormalMean(R * (*it)->GetNormalMean());
      if ((*it)->GetNHasGaussian())
        cell->SetNormalCov(R * (*it)->GetNormalCov() * R.transpose());
      if (include_data) {
        for (auto pt : (*it)->GetPoints()) {
          if (pt.allFinite())
            pt = R * pt + t;
          cell->AddPoint(pt);
        }
        for (auto nm : (*it)->GetNormals()) {
          if (nm.allFinite())
            nm = R * nm;
          cell->AddNormal((nm(1) > 0) ? nm : -nm);
        }
      }
      ret.push_back(cell);
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
      if (!isfinite(pt.x) || !isfinite(pt.y)) continue;
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
  
  void GuessMapSize(const MatrixXd &pc, double range_limit = -1) {
    Expects(pc.rows() == 2);
    int n = 0;
    map_center_.setZero();
    for (int i = 0; i < pc.cols(); ++i) {
      Vector2d pt = pc.col(i);
      if (!pt.allFinite()) continue;
      if (range_limit > 0 && pt.norm() > range_limit) continue;
      map_center_ += pt;
      ++n;
    }
    map_center_ /= n;

    double maxdist = 0;
    for (int i = 0; i < pc.cols(); ++i) {
      auto pt = pc.block<2, 1>(0, i);
      if (!pt.allFinite()) continue;
      if (range_limit > 0 && pt.norm() > range_limit) continue;
      maxdist = max(maxdist, (pt - map_center_).norm());
    }
    map_size_ << maxdist * 4, maxdist * 4;
  }
};
