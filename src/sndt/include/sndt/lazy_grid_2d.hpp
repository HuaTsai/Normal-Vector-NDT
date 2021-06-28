#pragma once

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <gsl/gsl>
#include "sndt/ndt_cell_2d.hpp"

using namespace std;
using namespace Eigen;

class LazyGrid2D {
 public:
  explicit LazyGrid2D(double cell_size) {
    has_grid_center_ = has_cellptrs_size_ = is_initialized_ = false;
    cell_size_ << cell_size, cell_size;
    grid_center_.setZero();
    grid_size_.setZero();
    grid_center_index_.setZero();
    cellptrs_size_.setZero();
  }

  ~LazyGrid2D() {
    if (is_initialized_) {
      for (auto &cell : active_cells_)
        if (cell)
          delete cell;
      for (int i = 0; i < cellptrs_size_(0); ++i)
        delete[] cellptrs_[i];
      delete[] cellptrs_;
    }
  }

  void SetGridCenter(const Vector2d &grid_center) {
    grid_center_ = grid_center;
    has_grid_center_ = true;
    if (has_cellptrs_size_)
      Initialize();
  }

  void SetGridSize(const Vector2d &grid_size) {
    grid_size_ = grid_size;
    cellptrs_size_(0) = int(ceil(grid_size_(0) / cell_size_(0)));
    cellptrs_size_(1) = int(ceil(grid_size_(1) / cell_size_(1)));
    grid_center_index_(0) = cellptrs_size_(0) / 2;
    grid_center_index_(1) = cellptrs_size_(1) / 2;
    has_cellptrs_size_ = true;
    if (has_grid_center_)
      Initialize();
  }

  NDTCell *GetCellForPoint(const Vector2d &point) {
    Expects(is_initialized_);
    auto idx = GetIndexForPoint(point);
    if (!IsValidIndex(idx)) { return nullptr; }
    return cellptrs_[idx(0)][idx(1)];
  }

  NDTCell *AddPoint(const Vector2d &point) {
    Expects(is_initialized_);
    NDTCell *cell = GetCellAndAllocate(point);
    if (cell)
      cell->AddPoint(point);
    return cell;
  }

  NDTCell *AddPointAndNormal(const Vector2d &point, const Vector2d &normal) {
    Expects(is_initialized_);
    NDTCell *cell = GetCellAndAllocate(point);
    if (cell) {
      cell->AddPoint(point);
      cell->AddNormal(normal);
    }
    return cell;
  }

  NDTCell *AddPointAndNormalWithCovariance(const Vector2d &point,
                                           const Matrix2d &point_cov,
                                           const Vector2d &normal) {
    Expects(is_initialized_);
    NDTCell *cell = GetCellAndAllocate(point);
    if (cell) {
      cell->AddPointWithCovariance(point, point_cov);
      cell->AddNormal(normal);
    }
    return cell;
  }

  bool InsertCell(const NDTCell &cell) {
    if (!is_initialized_) { return false; }
    auto idx = GetIndexForPoint(cell.GetCenter());
    if (!IsValidIndex(idx)) { return false; }
    if (!cellptrs_[idx(0)][idx(1)]) {
      cellptrs_[idx(0)][idx(1)] = new NDTCell();
      active_cells_.push_back(cellptrs_[idx(0)][idx(1)]);
    }
    NDTCell *ret = cellptrs_[idx(0)][idx(1)];
    ret->SetN(cell.GetN());
    ret->SetPHasGaussian(cell.GetPHasGaussian());
    ret->SetNHasGaussian(cell.GetNHasGaussian());
    ret->SetCenter(cell.GetCenter());
    ret->SetSize(cell.GetSize());
    ret->SetPointMean(cell.GetPointMean());
    ret->SetPointCov(cell.GetPointCov());
    ret->SetPointEvals(cell.GetPointEvals());
    ret->SetPointEvecs(cell.GetPointEvecs());
    ret->SetNormalMean(cell.GetNormalMean());
    ret->SetNormalCov(cell.GetNormalCov());
    ret->SetNormalEvals(cell.GetNormalEvals());
    ret->SetNormalEvecs(cell.GetNormalEvecs());
    return true;
  }

  NDTCell *GetClosestCellForPointByRadius(const Vector2d &point,
                                          double radius,
                                          bool include_locate = true) {
    int maxdist = (int)ceil(radius / cell_size_(0));
    return GetClosestCellForPoint(point, maxdist, include_locate);
  }

  NDTCell *GetClosestCellForPoint(const Vector2d &point,
                                  int maxdist_of_cells,
                                  bool include_locate = true) {
    auto cells = GetClosestCellsForPoint(point, maxdist_of_cells, include_locate);
    if (!cells.size()) { return nullptr; }
    return *min_element(cells.begin(), cells.end(), [&](auto p, auto q) {
      return (p->GetPointMean() - point).norm() < (q->GetPointMean() - point).norm();
    });
  }

  vector<NDTCell *> GetClosestCellsForPointByRadius(
      const Vector2d &point, double radius, bool include_locate = false) {
    int maxdist = (int)ceil(radius / cell_size_(0));
    return GetClosestCellsForPoint(point, maxdist, include_locate);
  }

  vector<NDTCell *> GetClosestCellsForPoint(const Vector2d &point,
                                            int maxdist_of_cells,
                                            bool include_locate = false) {
    auto idx = GetIndexForPoint(point);
    vector<NDTCell *> cells;
    if (include_locate && IsValidIndex(idx) && cellptrs_[idx(0)][idx(1)] &&
        cellptrs_[idx(0)][idx(1)]->BothHasGaussian()) {
      cells.push_back(cellptrs_[idx(0)][idx(1)]);
    }
    Vector2i xy;
    for (int x = 1; x < 2 * maxdist_of_cells + 2; ++x) {
      xy(0) = (x % 2) ? idx(0) - x / 2 : idx(0) + x / 2;
      for (int y = 1; y < 2 * maxdist_of_cells + 2; ++y) {
        xy(1) = (y % 2) ? idx(1) - y / 2 : idx(1) + y / 2;
        if (IsValidIndex(xy) && cellptrs_[xy(0)][xy(1)] &&
            cellptrs_[xy(0)][xy(1)]->BothHasGaussian()) {
          cells.push_back(cellptrs_[xy(0)][xy(1)]);
        }
      }
    }
    return cells;
  }

  NDTCell *GetCellAndAllocate(const Vector2d &point) {
    Expects(is_initialized_);
    if (isnan(point(0)) || isnan(point(1))) { return nullptr; }
    auto idx = GetIndexForPoint(point);
    if (!IsValidIndex(idx)) { return nullptr; }
    if (!cellptrs_[idx(0)][idx(1)]) {
      cellptrs_[idx(0)][idx(1)] = new NDTCell();
      Vector2d center;
      center(0) = grid_center_(0) + (idx(0) - grid_center_index_(0)) * cell_size_(0);
      center(1) = grid_center_(1) + (idx(1) - grid_center_index_(1)) * cell_size_(1);
      cellptrs_[idx(0)][idx(1)]->SetCenter(center);
      cellptrs_[idx(0)][idx(1)]->SetSize(cell_size_);
      active_cells_.push_back(cellptrs_[idx(0)][idx(1)]);
    }
    return cellptrs_[idx(0)][idx(1)];
  }

  Vector2i GetIndexForPoint(const Vector2d &point) {
    Vector2i ret;
    ret(0) = floor((point(0) - grid_center_(0)) / cell_size_(0) + 0.5) + grid_center_index_(0);
    ret(1) = floor((point(1) - grid_center_(1)) / cell_size_(1) + 0.5) + grid_center_index_(1);
    return ret;
  }

  string ToString() {
    if (!is_initialized_) { return "LazyGrid uninitialized"; }
    stringstream ss;
    ss.setf(ios::fixed | ios::boolalpha);
    ss.precision(2);
    ss << "cell size: (" << cell_size_(0) << ", " << cell_size_(1) << ")" << endl
       << "grid center: (" << grid_center_(0) << ", " << grid_center_(1) << ")" << endl
       << "grid size: (" << grid_size_(0) << ", " << grid_size_(1) << ")" << endl
       << "cellptrs size: (" << cellptrs_size_(0) << ", " << cellptrs_size_(1) << ")" << endl
       << "center index: (" << grid_center_index_(0) << ", " << grid_center_index_(1) << ")" << endl
       << "active cells: " << active_cells_.size() << endl;
    for (size_t i = 0; i < active_cells_.size(); ++i) {
      auto idx = GetIndexForPoint(active_cells_.at(i)->GetCenter());
      ss << "[#" << i << ", (" << idx(0) << ", " << idx(1) << ")] "
         << active_cells_.at(i)->ToString();
    }
    return ss.str();
  }

  // Size and Iterators
  int size() { return active_cells_.size(); }
  vector<NDTCell *>::iterator begin() { return active_cells_.begin(); }
  vector<NDTCell *>::const_iterator begin() const { return active_cells_.begin(); }
  vector<NDTCell *>::iterator end() { return active_cells_.end(); }
  vector<NDTCell *>::const_iterator end() const { return active_cells_.end(); }

  // Get and Set methods
  double GetCellSize() const { return cell_size_(0); }
  NDTCell ***cellptrs() const { return cellptrs_; }
  Vector2d cell_size() const { return cell_size_; }
  Vector2d grid_size() const { return grid_size_; }
  Vector2d grid_center() const { return grid_center_; }
  Vector2i cellptrs_size() const { return cellptrs_size_; }

 private:
  bool is_initialized_, has_cellptrs_size_, has_grid_center_;
  NDTCell ***cellptrs_;
  vector<NDTCell *> active_cells_;
  Vector2d cell_size_;
  Vector2d grid_size_;
  Vector2d grid_center_;
  Vector2i grid_center_index_;
  Vector2i cellptrs_size_;

  void Initialize() {
    Expects(!is_initialized_);
    cellptrs_ = new NDTCell **[cellptrs_size_(0)];
    for (int i = 0; i < cellptrs_size_(0); ++i) {
      cellptrs_[i] = new NDTCell *[cellptrs_size_(1)];
      memset(cellptrs_[i], 0, cellptrs_size_(1) * sizeof(NDTCell *));
    }
    is_initialized_ = true;
  }

  bool IsValidIndex(const Vector2i &index) {
    return index(0) >= 0 && index(1) < cellptrs_size_(0) &&
           index(1) >= 0 && index(1) < cellptrs_size_(1);
  }
};
