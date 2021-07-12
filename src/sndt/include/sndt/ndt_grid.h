#pragma once
#include <bits/stdc++.h>

#include <Eigen/Dense>

#include "sndt/ndt_cell.h"

using namespace std;
using namespace Eigen;

class NDTGrid {
 public:
  explicit NDTGrid(double cell_size);

  ~NDTGrid();

  // Set center, then initialize if size is also known
  void SetGridCenter(const Vector2d &grid_center);

  // Set size, then initialize if center is also known
  void SetGridSize(const Vector2d &grid_size);

  Vector2i GetIndexForPoint(const Vector2d &point);

  NDTCell *GetCellForPoint(const Vector2d &point);

  NDTCell *GetCellAndAllocate(const Vector2d &point);

  NDTCell *AddPoint(const Vector2d &point);

  NDTCell *AddPointAndNormal(const Vector2d &point, const Vector2d &normal);

  NDTCell *AddPointAndNormalWithCovariance(const Vector2d &point,
                                           const Matrix2d &point_cov,
                                           const Vector2d &normal);

  NDTCell *GetClosestCellForPointByRadius(const Vector2d &point, double radius,
                                          bool include_locate = true);

  NDTCell *GetClosestCellForPoint(const Vector2d &point, int maxdist_of_cells,
                                  bool include_locate = true);

  vector<NDTCell *> GetClosestCellsForPointByRadius(
      const Vector2d &point, double radius, bool include_locate = false);

  vector<NDTCell *> GetClosestCellsForPoint(const Vector2d &point,
                                            int maxdist_of_cells,
                                            bool include_locate = false);

  string ToString();

  // Defined Size and Iterators
  int size() { return active_cells_.size(); }
  vector<NDTCell *>::iterator begin() { return active_cells_.begin(); }
  vector<NDTCell *>::const_iterator begin() const { return active_cells_.begin(); }
  vector<NDTCell *>::iterator end() { return active_cells_.end(); }
  vector<NDTCell *>::const_iterator end() const { return active_cells_.end(); }

  // Defined Get methods
  double GetCellSize() const { return cell_size_; }
  Vector2d GetGridSize() const { return grid_size_; }
  Vector2d GetGridCenter() const { return grid_center_; }

 private:
  void Initialize();
  bool IsValidIndex(const Vector2i &index);

  bool is_initialized_, has_cellptrs_size_, has_grid_center_;
  NDTCell ***cellptrs_;
  vector<NDTCell *> active_cells_;
  double cell_size_;
  Vector2d grid_size_;
  Vector2d grid_center_;
  Vector2i grid_center_index_;
  Vector2i cellptrs_size_;
};
