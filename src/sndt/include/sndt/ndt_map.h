#pragma once

#include <bits/stdc++.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <Eigen/Dense>

#include "sndt/ndt_grid.h"

using namespace std;
using namespace Eigen;

typedef pcl::PointCloud<pcl::PointXYZ> PCXYZ;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PCXYZPtr;
typedef pcl::PointCloud<pcl::PointXY> PCXY;
typedef pcl::PointCloud<pcl::PointXY>::Ptr PCXYPtr;

PCXYZPtr ComputeNormals(PCXYZPtr pc, double radius);

MatrixXd ComputeNormals2(const MatrixXd &pc, double radius);

class NDTMap {
 public:
  explicit NDTMap(double cell_size);

  ~NDTMap();

  // Three main functions
  void LoadPointCloud(PCXYZPtr pc, double radius, double range_limit = -1);

  void LoadPointCloud(const PCXYZ &pc, const PCXYZ &normals,
                      double range_limit = -1);

  void LoadPointCloud(const PCXY &pc, const PCXY &normals,
                      double range_limit = -1);

  // TODO
  // each batch of points has the same sensor origin
  // get r, theta -> compute cov by sensor origin
  // then points, point_covs get
  void LoadPointCloudWithCovariances(const MatrixXd &points,
                                     const MatrixXd &normals,
                                     const MatrixXd &point_covs,
                                     double range_limit = -1);

  vector<shared_ptr<NDTCell>> PseudoTransformCells(const Affine2d &T, bool include_data = false) const {
    return PseudoTransformCells(T.matrix(), include_data);
  }

  vector<shared_ptr<NDTCell>> PseudoTransformCells(const Matrix3d &T, bool include_data = false) const;

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

  vector<Vector2d> GetPoints();
  vector<Vector2d> GetNormals();

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
  NDTGrid *index() const { return index_; }
  Vector2d map_size() const { return map_size_; }
  Vector2d map_center() const { return map_center_; }
  double cell_size() const { return cell_size_; }

 private:
  NDTGrid *index_;
  bool is_initialized_;
  bool guess_map_size_;
  double cell_size_;
  Vector2d map_size_;
  Vector2d map_center_;

  void GuessMapSize(const PCXY &pc, double range_limit = -1);
  void GuessMapSize(const MatrixXd &pc, double range_limit = -1);
};
