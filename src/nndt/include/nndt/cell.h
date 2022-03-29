#pragma once
#include <bits/stdc++.h>

#include <Eigen/Dense>

class Cell {
 public:
  // FIXME: replace with not enough points
  enum CellType {
    kNoInit,    /**< Covariance is not computed yet */
    kFewPoints, /**< Covariance is not computed because of few points */
    kLine,      /**< Covariance degenerates as a line */
    kPlane,     /**< Covariance degenerates as a plane */
    kRegular    /**< Covariance is computed well */
  };

  Cell();

  void AddPoint(const Eigen::Vector3d &point);

  void AddPointWithCovariance(const Eigen::Vector3d &point,
                              const Eigen::Matrix3d &covariance);

  void ComputeGaussian();

  int GetN() const { return n_; }
  bool GetHasGaussian() const { return hasgaussian_; }
  double GetSize() const { return size_; }
  Eigen::Vector3d GetCenter() const { return center_; }
  Eigen::Vector3d GetMean() const { return mean_; }
  Eigen::Matrix3d GetCov() const { return cov_; }
  Eigen::Vector3d GetEvals() const { return evals_; }
  Eigen::Matrix3d GetEvecs() const { return evecs_; }
  Eigen::Vector3d GetNormal() const { return normal_; }
  std::vector<Eigen::Vector3d> GetPoints() const { return points_; }
  std::vector<Eigen::Matrix3d> GetPointCovs() const { return point_covs_; }
  CellType GetCellType() const { return celltype_; }
  double GetRescaleRatio() const { return rescale_ratio_; }

  void SetN(int n) { n_ = n; }
  void SetHasGaussian(bool hasgaussian) { hasgaussian_ = hasgaussian; }
  void SetSize(double size) { size_ = size; }
  void SetCenter(const Eigen::Vector3d &center) { center_ = center; }
  void SetMean(const Eigen::Vector3d &mean) { mean_ = mean; }
  void SetCov(const Eigen::Matrix3d &cov) { cov_ = cov; }
  void SetEvals(const Eigen::Vector3d &evals) { evals_ = evals; }
  void SetEvecs(const Eigen::Matrix3d &evecs) { evecs_ = evecs; }
  void SetNormal(const Eigen::Vector3d &normal) { normal_ = normal; }
  void SetPoints(const std::vector<Eigen::Vector3d> &points) {
    points_ = points;
  }
  void SetPointCovs(const std::vector<Eigen::Matrix3d> &point_covs) {
    point_covs_ = point_covs;
  }
  void SetCellType(CellType celltype) { celltype_ = celltype; }
  void SetRescaleRatio(double rescale_ratio) { rescale_ratio_ = rescale_ratio; }

 private:
  int n_;                                   /**< Number of points */
  bool hasgaussian_;                        /**< Whether cell has a gaussian */
  double size_;                             /**< Cell size */
  Eigen::Vector3d center_;                  /**< Center of the cell */
  Eigen::Vector3d mean_;                    /**< Point mean */
  Eigen::Matrix3d cov_;                     /**< Point covariance */
  Eigen::Vector3d evals_;                   /**< Point eigenvalues */
  Eigen::Matrix3d evecs_;                   /**< Point eigenvectors */
  Eigen::Vector3d normal_;                  /**< Point Normal */
  std::vector<Eigen::Vector3d> points_;     /**< Points */
  std::vector<Eigen::Matrix3d> point_covs_; /**< Point covariances */
  CellType celltype_;                       /**< Cell type */
  double rescale_ratio_;                    /**< Rescale ratio */
};
