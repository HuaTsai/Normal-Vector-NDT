#pragma once
#include <bits/stdc++.h>

#include <Eigen/Dense>

class Cell {
 public:
  enum CellType {
    kNoInit,   /**< Covariance is not computed yet */
    kNoPoints, /**< Covariance is not computed because of no points */
    kRegular,  /**< Covariance is computed well */
    kRescale,  /**< Covariance is rescaled */
    kInvalid   /**< Covariance is invalid */
  };

  Cell();

  void AddPoint(const Eigen::Vector2d &point);

  void AddPointWithCovariance(const Eigen::Vector2d &point,
                              const Eigen::Matrix2d &covariance);

  void ComputeGaussian();

  int GetN() const { return n_; }
  bool GetHasGaussian() const { return hasgaussian_; }
  double GetSkewRad() const { return skew_rad_; }
  double GetSize() const { return size_; }
  Eigen::Vector3d GetCenter() const { return center_; }
  Eigen::Vector3d GetPointMean() const { return mean_; }
  Eigen::Matrix3d GetPointCov() const { return cov_; }
  Eigen::Vector3d GetPointEvals() const { return evals_; }
  Eigen::Matrix3d GetPointEvecs() const { return evecs_; }
  std::vector<Eigen::Vector3d> GetPoints() const { return points_; }
  std::vector<Eigen::Matrix3d> GetPointCovs() const { return point_covs_; }
  CellType GetCellType() const { return celltype_; }
  double GetRescaleRatio() const { return rescale_ratio_; }
  double GetTolerance() const { return tolerance_; }

  void SetN(int n) { n_ = n; }
  void SetPHasGaussian(bool hasgaussian) { hasgaussian_ = hasgaussian; }
  void SetSkewRad(double skew_rad) { skew_rad_ = skew_rad; }
  void SetSize(double size) { size_ = size; }
  void SetCenter(const Eigen::Vector3d &center) { center_ = center; }
  void SetPointMean(const Eigen::Vector3d &mean) { mean_ = mean; }
  void SetPointCov(const Eigen::Matrix3d &cov) { cov_ = cov; }
  void SetPointEvals(const Eigen::Vector3d &evals) { evals_ = evals; }
  void SetPointEvecs(const Eigen::Matrix3d &evecs) { evecs_ = evecs; }
  void SetPoints(const std::vector<Eigen::Vector3d> &points) {
    points_ = points;
  }
  void SetPointCovs(const std::vector<Eigen::Matrix3d> &point_covs) {
    point_covs_ = point_covs;
  }
  void SetCellType(CellType celltype) { celltype_ = celltype; }
  void SetRescaleRatio(double rescale_ratio) { rescale_ratio_ = rescale_ratio; }
  void SetTolerance(double tolerance) { tolerance_ = tolerance; }

 private:
  int n_;                                   /**< Number of points */
  bool hasgaussian_;                        /**< Whether cell has a gaussian */
  double skew_rad_;                         /**< Tilted angle of the cell */
  double size_;                             /**< Cell size */
  Eigen::Vector3d center_;                  /**< Center of the cell */
  Eigen::Vector3d mean_;                    /**< Point mean */
  Eigen::Matrix3d cov_;                     /**< Point covariance */
  Eigen::Vector3d evals_;                   /**< Point eigenvalues */
  Eigen::Matrix3d evecs_;                   /**< Point eigenvectors */
  std::vector<Eigen::Vector3d> points_;     /**< Points */
  std::vector<Eigen::Matrix3d> point_covs_; /**< Point covariances */
  CellType celltype_;                       /**< Cell type */
  double rescale_ratio_;                    /**< Rescale ratio */
  double tolerance_;                        /**< Comparison tolerance */
};
