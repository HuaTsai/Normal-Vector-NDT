#pragma once
#include <bits/stdc++.h>

#include <Eigen/Dense>

class Cell2D {
 public:
  enum CellType {
    kNoInit,   /**< Covariance is not computed yet */
    kNoPoints, /**< Covariance is not computed because of no points */
    kPoint,    /**< Covariance degenerates as a point */
    kLine,     /**< Covariance degenerates as a line */
    kRegular   /**< Covariance is computed well */
  };

  Cell2D();

  void AddPoint(const Eigen::Vector2d &point);

  void AddPointWithCovariance(const Eigen::Vector2d &point,
                              const Eigen::Matrix2d &covariance);

  void ComputeGaussian();

  int GetN() const { return n_; }
  bool GetHasGaussian() const { return hasgaussian_; }
  double GetSize() const { return size_; }
  Eigen::Vector2d GetCenter() const { return center_; }
  Eigen::Vector2d GetMean() const { return mean_; }
  Eigen::Matrix2d GetCov() const { return cov_; }
  Eigen::Vector2d GetEvals() const { return evals_; }
  Eigen::Matrix2d GetEvecs() const { return evecs_; }
  Eigen::Vector2d GetNormal() const { return normal_; }
  std::vector<Eigen::Vector2d> GetPoints() const { return points_; }
  std::vector<Eigen::Matrix2d> GetPointCovs() const { return point_covs_; }
  CellType GetCellType() const { return celltype_; }
  double GetRescaleRatio() const { return rescale_ratio_; }

  void SetN(int n) { n_ = n; }
  void SetHasGaussian(bool hasgaussian) { hasgaussian_ = hasgaussian; }
  void SetSize(double size) { size_ = size; }
  void SetCenter(const Eigen::Vector2d &center) { center_ = center; }
  void SetMean(const Eigen::Vector2d &mean) { mean_ = mean; }
  void SetCov(const Eigen::Matrix2d &cov) { cov_ = cov; }
  void SetEvals(const Eigen::Vector2d &evals) { evals_ = evals; }
  void SetEvecs(const Eigen::Matrix2d &evecs) { evecs_ = evecs; }
  void SetNormal(const Eigen::Vector2d &normal) { normal_ = normal; }
  void SetPoints(const std::vector<Eigen::Vector2d> &points) {
    points_ = points;
  }
  void SetPointCovs(const std::vector<Eigen::Matrix2d> &point_covs) {
    point_covs_ = point_covs;
  }
  void SetCellType(CellType celltype) { celltype_ = celltype; }
  void SetRescaleRatio(double rescale_ratio) { rescale_ratio_ = rescale_ratio; }

 private:
  int n_;                                   /**< Number of points */
  bool hasgaussian_;                        /**< Whether cell has a gaussian */
  double size_;                             /**< Cell size */
  Eigen::Vector2d center_;                  /**< Center of the cell */
  Eigen::Vector2d mean_;                    /**< Point mean */
  Eigen::Matrix2d cov_;                     /**< Point covariance */
  Eigen::Vector2d evals_;                   /**< Point eigenvalues */
  Eigen::Matrix2d evecs_;                   /**< Point eigenvectors */
  Eigen::Vector2d normal_;                  /**< Point Normal */
  std::vector<Eigen::Vector2d> points_;     /**< Points */
  std::vector<Eigen::Matrix2d> point_covs_; /**< Point covariances */
  CellType celltype_;                       /**< Cell type */
  double rescale_ratio_;                    /**< Rescale ratio */
};
