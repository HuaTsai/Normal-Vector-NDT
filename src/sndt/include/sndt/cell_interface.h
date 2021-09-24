/**
 * @file cell_interface.h
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Class of CellInterface
 * @version 0.1
 * @date 2021-07-28
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <bits/stdc++.h>

#include <Eigen/Dense>

class CellInterface {
 public:
  /**
   * @brief Construct a new CellInterface object
   */
  CellInterface() : n_(0), phasgaussian_(false), skew_rad_(0), size_(0) {
    center_.setZero();
    pmean_.setZero();
    pevals_.setZero();
    pcov_.setZero();
    pevecs_.setZero();
  }

  /**
   * @brief Declare destructor virtual in polymorphic base class
   */
  virtual ~CellInterface() = default;

  /**
   * @brief Pure virtual function for computing gaussian
   *
   * @see NDTCell::ComputeGaussian() and SNDTCell::ComputeGaussian()
   */
  virtual void ComputeGaussian() = 0;

  /**
   * @brief Pure virtual function for checking if the cell has a valid gaussian
   *
   * @see NDTCell::HasGaussian() and SNDTCell::HasGaussian()
   */
  virtual bool HasGaussian() const = 0;

  /**
   * @brief Pure virutal function for converting the information of this cell to
   * a string
   *
   * @see NDTCell::ToString() and SNDTCell::ToString()
   */
  virtual std::string ToString() const = 0;

  /**
   * @brief Add a point to this cell
   *
   * @param point Point (x, y) to be added
   * @note This function does not check whether the point is inside the cell
   * or not. The caller should take responsibility for calling this function.
   */
  void AddPoint(const Eigen::Vector2d &point) { points_.push_back(point); }

  /**
   * @brief Add a point and its covariance to this cell
   *
   * @param point Point (x, y) to be added
   * @param covariance Covariance to be added
   * @note This function does not check whether the point is inside the cell
   * or not. Besides, it also does not check the symmetricity of covariance. The
   * caller should take responsibility for calling this function.
   */
  void AddPointWithCovariance(const Eigen::Vector2d &point,
                              const Eigen::Matrix2d &covariance) {
    points_.push_back(point);
    point_covs_.push_back(covariance);
  }

  int GetN() const { return n_; }
  bool GetPHasGaussian() const { return phasgaussian_; }
  double GetSkewRad() const { return skew_rad_; }
  double GetSize() const { return size_; }
  Eigen::Vector2d GetCenter() const { return center_; }
  Eigen::Vector2d GetPointMean() const { return pmean_; }
  Eigen::Matrix2d GetPointCov() const { return pcov_; }
  Eigen::Vector2d GetPointEvals() const { return pevals_; }
  Eigen::Matrix2d GetPointEvecs() const { return pevecs_; }
  std::vector<Eigen::Vector2d> GetPoints() const { return points_; }
  std::vector<Eigen::Matrix2d> GetPointCovs() const { return point_covs_; }

  void SetN(int n) { n_ = n; }
  void SetPHasGaussian(bool phasgaussian) { phasgaussian_ = phasgaussian; }
  void SetSkewRad(double skew_rad) { skew_rad_ = skew_rad; }
  void SetSize(double size) { size_ = size; }
  void SetCenter(const Eigen::Vector2d &center) { center_ = center; }
  void SetPointMean(const Eigen::Vector2d &mean) { pmean_ = mean; }
  void SetPointCov(const Eigen::Matrix2d &cov) { pcov_ = cov; }
  void SetPointEvals(const Eigen::Vector2d &evals) { pevals_ = evals; }
  void SetPointEvecs(const Eigen::Matrix2d &evecs) { pevecs_ = evecs; }
  void SetPoints(const std::vector<Eigen::Vector2d> &points) {
    points_ = points;
  }
  void SetPointCovs(const std::vector<Eigen::Matrix2d> &point_covs) {
    point_covs_ = point_covs;
  }

 protected:
  int n_;                  /**< Number of points, no matter the validity */
  bool phasgaussian_;      /**< Whether this cell has a valid point gaussian */
  double skew_rad_;        /**< Tilted angle of the cell */
  double size_;            /**< Cell size */
  Eigen::Vector2d center_; /**< Center of the cell */
  Eigen::Vector2d pmean_;  /**< Point mean */
  Eigen::Matrix2d pcov_;   /**< Point covariance */
  Eigen::Vector2d pevals_; /**< Point eigenvectors */
  Eigen::Matrix2d pevecs_; /**< Point eigenvectors */
  std::vector<Eigen::Vector2d> points_;     /**< Points data */
  std::vector<Eigen::Matrix2d> point_covs_; /**< Point covariances data */
};
