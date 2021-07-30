/**
 * @file eigen_utils.h
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Eigen Utilities
 * @version 0.1
 * @date 2021-07-29
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <bits/stdc++.h>

#include <Eigen/Dense>
#include <gsl/gsl>

/**
 * @brief Compute mean of points
 *
 * @param points Input points
 * @return Mean of points
 * @note This function does not check the validity of its elements
 */
inline Eigen::Vector2d ComputeMean(const std::vector<Eigen::Vector2d> &points) {
  Eigen::Vector2d init = Eigen::Vector2d::Zero();
  auto ret = std::accumulate(points.begin(), points.end(), init);
  ret /= points.size();
  return ret;
}

/**
 * @brief Compute mean of matrices
 *
 * @param matrices Input matrices
 * @return Mean of matrices
 * @note This function does not check the validity of its elements
 */
inline Eigen::Matrix2d ComputeMean(
    const std::vector<Eigen::Matrix2d> &matrices) {
  Eigen::Matrix2d init = Eigen::Matrix2d::Zero();
  auto ret = std::accumulate(matrices.begin(), matrices.end(), init);
  ret /= matrices.size();
  return ret;
}

/**
 * @brief Compute covariance of points
 *
 * @param points Input points
 * @param mean Input mean of points
 * @param offset Offset added to the covariance (default zero)
 * @return Covariance of points
 * @note This function does not check the validity of its elements
 */
inline Eigen::Matrix2d ComputeCov(
    const std::vector<Eigen::Vector2d> &points, const Eigen::Vector2d &mean,
    const Eigen::Matrix2d &offset = Eigen::Matrix2d::Zero()) {
  int n = points.size();
  if (n <= 2) return offset;
  Eigen::MatrixXd mp(2, n);
  for (int i = 0; i < n; ++i) mp.col(i) = points.at(i) - mean;
  Eigen::Matrix2d ret;
  ret = mp * mp.transpose() / (n - 1) + offset;
  return ret;
}

/**
 * @brief Compute covariance of points
 *
 * @param points Input points
 * @param mean Input mean of points
 * @param offsets Offsets that will be averaged and added to the covariance
 * @return Covariance of points
 * @pre Arguments @c points and @c offsets should have the same size.
 * @note This function does not check the validity of its elements
 */
inline Eigen::Matrix2d ComputeCov(const std::vector<Eigen::Vector2d> &points,
                                  const Eigen::Vector2d &mean,
                                  const std::vector<Eigen::Matrix2d> &offsets) {
  Expects(points.size() == offsets.size());
  return ComputeCov(points, mean, ComputeMean(offsets));
}

/**
 * @brief Compute eigenvalues and eigenvectors from covariance
 *
 * @param[in] covariance Input covariance
 * @param[out] evals Output eigenvalues
 * @param[out] evecs Output eigenvectors
 */
inline void ComputeEvalEvec(const Eigen::Matrix2d &covariance,
                            Eigen::Vector2d &evals, Eigen::Matrix2d &evecs) {
  Eigen::EigenSolver<Eigen::Matrix2d> es(covariance);
  evals = es.pseudoEigenvalueMatrix().diagonal();
  evecs = es.pseudoEigenvectors();
}

/**
 * @brief Remove points that contain NaN or Inf
 *
 * @param points Input points
 * @return Result points
 */
inline std::vector<Eigen::Vector2d> ExcludeNaNInf(
    const std::vector<Eigen::Vector2d> &points) {
  std::vector<Eigen::Vector2d> ret;
  std::copy_if(points.begin(), points.end(), back_inserter(ret),
               [](const Eigen::Vector2d &v) { return v.allFinite(); });
  return ret;
}

/**
 * @brief Remove points and covariances that contain NaN or Inf
 *
 * @param points Input points
 * @param covariances Input covariances
 * @return Pair of result points and result covariances
 * @details This function keeps the entries that have both valid point and valid
 * covariance
 * @pre Arguments @c points and @c covariances should have the same size
 */
inline std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Matrix2d>>
ExcludeNaNInf2(const std::vector<Eigen::Vector2d> &points,
               const std::vector<Eigen::Matrix2d> &covariances) {
  Expects(points.size() == covariances.size());
  std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Matrix2d>> ret;
  for (size_t i = 0; i < points.size(); ++i) {
    if (points[i].allFinite() && covariances[i].allFinite()) {
      ret.first.push_back(points[i]);
      ret.second.push_back(covariances[i]);
    }
  }
  return ret;
}
