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
 * @param indices Input indices ({} means using all the points)
 * @return Mean of points
 * @note This function does not check the validity of its elements
 */
inline Eigen::Vector2d ComputeMean(const std::vector<Eigen::Vector2d> &points,
                                   const std::vector<int> &indices = {}) {
  Eigen::Vector2d ret = Eigen::Vector2d::Zero();
  if (indices.size()) {
    ret = std::accumulate(
        indices.begin(), indices.end(), ret,
        [&points](const auto &a, int b) { return a + points[b]; });
    ret /= indices.size();
  } else {
    ret = std::accumulate(points.begin(), points.end(), ret);
    ret /= points.size();
  }
  return ret;
}

/**
 * @brief Compute mean of matrices
 *
 * @param matrices Input matrices
 * @param indices Input indices
 * @return Mean of matrices
 * @note This function does not check the validity of its elements
 */
inline Eigen::Matrix2d ComputeMean(const std::vector<Eigen::Matrix2d> &matrices,
                                   const std::vector<int> &indices = {}) {
  Eigen::Matrix2d ret = Eigen::Matrix2d::Zero();
  if (indices.size()) {
    ret = std::accumulate(
        indices.begin(), indices.end(), ret,
        [&matrices](const auto &a, int b) { return a + matrices[b]; });
    ret /= indices.size();
  } else {
    ret = std::accumulate(matrices.begin(), matrices.end(), ret);
    ret /= matrices.size();
  }
  return ret;
}

// TODO: incorporate this
inline void ComputeMeanAndCov(const std::vector<Eigen::Vector2d> &points,
                              const std::vector<Eigen::Matrix2d> &offsets,
                              const std::vector<int> &indices,
                              Eigen::Vector2d &mean,
                              Eigen::Matrix2d &cov) {
  int n = indices.size();
  mean = ComputeMean(points, indices);
  if (n == 1) {
    cov = offsets[indices[0]];
    return;
  }
  Eigen::MatrixXd mp(2, n);
  for (int i = 0; i < n; ++i) mp.col(i) = points[indices[i]] - mean;
  cov = mp * mp.transpose() / (n - 1) + ComputeMean(offsets, indices);
}

// TODO: incorporate this
inline void ComputeMeanAndCov(const std::vector<Eigen::Vector2d> &points,
                              const std::vector<int> &indices,
                              Eigen::Vector2d &mean,
                              Eigen::Matrix2d &cov) {
  int n = indices.size();
  mean = ComputeMean(points, indices);
  if (n <= 2) {
    cov.fill(std::numeric_limits<double>::quiet_NaN());
    return;
  }
  Eigen::MatrixXd mp(2, n);
  for (int i = 0; i < n; ++i) mp.col(i) = points[indices[i]] - mean;
  cov = mp * mp.transpose() / (n - 1);
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
    const std::vector<Eigen::Vector2d> &points,
    const Eigen::Vector2d &mean,
    const Eigen::Matrix2d &offset = Eigen::Matrix2d::Zero()) {
  int n = points.size();
  if (n == 1) return offset;
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
  return ComputeCov(points, mean, ComputeMean(offsets));
}

/**
 * @brief Compute eigenvalues and eigenvectors from covariance
 *
 * @param[in] covariance Input covariance
 * @param[out] evals Output eigenvalues
 * @param[out] evecs Output eigenvectors
 * @details The function computeDirect() uses closed-form algorithm to perform
 * eigenvalue decomposition for a symmetric real matrix. This method is
 * significantly faster than the QR iterative algorithm. Besides, evals(0) will
 * be smaller than or equal to evals(1).
 * @see Eigen::SelfAdjointEigenSolver and Catalogue of dense decompositions
 */
inline void ComputeEvalEvec(const Eigen::Matrix2d &covariance,
                            Eigen::Vector2d &evals,
                            Eigen::Matrix2d &evecs) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> evd;
  evd.computeDirect(covariance);
  evals = evd.eigenvalues();
  evecs = evd.eigenvectors();
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

// TODO: document and incorporate
inline std::vector<int> ExcludeNaNInf3(
    const std::vector<Eigen::Vector2d> &points) {
  std::vector<int> ret;
  for (size_t i = 0; i < points.size(); ++i)
    if (points[i].allFinite()) ret.push_back(i);
  return ret;
}

// TODO: document and incorporate
inline std::vector<int> ExcludeNaNInf3(
    const std::vector<Eigen::Vector2d> &points,
    const std::vector<Eigen::Matrix2d> &covariances) {
  std::vector<int> ret;
  for (size_t i = 0; i < points.size(); ++i)
    if (points[i].allFinite() && covariances[i].allFinite()) ret.push_back(i);
  return ret;
}

class RandomTransformGenerator2D {
 public:
  RandomTransformGenerator2D(double radius)
      : dre_(new std::default_random_engine()), radius_(radius) {}
  std::vector<Eigen::Affine2d> Generate(int sizes) {
    std::vector<Eigen::Affine2d> ret;
    std::uniform_real_distribution<> urd(-M_PI, M_PI);
    std::uniform_real_distribution<> urd2(-M_PI / 4, M_PI / 4);
    for (int i = 0; i < sizes; ++i) {
      double angle = urd(*dre_);
      double x = radius_ * cos(angle);
      double y = radius_ * sin(angle);
      // double t = urd2(*dre_);
      double t = 0;
      ret.push_back(Eigen::Translation2d(x, y) * Eigen::Rotation2Dd(t));
    }
    return ret;
  }

 private:
  std::shared_ptr<std::default_random_engine> dre_;
  double radius_;
};
