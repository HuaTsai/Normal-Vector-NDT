#pragma once
#include <bits/stdc++.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>

#include <Eigen/Dense>

Eigen::Affine3d Affine3dFromXYZRPY(const Eigen::Matrix<double, 6, 1> &xyzrpy);

Eigen::Affine3d Affine3dFromXYZRPY(const std::vector<double> &xyzrpy);

Eigen::Matrix<double, 6, 1> XYZRPYFromAffine3d(const Eigen::Affine3d &mtx);

Eigen::Affine3d Affine3dFromAffine2d(const Eigen::Affine2d &aff);

Eigen::Affine3d Conserve2DFromAffine3d(const Eigen::Affine3d &T);

Eigen::Vector2d TransNormRotDegAbsFromAffine2d(const Eigen::Affine2d &aff);

Eigen::Vector2d TransNormRotDegAbsFromAffine3d(const Eigen::Affine3d &aff);

geometry_msgs::PoseStamped MakePoseStampedMsg(const ros::Time &time,
                                              const Eigen::Affine3d &aff);

geometry_msgs::PoseStamped MakePoseStampedMsg(const ros::Time &time,
                                              const Eigen::Matrix4f &mtx);

template <int D>
inline std::vector<Eigen::Matrix<double, D, 1>> TransformPoints(
    const std::vector<Eigen::Matrix<double, D, 1>> &points,
    const Eigen::Transform<double, D, Eigen::TransformTraits::Affine> &aff) {
  std::vector<Eigen::Matrix<double, D, 1>> ret(points.size());
  std::transform(points.begin(), points.end(), ret.begin(),
                 [&aff](auto p) { return aff * p; });
  return ret;
}

template <int D>
inline void TransformPointsInPlace(
    std::vector<Eigen::Matrix<double, D, 1>> &points,
    const Eigen::Transform<double, D, Eigen::TransformTraits::Affine> &aff) {
  std::transform(points.begin(), points.end(), points.begin(),
                 [&aff](auto p) { return aff * p; });
}

template <int D>
inline std::vector<Eigen::Matrix<double, D, 1>> TransformNormals(
    const std::vector<Eigen::Matrix<double, D, 1>> &normals,
    const Eigen::Transform<double, D, Eigen::TransformTraits::Affine> &aff) {
  std::vector<Eigen::Matrix<double, D, 1>> ret(normals.size());
  std::transform(normals.begin(), normals.end(), ret.begin(), [&aff](auto p) {
    return Eigen::Matrix<double, D, 1>(aff.rotation() * p);
  });
  return ret;
}

/**
 * @brief Compute eigenvalues and eigenvectors from covariance
 *
 * @tparam D Dimension, only 2 or 3 is allowed
 * @param[in] covariance Input covariance
 * @param[out] evals Output eigenvalues
 * @param[out] evecs Output eigenvectors
 * @details The function computeDirect() uses closed-form algorithm to perform
 * eigenvalue decomposition for a symmetric real matrix.\n This method is
 * significantly faster than the QR iterative algorithm.\n The eigenvalues are
 * sorted in increasing order.
 * @see Eigen::SelfAdjointEigenSolver and Catalogue of dense decompositions
 * @note This is a template function, so we add inline to prevent include
 * issue.
 */
template <int D>
inline void ComputeEvalEvec(const Eigen::Matrix<double, D, D> &covariance,
                            Eigen::Matrix<double, D, 1> &evals,
                            Eigen::Matrix<double, D, D> &evecs) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, D, D>> evd;
  evd.computeDirect(covariance);
  evals = evd.eigenvalues();
  evecs = evd.eigenvectors();
}

/**
 * @brief Mean of input data
 *
 * @tparam T VectorXd or MatrixXd
 * @param data Input data
 * @return Mean of input data
 * @note This is a template function, so we add inline to prevent include
 * issue.
 */
template <typename T>
inline T ComputeMean(const std::vector<T> &data) {
  if (!data.size()) return T::Zero();
  T ret = T::Zero();
  ret = std::accumulate(data.begin(), data.end(), ret);
  ret /= data.size();
  return ret;
}

/**
 * @brief Covariance of input data
 *
 * @tparam D Dimension (2 or 3)
 * @param data input data
 * @param mean mean of input data
 * @return Covariance of input data
 * @note This is a template function, so we add inline to prevent include
 * issue.
 */
template <int D>
inline Eigen::Matrix<double, D, D> ComputeCov(
    const std::vector<Eigen::Matrix<double, D, 1>> &data,
    const Eigen::Matrix<double, D, 1> &mean) {
  int n = data.size();
  if (n == 1) return Eigen::Matrix<double, D, D>::Zero();
  Eigen::MatrixXd mp(D, n);
  for (int i = 0; i < n; ++i) mp.col(i) = data[i] - mean;
  Eigen::Matrix<double, D, D> ret = mp * mp.transpose() / (n - 1);
  return ret;
}

template <typename T>
inline void ExcludeInfinite(const std::vector<T> &data, std::vector<T> &valid) {
  valid.clear();
  for (const auto &elem : data)
    if (elem.allFinite()) valid.push_back(elem);
}

template <typename T, typename U>
inline void ExcludeInfinite(const std::vector<T> &data1,
                            const std::vector<U> &data2,
                            std::vector<T> &valid1,
                            std::vector<U> &valid2) {
  valid1.clear();
  valid2.clear();
  for (size_t i = 0; i < data1.size(); ++i) {
    if (data1[i].allFinite() && data2[i].allFinite()) {
      valid1.push_back(data1[i]);
      valid2.push_back(data2[i]);
    }
  }
}

template <typename T>
inline void ExcludeInfiniteInPlace(std::vector<T> &data) {
  size_t n = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    if (data[i].allFinite()) {
      if (n != i) data[n] = data[i];
      ++n;
    }
  }
  data.resize(n);
}

template <typename T, typename U>
inline void ExcludeInfiniteInPlace(std::vector<T> &data1,
                                   std::vector<U> &data2) {
  size_t n = 0;
  for (size_t i = 0; i < data1.size(); ++i) {
    if (data1[i].allFinite() && data2[i].allFinite()) {
      if (n != i) {
        data1[n] = data1[i];
        data2[n] = data2[i];
      }
      ++n;
    }
  }
  data1.resize(n);
  data2.resize(n);
}
