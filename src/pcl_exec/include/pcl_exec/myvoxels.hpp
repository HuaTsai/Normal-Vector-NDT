#pragma once

#include <pcl/common/common.h>
#include <pcl/common/point_tests.h>
#include <pcl/filters/voxel_grid_covariance.h>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <boost/mpl/size.hpp>

template <typename PointT>
void pcl::VoxelGridCovariance<PointT>::applyFilter(PointCloud& output) {
  voxel_centroids_leaf_indices_.clear();
  output.height = 1;
  output.is_dense = true;
  output.clear();

  Eigen::Vector4f min_p, max_p;
  getMinMax3D<PointT>(*input_, min_p, max_p);

  min_b_[0] = static_cast<int>(std::floor(min_p[0] * inverse_leaf_size_[0]));
  max_b_[0] = static_cast<int>(std::floor(max_p[0] * inverse_leaf_size_[0]));
  min_b_[1] = static_cast<int>(std::floor(min_p[1] * inverse_leaf_size_[1]));
  max_b_[1] = static_cast<int>(std::floor(max_p[1] * inverse_leaf_size_[1]));
  min_b_[2] = static_cast<int>(std::floor(min_p[2] * inverse_leaf_size_[2]));
  max_b_[2] = static_cast<int>(std::floor(max_p[2] * inverse_leaf_size_[2]));
  div_b_ = max_b_ - min_b_ + Eigen::Vector4i::Ones();
  div_b_[3] = 0;

  leaves_.clear();
  divb_mul_ = Eigen::Vector4i(1, div_b_[0], div_b_[0] * div_b_[1], 0);

  int centroid_size = 4;
  for (const auto& point : *input_) {
    if (!input_->is_dense && !isXYZFinite(point)) continue;

    const Eigen::Vector4i ijk =
        Eigen::floor(point.getArray4fMap() * inverse_leaf_size_.array())
            .template cast<int>();

    int idx = (ijk - min_b_).dot(divb_mul_);

    Leaf& leaf = leaves_[idx];

    Eigen::Vector3d pt3d = point.getVector3fMap().template cast<double>();
    leaf.mean_ += pt3d;
    leaf.cov_ += pt3d * pt3d.transpose();

    // leaf.centroid.template head<3>() += point.getVector3fMap();
    leaf.centroid.head(3) += point.getVector3fMap();
    ++leaf.n;
  }

  output.reserve(leaves_.size());
  voxel_centroids_leaf_indices_.reserve(leaves_.size());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver;
  Eigen::Vector3d eigen_val;
  Eigen::Vector3d pt_sum;

  double min_covar_eigvalue;

  for (auto& [idx, leaf] : leaves_) {
    leaf.centroid /= static_cast<float>(leaf.n);
    pt_sum = leaf.mean_;
    leaf.mean_ /= leaf.n;

    if (leaf.n >= min_points_per_voxel_) {
      output.push_back(PointT());
      output.back().x = leaf.centroid[0];
      output.back().y = leaf.centroid[1];
      output.back().z = leaf.centroid[2];
      voxel_centroids_leaf_indices_.push_back(static_cast<int>(idx));

      leaf.cov_ = (leaf.cov_ - pt_sum * pt_sum.transpose()) / (leaf.n - 1.0);

      eigensolver.compute(leaf.cov_);
      eigen_val = eigensolver.eigenvalues();
      leaf.evecs_ = eigensolver.eigenvectors();

      if (eigen_val(0) < -Eigen::NumTraits<double>::dummy_precision() ||
          eigen_val(1) < -Eigen::NumTraits<double>::dummy_precision() ||
          eigen_val(2) <= 0) {
        PCL_WARN("Invalid eigen value! (%g, %g, %g)\n", eigen_val(0),
                 eigen_val(1), eigen_val(2));
        leaf.n = -1;
        continue;
      }

      min_covar_eigvalue = min_covar_eigvalue_mult_ * eigen_val(2);
      if (eigen_val(0) < min_covar_eigvalue) {
        eigen_val(0) = min_covar_eigvalue;
        if (eigen_val(1) < min_covar_eigvalue) {
          eigen_val(1) = min_covar_eigvalue;
        }
        leaf.cov_ =
            leaf.evecs_ * eigen_val.asDiagonal() * leaf.evecs_.inverse();
      }
      leaf.evals_ = eigen_val;

      leaf.icov_ = leaf.cov_.inverse();
      if (leaf.icov_.maxCoeff() == std::numeric_limits<float>::infinity() ||
          leaf.icov_.minCoeff() == -std::numeric_limits<float>::infinity()) {
        leaf.n = -1;
      }
    }
  }

  output.width = output.size();
}
