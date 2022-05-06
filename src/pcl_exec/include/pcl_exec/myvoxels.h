// This file is originated and modified from PCL.
// Removing all the comments makes it easier for development
#pragma once

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>

#include <map>

// IMPLEMNET
// 1) radiusSearch
// 2) filter
// 3) setleafsize
// 4) setinputcloud
namespace pcl {
template <typename PointT>
class MyVoxels : public VoxelGrid<PointT> {
protected:
  using VoxelGrid<PointT>::input_;
  using VoxelGrid<PointT>::leaf_size_;
  using VoxelGrid<PointT>::min_b_;
  using VoxelGrid<PointT>::max_b_;
  using VoxelGrid<PointT>::inverse_leaf_size_;
  using VoxelGrid<PointT>::div_b_;
  using VoxelGrid<PointT>::divb_mul_;

  using PointCloud = typename Filter<PointT>::PointCloud;
  using PointCloudPtr = typename PointCloud::Ptr;

public:
  using Ptr = shared_ptr<VoxelGrid<PointT>>;
  using ConstPtr = shared_ptr<const VoxelGrid<PointT>>;

  struct Leaf {
    Leaf()
    : n(0)
    , mean_(Eigen::Vector3d::Zero())
    , centroid(Eigen::Vector4f::Zero())
    , cov_(Eigen::Matrix3d::Zero())
    , icov_(Eigen::Matrix3d::Zero())
    , evecs_(Eigen::Matrix3d::Identity())
    , evals_(Eigen::Vector3d::Zero())
    {}
    int n;
    Eigen::Vector3d mean_;
    Eigen::Vector4f centroid;
    Eigen::Matrix3d cov_;
    Eigen::Matrix3d icov_;
    Eigen::Matrix3d evecs_;
    Eigen::Vector3d evals_;
  };

  using LeafPtr = Leaf*;
  using LeafConstPtr = const Leaf*;

  MyVoxels()
  : min_points_per_voxel_(6)
  , min_covar_eigvalue_mult_(0.01)
  , leaves_()
  , voxel_centroids_()
  , kdtree_()
  {
    leaf_size_.setZero();
    min_b_.setZero();
    max_b_.setZero();
  }

  inline void
  filter()
  {
    voxel_centroids_ = PointCloudPtr(new PointCloud);
    applyFilter(*voxel_centroids_);
    kdtree_.setInputCloud(voxel_centroids_);
  }

  inline LeafConstPtr
  getLeaf(int index)
  {
    auto leaf_iter = leaves_.find(index);
    if (leaf_iter != leaves_.end()) {
      LeafConstPtr ret(&(leaf_iter->second));
      return ret;
    }
    return nullptr;
  }

  inline LeafConstPtr
  getLeaf(PointT& p)
  {
    // clang-format off
    int ijk0 = static_cast<int>(std::floor(p.x * inverse_leaf_size_[0]) - min_b_[0]);
    int ijk1 = static_cast<int>(std::floor(p.y * inverse_leaf_size_[1]) - min_b_[1]);
    int ijk2 = static_cast<int>(std::floor(p.z * inverse_leaf_size_[2]) - min_b_[2]);
    // clang-format on
    int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];
    return getLeaf(idx);
  }

  inline const std::map<std::size_t, Leaf>&
  getLeaves()
  {
    return leaves_;
  }

  inline PointCloudPtr
  getCentroids()
  {
    return voxel_centroids_;
  }

  int
  nearestKSearch(const PointT& point,
                 int k,
                 std::vector<LeafConstPtr>& k_leaves,
                 std::vector<float>& k_sqr_distances) const
  {
    k_leaves.clear();

    // Find k-nearest neighbors in the occupied voxel centroid cloud
    Indices k_indices;
    k = kdtree_.nearestKSearch(point, k, k_indices, k_sqr_distances);

    // Find leaves corresponding to neighbors
    k_leaves.reserve(k);
    for (const auto& k_index : k_indices) {
      auto voxel = leaves_.find(voxel_centroids_leaf_indices_[k_index]);
      if (voxel == leaves_.end()) {
        continue;
      }

      k_leaves.push_back(&voxel->second);
    }
    return k_leaves.size();
  }

  inline int
  nearestKSearch(const PointCloud& cloud,
                 int index,
                 int k,
                 std::vector<LeafConstPtr>& k_leaves,
                 std::vector<float>& k_sqr_distances) const
  {
    if (index >= static_cast<int>(cloud.size()) || index < 0)
      return (0);
    return (nearestKSearch(cloud[index], k, k_leaves, k_sqr_distances));
  }

  int
  radiusSearch(const PointT& point,
               double radius,
               std::vector<LeafConstPtr>& k_leaves,
               std::vector<float>& k_sqr_distances,
               unsigned int max_nn = 0) const
  {
    k_leaves.clear();
    Indices indices;
    int k = kdtree_.radiusSearch(point, radius, indices, k_sqr_distances, max_nn);
    k_leaves.reserve(k);
    for (auto idx : indices) {
      const auto voxel = leaves_.find(voxel_centroids_leaf_indices_[idx]);
      if (voxel == leaves_.end())
        continue;
      k_leaves.push_back(&voxel->second);
    }
    return k_leaves.size();
  }

  inline int
  radiusSearch(const PointCloud& cloud,
               int index,
               double radius,
               std::vector<LeafConstPtr>& k_leaves,
               std::vector<float>& k_sqr_distances,
               unsigned int max_nn = 0) const
  {
    if (index >= static_cast<int>(cloud.size()) || index < 0)
      return 0;
    return radiusSearch(cloud[index], radius, k_leaves, k_sqr_distances, max_nn);
  }

protected:
  void
  applyFilter(PointCloud& output) override;
  int min_points_per_voxel_;
  double min_covar_eigvalue_mult_;
  std::map<std::size_t, Leaf> leaves_;
  PointCloudPtr voxel_centroids_;
  std::vector<int> voxel_centroids_leaf_indices_;
  KdTreeFLANN<PointT> kdtree_;
};
} // namespace pcl

#include "myvoxels.hpp"
