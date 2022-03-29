#pragma once
#include <nndt/cell.h>
#include <pcl/kdtree/kdtree_flann.h>

struct Vector3iComparator {
  inline bool operator()(const Eigen::Vector3i &a,
                         const Eigen::Vector3i &b) const {
    // return std::tie(a(0), a(1), a(2)) < std::tie(b(0), b(1), b(2));
    if (a(0) != b(0))
      return a(0) < b(0);
    else if (a(1) != b(1))
      return a(1) < b(1);
    else
      return a(2) < b(2);
  }
};

class NMap {
  using MapType = std::map<Eigen::Vector3i, Cell, Vector3iComparator>;

 public:
  /**
   * @brief Construct a new NMap object
   *
   * @param cell_size Cell size of the NMap
   */
  explicit NMap(double cell_size);

  /**
   * @brief Load points into the NDTMap
   *
   * @param points Points to load
   */
  void LoadPoints(const std::vector<Eigen::Vector3d> &points);

  /**
   * @brief Load points and point covariances into the NDTMap
   *
   * @param points Points to load
   * @param point_covs Point covariances to load
   * @note @c points, @c point_covs should have the same size
   */
  void LoadPointsWithCovariances(
      const std::vector<Eigen::Vector3d> &points,
      const std::vector<Eigen::Matrix3d> &point_covs);

  /**
   * @brief Perform transformation to NDTMap
   *
   * @param T Transformation
   * @param include_data Transform data as well (default false)
   * @return Transformed cells
   */
  std::vector<Cell> TransformCells(const Eigen::Affine3d &T,
                                   bool include_data = false) const;

  /**
   * @brief Search Nearest Cell by kd-tree
   *
   * @param point input point
   * @return Nearest Cell
   */
  const Cell &SearchNearestCell(const Eigen::Vector3d &query) const;

  /**
   * @brief Search Nearest Cell by kd-tree
   *
   * @param[in] point input point
   * @param[out] index index of the result
   * @return Nearest Cell
   */
  const Cell &SearchNearestCell(const Eigen::Vector3d &query,
                                Eigen::Vector3i &index) const;

  /**
   * @brief Get the index by a given point
   *
   * @param point Input point
   */
  Eigen::Vector3i GetIndexForPoint(const Eigen::Vector3d &point) const;

  size_t size() const { return cells_.size(); }
  MapType::iterator begin() { return cells_.begin(); }
  MapType::const_iterator begin() const { return cells_.begin(); }
  MapType::iterator end() { return cells_.end(); }
  MapType::const_iterator end() const { return cells_.end(); }
  Cell &at(const Eigen::Vector3i &idx) { return cells_.at(idx); }
  const Cell &at(const Eigen::Vector3i &idx) const { return cells_.at(idx); }

  pcl::KdTreeFLANN<pcl::PointXYZ> GetKDTree() const { return kdtree_; }
  double GetCellSize() const { return cell_size_; }
  Eigen::Vector3d GetMinVoxel() const { return min_voxel_; }

 private:
  /**
   * @brief Compute min_voxel_ value
   *
   * @param points Input points
   */
  void ComputeVoxelOffset(const std::vector<Eigen::Vector3d> &points);

  /**
   * @brief Make KD Tree
   *
   * @param points Input points
   */
  void MakeKDTree(const std::vector<Eigen::Vector3d> &points);

  MapType cells_;
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_;
  double cell_size_;
  Eigen::Vector3d min_voxel_;
};
