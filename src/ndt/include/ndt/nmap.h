#pragma once
#include <ndt/cell.h>
#include <pcl/kdtree/kdtree_flann.h>

// Hash function for Eigen matrix and vector.
// The code is from `hash_combine` function of the Boost library. See
// http://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
// .
template <typename T>
struct MatrixHash : std::unary_function<T, size_t> {
  std::size_t operator()(T const &matrix) const {
    // Note that it is oblivious to the storage order of Eigen matrix (column-
    // or row-major). It will give you the same hash value for two different
    // matrices if they are the transpose of each other in different storage
    // order.
    size_t seed = 0;
    for (int i = 0; i < matrix.size(); ++i) {
      auto elem = *(matrix.data() + i);
      seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) +
              (seed >> 2);
    }
    return seed;
  }
};

class NMap {
  using MapType =
      std::unordered_map<Eigen::Vector3i, Cell, MatrixHash<Eigen::Vector3i>>;

 public:
  NMap() = delete;

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

  // We use reference_wrapper to accompany the const Cell & in implementation
  std::vector<std::reference_wrapper<const Cell>> SearchCellsInRadius(
      const Eigen::Vector3d &query, double radius) const;

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
