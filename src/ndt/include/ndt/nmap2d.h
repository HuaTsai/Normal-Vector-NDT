#pragma once
#include <ndt/cell2d.h>
#include <pcl/kdtree/kdtree_flann.h>

class NMap2D {
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
        seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 +
                (seed << 6) + (seed >> 2);
      }
      return seed;
    }
  };
  using MapType =
      std::unordered_map<Eigen::Vector2i, Cell2D, MatrixHash<Eigen::Vector2i>>;

 public:
  NMap2D() = delete;

  /**
   * @brief Construct a new NMap2D object
   *
   * @param cell_size Cell size of the NMap2D
   */
  explicit NMap2D(double cell_size);

  /**
   * @brief Load points into the NMap2D
   *
   * @param points Points to load
   */
  void LoadPoints(const std::vector<Eigen::Vector2d> &points);

  /**
   * @brief Load points and covariances into the NMap2D
   *
   * @param points Points to load
   * @param point_cov Same covariance for all points
   */
  void LoadPointsWithCovariances(const std::vector<Eigen::Vector2d> &points,
                                 const Eigen::Matrix2d &point_cov);

  /**
   * @brief Load points and point covariances into the NMap2D
   *
   * @param points Points to load
   * @param point_covs Point covariances to load
   * @note @c points, @c point_covs should have the same size
   */
  void LoadPointsWithCovariances(
      const std::vector<Eigen::Vector2d> &points,
      const std::vector<Eigen::Matrix2d> &point_covs);

  /**
   * @brief Perform transformation to NMap2D
   *
   * @param T Transformation
   * @param include_data Transform data as well (default false)
   * @return Transformed cells
   */
  std::vector<Cell2D> TransformCells(const Eigen::Affine2d &T,
                                     bool include_data = false) const;

  /**
   * @brief Search Nearest Cell2D by kd-tree
   *
   * @param query input point
   * @return Nearest Cell2D
   */
  const Cell2D &SearchNearestCell(const Eigen::Vector2d &query) const;

  /**
   * @brief Search Nearest Cell2D by kd-tree
   *
   * @param[in] query input point
   * @param[out] index index of the result
   * @return Nearest Cell2D
   */
  const Cell2D &SearchNearestCell(const Eigen::Vector2d &query,
                                  Eigen::Vector2i &index) const;

  // We use reference_wrapper to accompany the const Cell & in implementation
  std::vector<std::reference_wrapper<const Cell2D>> SearchCellsInRadius(
      const Eigen::Vector2d &query, double radius) const;

  /**
   * @brief Get the index by a given point
   *
   * @param point Input point
   */
  Eigen::Vector2i GetIndexForPoint(const Eigen::Vector2d &point) const;

  size_t size() const { return cells_.size(); }
  MapType::iterator begin() { return cells_.begin(); }
  MapType::const_iterator begin() const { return cells_.begin(); }
  MapType::iterator end() { return cells_.end(); }
  MapType::const_iterator end() const { return cells_.end(); }
  Cell2D &at(const Eigen::Vector2i &idx) { return cells_.at(idx); }
  const Cell2D &at(const Eigen::Vector2i &idx) const { return cells_.at(idx); }

  pcl::KdTreeFLANN<pcl::PointXYZ> GetKDTree() const { return kdtree_; }
  double GetCellSize() const { return cell_size_; }
  Eigen::Vector2d GetMinVoxel() const { return min_voxel_; }

 private:
  /**
   * @brief Compute min_voxel_ value
   *
   * @param points Input points
   */
  void ComputeVoxelOffset(const std::vector<Eigen::Vector2d> &points);

  /**
   * @brief Make KD Tree
   *
   * @param points Input points
   */
  void MakeKDTree(const std::vector<Eigen::Vector2d> &points);

  MapType cells_;
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_;
  std::vector<Eigen::Vector2d> kdtree_pts_; /* Avoid precision loss issue */
  double cell_size_;
  Eigen::Vector2d min_voxel_;
};
