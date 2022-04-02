#pragma once
#include <ndt/cell2d.h>
#include <pcl/kdtree/kdtree_flann.h>

struct Vector2iComparator {
  inline bool operator()(const Eigen::Vector2i &a,
                         const Eigen::Vector2i &b) const {
    return std::tie(a(0), a(1)) < std::tie(b(0), b(1));
  }
};

class NMap2D {
  using MapType = std::map<Eigen::Vector2i, Cell2D, Vector2iComparator>;

 public:
  NMap2D() = delete;

  /**
   * @brief Construct a new NMap2D object
   *
   * @param cell_size Cell size of the NMap2D
   */
  explicit NMap2D(double cell_size);

  /**
   * @brief Load points into the NDTMap
   *
   * @param points Points to load
   */
  void LoadPoints(const std::vector<Eigen::Vector2d> &points);

  /**
   * @brief Load points and point covariances into the NDTMap
   *
   * @param points Points to load
   * @param point_covs Point covariances to load
   * @note @c points, @c point_covs should have the same size
   */
  void LoadPointsWithCovariances(
      const std::vector<Eigen::Vector2d> &points,
      const std::vector<Eigen::Matrix2d> &point_covs);

  /**
   * @brief Perform transformation to NDTMap
   *
   * @param T Transformation
   * @param include_data Transform data as well (default false)
   * @return Transformed cells
   */
  std::vector<Cell2D> TransformCells(const Eigen::Affine2d &T,
                                     bool include_data = false) const;

  /**
   * @brief Search Nearest Cell by kd-tree
   *
   * @param point input point
   * @return Nearest Cell
   */
  const Cell2D &SearchNearestCell(const Eigen::Vector2d &query) const;

  /**
   * @brief Search Nearest Cell by kd-tree
   *
   * @param[in] point input point
   * @param[out] index index of the result
   * @return Nearest Cell
   */
  const Cell2D &SearchNearestCell(const Eigen::Vector2d &query,
                                  Eigen::Vector2i &index) const;

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
  double cell_size_;
  Eigen::Vector2d min_voxel_;
};
