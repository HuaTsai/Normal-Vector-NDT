/**
 * @file ndt_map.h
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Class Declaration of NDTMap
 * @version 0.1
 * @date 2021-07-29
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#pragma once
#include <sndt/map_interface.h>
#include <sndt/ndt_cell.h>

class NDTMap : public MapInterface {
 public:
  /**
   * @brief Construct a new NDTMap object
   *
   * @param cell_size Cell size of the NDTMap
   */
  explicit NDTMap(double cell_size);

  ~NDTMap();

  /**
   * @brief Load points into the NDTMap
   *
   * @param points Points to load
   * @param normals Normals to load
   * @note @c points and @c normals should have the same size
   */
  void LoadPoints(const std::vector<Eigen::Vector2d> &points);

  /**
   * @brief Load points and point covariances into the NDTMap
   *
   * @param points Points to load
   * @param point_covs Point covariances to load
   * @param normals Normals to load
   * @note @c points, @c point_covs, and @c normals should have the same size
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
  std::vector<std::shared_ptr<NDTCell>> PseudoTransformCells(
      const Eigen::Affine2d &T, bool include_data = false) const;

  /**
   * @brief Get and allocate the cell by a given point
   * 
   * @param point Input point
   */
  NDTCell *GetCellAndAllocate(const Eigen::Vector2d &point);

  /**
   * @brief Get the cell by a given point
   * 
   * @param point Input point
   */
  const NDTCell *GetCellForPoint(const Eigen::Vector2d &point) const;

  /**
   * @brief Convert the information of the NDTMap to a string
   */
  virtual std::string ToString() const override;

  /**
   * @brief Get the Points object
   * 
   * @note The points are retrived from active_cells_, which does not
   * necessarily equal to the original input
   */
  std::vector<Eigen::Vector2d> GetPoints() const;

  size_t size() const { return active_cells_.size(); }
  std::vector<NDTCell *>::iterator begin() { return active_cells_.begin(); }
  std::vector<NDTCell *>::const_iterator begin() const { return active_cells_.begin(); }
  std::vector<NDTCell *>::iterator end() { return active_cells_.end(); }
  std::vector<NDTCell *>::const_iterator end() const { return active_cells_.end(); }

 private:
  /**
   * @brief Initialize @c cellptrs_ by mata information of NDTMap
   */
  virtual void Initialize() override;

  std::vector<NDTCell *> active_cells_;
  NDTCell ***cellptrs_;
};
