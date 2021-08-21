/**
 * @file sndt_map.h
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Class Declaration of SNDTMap
 * @version 0.1
 * @date 2021-07-28
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <sndt/map_interface.h>
#include <sndt/sndt_cell.h>

class SNDTMap : public MapInterface {
 public:
  /**
   * @brief Construct a new SNDTMap object
   *
   * @param cell_size Cell size of the SNDTMap
   */
  explicit SNDTMap(double cell_size);

  ~SNDTMap();

  /**
   * @brief Load points into the SNDTMap
   *
   * @param points Points to load
   * @param normals Normals to load
   * @note @c points and @c normals should have the same size
   */
  void LoadPointsAndNormals(const std::vector<Eigen::Vector2d> &points,
                            const std::vector<Eigen::Vector2d> &normals);

  /**
   * @brief Load points and point covariances into the SNDTMap
   *
   * @param points Points to load
   * @param point_covs Point covariances to load
   * @param normals Normals to load
   * @note @c points, @c point_covs, and @c normals should have the same size
   */
  void LoadPointsWithCovariancesAndNormals(
      const std::vector<Eigen::Vector2d> &points,
      const std::vector<Eigen::Matrix2d> &point_covs,
      const std::vector<Eigen::Vector2d> &normals);

  /**
   * @brief Perform transformation to SNDTMap
   *
   * @param T Transformation
   * @param include_data Transform data as well (default false)
   * @return Transformed cells
   */
  std::vector<std::shared_ptr<SNDTCell>> PseudoTransformCells(
      const Eigen::Affine2d &T, bool include_data = false) const;

  /**
   * @brief Get and allocate the cell by a given point
   * 
   * @param point Input point
   */
  SNDTCell *GetCellAndAllocate(const Eigen::Vector2d &point);

  /**
   * @brief Get the cell by a given point
   * 
   * @param point Input point
   */
  const SNDTCell *GetCellForPoint(const Eigen::Vector2d &point) const;

  /**
   * @brief Convert the information of the SNDTMap to a string
   */
  virtual std::string ToString() const override;

  /**
   * @brief Get the Points object
   * 
   * @note The points are retrived from active_cells_, which does not
   * necessarily equal to the original input
   */
  std::vector<Eigen::Vector2d> GetPoints() const;

  /**
   * @brief Get the Points object
   * 
   * @brief return points that in valid gaussian cells
   */
  std::vector<Eigen::Vector2d> GetPointsWithGaussianCell() const;

  /**
   * @brief Get the Normals object
   * 
   * @note The normals are retrived from active_cells_, which does not
   * necessarily equal to the original input
   */
  std::vector<Eigen::Vector2d> GetNormals() const;

  size_t size() const { return active_cells_.size(); }
  std::vector<SNDTCell *>::iterator begin() { return active_cells_.begin(); }
  std::vector<SNDTCell *>::const_iterator begin() const { return active_cells_.begin(); }
  std::vector<SNDTCell *>::iterator end() { return active_cells_.end(); }
  std::vector<SNDTCell *>::const_iterator end() const { return active_cells_.end(); }

 private:
  /**
   * @brief Initialize @c cellptrs_ by mata information of SNDTMap
   */
  virtual void Initialize() override;

  std::vector<SNDTCell *> active_cells_;
  SNDTCell ***cellptrs_;
};
