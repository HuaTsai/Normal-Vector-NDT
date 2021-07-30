/**
 * @file map_interface.h
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Class of MapInterface
 * @version 0.1
 * @date 2021-07-29
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <sndt/eigen_utils.h>

class MapInterface {
 public:
  /**
   * @brief Construct a new MapInterface object
   *
   * @param cell_size Cell size of the MapInterface
   */
  explicit MapInterface(double cell_size) {
    is_loaded_ = false;
    cell_size_ = cell_size;
    map_size_.setZero();
    map_center_.setZero();
    map_center_index_.setZero();
    cellptrs_size_.setZero();
  }

  virtual ~MapInterface() = default;

  /**
   * @brief Get the index by a given point
   * 
   * @param point Input point
   */
  Eigen::Vector2i GetIndexForPoint(const Eigen::Vector2d &point) const {
    Eigen::Vector2i ret;
    ret(0) = floor((point(0) - map_center_(0)) / cell_size_ + 0.5) + map_center_index_(0);
    ret(1) = floor((point(1) - map_center_(1)) / cell_size_ + 0.5) + map_center_index_(1);
    return ret;
  }

  /**
   * @brief Convert the information of the MapInterface to a string
   */
  virtual std::string ToString() const = 0;

  double GetCellSize() const { return cell_size_; }
  Eigen::Vector2d GetMapSize() const { return map_size_; }
  Eigen::Vector2d GetMapCenter() const { return map_center_; }

 protected:
  /**
   * @brief Update mata information of MapInterface by points
   *
   * @param points Input points
   * @details This function updates @c map_size_, @c map_center_, @c
   * map_center_index, and @c cellptrs_size
   */
  void GuessMapSize(const std::vector<Eigen::Vector2d> &points) {
    auto valids = ExcludeNaNInf(points);
    map_center_ = ComputeMean(valids);
    double maxdist = 0;
    for (size_t i = 0; i < valids.size(); ++i)
      maxdist = std::max(maxdist, (valids[i] - map_center_).norm());
    map_size_.fill(maxdist * 4);
    cellptrs_size_(0) = int(ceil(map_size_(0) / cell_size_));
    cellptrs_size_(1) = int(ceil(map_size_(1) / cell_size_));
    map_center_index_(0) = cellptrs_size_(0) / 2;
    map_center_index_(1) = cellptrs_size_(1) / 2;
  }

  /**
   * @brief Initialize @c cellptrs_ by mata information of MapInterface
   */
  virtual void Initialize() = 0;

  /**
   * @brief Check the validity of an index
   * 
   * @param index Input index
   */
  bool IsValidIndex(const Eigen::Vector2i &index) const {
    return index(0) >= 0 && index(1) < cellptrs_size_(0) &&
           index(1) >= 0 && index(1) < cellptrs_size_(1);
  }

  bool is_loaded_;
  double cell_size_;
  Eigen::Vector2d map_size_;
  Eigen::Vector2d map_center_;
  Eigen::Vector2i map_center_index_;
  Eigen::Vector2i cellptrs_size_;
};
