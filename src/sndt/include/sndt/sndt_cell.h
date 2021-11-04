/**
 * @file sndt_cell.h
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Class Declaration of SNDTCell
 * @version 0.1
 * @date 2021-07-28
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <sndt/cell_interface.h>

class SNDTCell : public CellInterface {
 public:
  /**
   * @brief Cell types
   */
  enum CellType {
    kNoInit,   /**< Covariance is not computed yet */
    kRegular,  /**< Covariance is computed well */
    kRescale,  /**< Covariance is rescaled */
    kAssign,   /**< Covariance is assigned */
    kNoPoints, /**< Covariance is not computed because of no points */
    kInvalid   /**< Covariance is invalid */
  };

  SNDTCell();

  /**
   * @brief Compute gaussian
   *
   * @pre The size of @c points_, @c point_covs_, and @c normals_ are the same
   * @details This function updates mean and covariance of this cell, along with
   * its cell type. Rescaling covariance is performed if necessary.
   */
  virtual void ComputeGaussian() override;

  /**
   * @brief Check if the cell has a valid gaussian
   */
  virtual bool HasGaussian() const override;

  /**
   * @brief Convert the information of this cell to a string
   */
  virtual std::string ToString() const override;

  /**
   * @brief  Add a normal to this cell
   *
   * @param normal Normal (x, y) to be added
   * @warning This function does not check whether the normal is normalized or
   * not. The caller should take responsibility for calling this function.
   */
  void AddNormal(const Eigen::Vector2d &normal) { normals_.push_back(normal); }

  CellType GetPCellType() const { return pcelltype_; }
  CellType GetNCellType() const { return ncelltype_; }
  double GetRescaleRatio() const { return rescale_ratio_; }
  bool GetNHasGaussian() const { return nhasgaussian_; }
  Eigen::Vector2d GetNormalMean() const { return nmean_; }
  Eigen::Matrix2d GetNormalCov() const { return ncov_; }
  Eigen::Vector2d GetNormalEvals() const { return nevals_; }
  Eigen::Matrix2d GetNormalEvecs() const { return nevecs_; }
  std::vector<Eigen::Vector2d> GetNormals() const { return normals_; }

  void SetPCellType(CellType pcelltype) { pcelltype_ = pcelltype; }
  void SetNCellType(CellType ncelltype) { ncelltype_ = ncelltype; }
  void SetRescaleRatio(double rescale_ratio) { rescale_ratio_ = rescale_ratio; }
  void SetNHasGaussian(bool nhasgaussian) { nhasgaussian_ = nhasgaussian; }
  void SetNormalMean(const Eigen::Vector2d &mean) { nmean_ = mean; }
  void SetNormalCov(const Eigen::Matrix2d &cov) { ncov_ = cov; }
  void SetNormalEvals(const Eigen::Vector2d &evals) { nevals_ = evals; }
  void SetNormalEvecs(const Eigen::Matrix2d &evecs) { nevecs_ = evecs; }
  void SetNormals(const std::vector<Eigen::Vector2d> &normals) {
    normals_ = normals;
  }

 private:
  /**
   * @brief Compute points gaussian
   */
  void ComputePGaussian();

  /**
   * @brief Compute normals gaussian
   */
  void ComputeNGaussian();
  void ComputeNGaussian2();
  CellType pcelltype_;     /**< Point cell type */
  CellType ncelltype_;     /**< Normal cell type */
  double rescale_ratio_;   /**< Rescale ratio */
  double tolerance_;       /**< Comparison tolerance */
  bool nhasgaussian_;      /**< Whether this cell has a valid normal gaussian */
  Eigen::Vector2d nmean_;  /**< Normal mean */
  Eigen::Vector2d nevals_; /**< Normal eigenvectors */
  Eigen::Matrix2d ncov_;   /**< Normal covariance */
  Eigen::Matrix2d nevecs_; /**< Normal eigenvectors */
  std::vector<Eigen::Vector2d> normals_; /**< Normals data */
};
