/**
 * @file ndt_cell.h
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Class Declaration of NDTCell
 * @version 0.1
 * @date 2021-07-29
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <sndt/cell_interface.h>

class NDTCell : public CellInterface {
 public:
  /**
   * @brief Cell types
   */
  enum CellType {
    kNoInit,   /**< Covariance is not computed yet */
    kNoPoints, /**< Covariance is not computed because of no points */
    kRegular,  /**< Covariance is computed well */
    kRescale,  /**< Covariance is rescaled */
    kInvalid   /**< Covariance is invalid */
  };

  /**
   * @brief Construct a new NDTCell object
   */
  NDTCell();

  /**
   * @brief Compute gaussian
   *
   * @pre The size of @c points and @c point_covs are the same
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

  CellType GetCellType() const { return celltype_; }
  double GetRescaleRatio() const { return rescale_ratio_; }

  void SetCellType(CellType celltype) { celltype_ = celltype; }
  void SetRescaleRatio(double rescale_ratio) { rescale_ratio_ = rescale_ratio; }

 private:
  CellType celltype_;    /**< Cell type */
  double rescale_ratio_; /**< Rescale ratio */
  double tolerance_;     /**< Comparison tolerance */
};
