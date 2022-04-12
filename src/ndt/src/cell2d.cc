#include <common/eigen_utils.h>
#include <ndt/cell2d.h>

Cell2D::Cell2D()
    : n_(0),
      hasgaussian_(false),
      size_(0),
      center_(Eigen::Vector2d::Zero()),
      mean_(Eigen::Vector2d::Zero()),
      cov_(Eigen::Matrix2d::Zero()),
      evals_(Eigen::Vector2d::Zero()),
      evecs_(Eigen::Matrix2d::Zero()),
      normal_(Eigen::Vector2d::Zero()),
      celltype_(CellType::kNoInit),
      rescale_ratio_(100.) {}

void Cell2D::AddPoint(const Eigen::Vector2d &point) {
  points_.push_back(point);
}

void Cell2D::AddPointWithCovariance(const Eigen::Vector2d &point,
                                    const Eigen::Matrix2d &covariance) {
  points_.push_back(point);
  point_covs_.push_back(covariance);
}

void Cell2D::ComputeGaussian() {
  mean_.setZero();
  cov_.setZero();
  if (point_covs_.size())
    ExcludeInfiniteInPlace(points_, point_covs_);
  else
    ExcludeInfiniteInPlace(points_);
  n_ = points_.size();

  // XXX: The decision of few points
  if (!point_covs_.size() && points_.size() < 3) {
    celltype_ = CellType::kFewPoints;
    return;
  }

  if (point_covs_.size() && points_.size() <= 1) {
    celltype_ = CellType::kFewPoints;
    return;
  }

  mean_ = ComputeMean(points_);
  cov_ = ComputeCov(points_, mean_) + ComputeMean(point_covs_);
  ComputeEvalEvec(cov_, evals_, evecs_);

  // XXX: Not sure whether omit line, but I think it should be fine.
  if (evals_(0) <= Eigen::NumTraits<double>::dummy_precision()) {
    celltype_ = CellType::kLine;
    return;
  }

  if (evals_(1) > rescale_ratio_ * evals_(0)) {
    evals_(0) = evals_(1) / rescale_ratio_;
    cov_ = evecs_ * evals_.asDiagonal() * evecs_.transpose();
  }

  celltype_ = CellType::kRegular;
  normal_ = evecs_.col(0);
  if (mean_.dot(normal_) < 0) normal_ *= -1.;
  hasgaussian_ = true;
}
