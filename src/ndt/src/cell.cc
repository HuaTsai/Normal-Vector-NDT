#include <common/eigen_utils.h>
#include <ndt/cell.h>

Cell::Cell()
    : n_(0),
      hasgaussian_(false),
      size_(0),
      center_(Eigen::Vector3d::Zero()),
      mean_(Eigen::Vector3d::Zero()),
      cov_(Eigen::Matrix3d::Zero()),
      evals_(Eigen::Vector3d::Zero()),
      evecs_(Eigen::Matrix3d::Zero()),
      normal_(Eigen::Vector3d::Zero()),
      celltype_(CellType::kNoInit),
      rescale_ratio_(100.) {}

void Cell::AddPoint(const Eigen::Vector3d &point) { points_.push_back(point); }

void Cell::AddPointWithCovariance(const Eigen::Vector3d &point,
                                  const Eigen::Matrix3d &covariance) {
  points_.push_back(point);
  point_covs_.push_back(covariance);
}

void Cell::ComputeGaussian() {
  mean_.setZero();
  cov_.setZero();
  if (point_covs_.size())
    ExcludeInfiniteInPlace(points_, point_covs_);
  else
    ExcludeInfiniteInPlace(points_);
  n_ = points_.size();

  // XXX: The decision of few points
  if (!point_covs_.size() && points_.size() < 6) {
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

  if (evals_(1) <= Eigen::NumTraits<double>::dummy_precision()) {
    celltype_ = CellType::kLine;
    return;
  }

  // XXX: Not sure whether omit plane, but I think it should be fine.
  if (evals_(0) <= Eigen::NumTraits<double>::dummy_precision()) {
    celltype_ = CellType::kPlane;
    return;
  }

  bool rescale = false;

  if (evals_(2) > rescale_ratio_ * evals_(0)) {
    evals_(0) = evals_(2) / rescale_ratio_;
    rescale = true;
  }

  if (evals_(2) > rescale_ratio_ * evals_(1)) {
    evals_(1) = evals_(2) / rescale_ratio_;
    rescale = true;
  }

  if (rescale) cov_ = evecs_ * evals_.asDiagonal() * evecs_.transpose();

  celltype_ = CellType::kRegular;
  normal_ = evecs_.col(0);
  if (mean_.dot(normal_) < 0) normal_ *= -1.;
  hasgaussian_ = true;
}
