#include <common/eigen_utils.h>
#include <nndt/cell.h>

Cell::Cell()
    : n_(0),
      hasgaussian_(false),
      size_(0),
      center_(Eigen::Vector3d::Zero()),
      mean_(Eigen::Vector3d::Zero()),
      cov_(Eigen::Matrix3d::Zero()),
      evals_(Eigen::Vector3d::Zero()),
      evecs_(Eigen::Matrix3d::Zero()),
      celltype_(kNoInit),
      rescale_ratio_(100.),
      tolerance_(Eigen::NumTraits<double>::dummy_precision()) {}

void Cell::AddPoint(const Eigen::Vector3d &point) { points_.push_back(point); }

void Cell::AddPointWithCovariance(const Eigen::Vector3d &point,
                                  const Eigen::Matrix3d &covariance) {
  points_.push_back(point);
  point_covs_.push_back(covariance);
}

void Cell::ComputeGaussian() {
  mean_.setZero();
  cov_.setZero();
  n_ = points_.size();
  std::vector<Eigen::Vector3d> pts;
  std::vector<Eigen::Matrix3d> covs;
  if (point_covs_.size())
    ExcludeInfinite(points_, point_covs_, pts, covs);
  else
    ExcludeInfinite(points_, pts);

  if (!pts.size()) {
    celltype_ = kNoPoints;
    return;
  }

  mean_ = ComputeMean(pts);
  cov_ = ComputeCov(pts, mean_) + ComputeMean(covs);
  celltype_ = kRegular;
  ComputeEvalEvec(cov_, evals_, evecs_);

  // XXX: Do we need to do this or just to rescale it?
  if (evals_(0) <= tolerance_ || evals_(1) <= tolerance_ ||
      evals_(2) <= tolerance_) {
    celltype_ = kInvalid;
    return;
  }

  if (evals_(2) > rescale_ratio_ * evals_(0)) {
    evals_(0) = evals_(1) / rescale_ratio_;
    celltype_ = kRescale;
  }

  if (evals_(2) > rescale_ratio_ * evals_(1)) {
    evals_(1) = evals_(1) / rescale_ratio_;
    celltype_ = kRescale;
  }

  if (celltype_ == kRescale)
    cov_ = evecs_ * evals_.asDiagonal() * evecs_.transpose();

  hasgaussian_ = true;
}

// TODO: Check whether it is okay.
bool Cell::Normal(Eigen::Vector3d &normal) const {
  if (celltype_ != kRegular) return false;
  normal = evecs_.col(0);
  if (mean_.dot(normal) < 0) normal *= -1.;
  return true;
}
