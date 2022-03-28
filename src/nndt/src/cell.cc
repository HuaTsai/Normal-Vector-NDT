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
      normal_(Eigen::Vector3d::Zero()),
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

  // XXX: omit n <= 6
  if (pts.size() <= 6) {
    celltype_ = kNoPoints;
    return;
  }

  if (!pts.size()) {
    celltype_ = kNoPoints;
    return;
  } else if (pts.size() == 1) {
    celltype_ = kPoint;
    return;
  } else if (pts.size() == 2) {
    celltype_ = kLine;
    return;
  }

  mean_ = ComputeMean(pts);
  cov_ = ComputeCov(pts, mean_) + ComputeMean(covs);
  ComputeEvalEvec(cov_, evals_, evecs_);

  if (evals_(1) <= tolerance_) {
    celltype_ = kLine;
    return;
  }

  if (evals_(0) <= tolerance_) {
    celltype_ = kPlane;
    return;
  }

  bool rescale = false;

  if (evals_(2) > rescale_ratio_ * evals_(0)) {
    evals_(0) = evals_(1) / rescale_ratio_;
    rescale = true;
  }

  if (evals_(2) > rescale_ratio_ * evals_(1)) {
    evals_(1) = evals_(1) / rescale_ratio_;
    rescale = true;
  }

  if (rescale)
    cov_ = evecs_ * evals_.asDiagonal() * evecs_.transpose();

  celltype_ = kRegular;
  normal_ = evecs_.col(0);
  if (mean_.dot(normal_) < 0) normal_ *= 1.;
  hasgaussian_ = true;
}
