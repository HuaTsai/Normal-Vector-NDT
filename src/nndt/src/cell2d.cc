#include <common/eigen_utils.h>
#include <nndt/cell2d.h>

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
      celltype_(kNoInit),
      rescale_ratio_(100.),
      tolerance_(Eigen::NumTraits<double>::dummy_precision()) {}

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
  n_ = points_.size();
  std::vector<Eigen::Vector2d> pts;
  std::vector<Eigen::Matrix2d> covs;
  if (point_covs_.size())
    ExcludeInfinite(points_, point_covs_, pts, covs);
  else
    ExcludeInfinite(points_, pts);

  // XXX: omit n <= 4
  if (pts.size() <= 4) {
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

  if (evals_(1) <= Eigen::NumTraits<double>::dummy_precision()) {
    celltype_ = kLine;
    return;
  }

  if (evals_(0) <= Eigen::NumTraits<double>::dummy_precision()) {
    celltype_ = kPlane;
    return;
  }

  bool rescale = false;

  if (evals_(2) > rescale_ratio_ * evals_(0)) {
    evals_(0) = evals_(1) / rescale_ratio_;
    rescale = true;
  }

  if (rescale) cov_ = evecs_ * evals_.asDiagonal() * evecs_.transpose();

  celltype_ = kRegular;
  normal_ = evecs_.col(0);
  if (mean_.dot(normal_) < 0) normal_ *= 1.;
  hasgaussian_ = true;
}
