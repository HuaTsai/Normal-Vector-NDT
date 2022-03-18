#include <nndt/cell.h>

Cell::Cell()
    : n_(0),
      hasgaussian_(false),
      skew_rad_(0),
      size_(0),
      center_(Eigen::Vector3d::Zero()),
      mean_(Eigen::Vector3d::Zero()),
      cov_(Eigen::Matrix3d::Zero()),
      evals_(Eigen::Vector3d::Zero()),
      evecs_(Eigen::Matrix3d::Zero()),
      celltype_(kNoInit),
      rescale_ratio_(100.),
      tolerance_(Eigen::NumTraits<double>::dummy_precision()) {}

void Cell::AddPoint(const Eigen::Vector2d &point) { points_.push_back(point); }

void Cell::AddPointWithCovariance(const Eigen::Vector2d &point,
                                  const Eigen::Matrix2d &covariance) {
  points_.push_back(point);
  point_covs_.push_back(covariance);
}

void Cell::ComputeGaussian() {
  if (point_covs_.size() && point_covs_.size() != points_.size()) {
    std::cerr << __FUNCTION__ << ": size of covariances is wrong\n";
    std::exit(-1);
  }

  n_ = points_.size();
  std::vector<Eigen::Vector2d> pts;
  std::vector<Eigen::Matrix2d> covs;
  if (point_covs_.size())
    ExcludeInfinite(points_, point_covs_, pts, covs);
  else
    ExcludeInfinite(points_, pts);

  if (!pts.size()) {
    celltype_ = kNoPoints;
    return;
  }

  pmean_ = ComputeMean(pts);
  pcov_ = ComputeCov(pts, pmean_, covs);
  celltype_ = kRegular;

  if (pcov_.isZero()) {
    celltype_ = kInvalid;
    return;
  }
  ComputeEvalEvec(pcov_, pevals_, pevecs_);

  if (pevals_(0) <= tolerance_ || pevals_(1) <= tolerance_) {
    celltype_ = kInvalid;
    return;
  }

  if (pevals_(1) > rescale_ratio_ * pevals_(0)) {
    pevals_(0) = pevals_(1) / rescale_ratio_;
    pcov_ = pevecs_ * pevals_.asDiagonal() * pevecs_.transpose();
    celltype_ = kRescale;
  }
  hasgaussian_ = true;
}
