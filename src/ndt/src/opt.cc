#include <common/angle_utils.h>
#include <common/eigen_utils.h>
#include <ndt/opt.h>

Optimizer::Optimizer(Options type)
    : type_(type),
      gproblem_(nullptr),
      cur_tf_(Eigen::Affine3d::Identity()),
      threshold_(0.001),
      threshold_ang_(0.01) {
  memset(xyzxyzw_, 0, sizeof(xyzxyzw_));
  xyzxyzw_[6] = 1;
  if (type_ != Options::kLineSearch && type_ != Options::kTrustRegion) {
    std::cerr << __FUNCTION__ << ": invalid type\n";
    std::exit(1);
  }
}

Optimizer::~Optimizer() {
  if (gproblem_) delete gproblem_;
}

void Optimizer::AddResidualBlock(ceres::CostFunction *func) {
  problem_.AddResidualBlock(func, nullptr, xyzxyzw_);
}

void Optimizer::BuildProblem(ceres::FirstOrderFunction *func) {
  param_ = new ceres::ProductParameterization(
      new ceres::IdentityParameterization(3),
      new ceres::EigenQuaternionParameterization());
  gproblem_ = new ceres::GradientProblem(func, param_);
}

void Optimizer::Optimize() {
  if (type_ == Options::kLineSearch) {
    ceres::GradientProblemSolver::Options options;
    ceres::GradientProblemSolver::Summary summary;
    options.max_num_iterations = 50;
    ceres::Solve(options, *gproblem_, xyzxyzw_, &summary);
  } else if (type_ == Options::kTrustRegion) {
    param_ = new ceres::ProductParameterization(
        new ceres::IdentityParameterization(3),
        new ceres::EigenQuaternionParameterization());
    problem_.SetParameterization(xyzxyzw_, param_);
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.max_num_iterations = 50;
    ceres::Solve(options, &problem_, &summary);
  }
  Eigen::Affine3d dtf;
  dtf.translation() = Eigen::Map<Eigen::Vector3d>(xyzxyzw_);
  dtf.linear() = Eigen::Map<Eigen::Quaterniond>(xyzxyzw_ + 3).matrix();
  // BUG: order????
  // cur_tf_ = cur_tf_ * dtf;
  cur_tf_ = dtf * cur_tf_;
}

bool Optimizer::CheckConverge(const std::vector<Eigen::Affine3d> &tfs) {
  double tl = Eigen::Map<Eigen::Vector3d>(xyzxyzw_).norm();
  double ang =
      std::abs(Eigen::AngleAxisd(Eigen::Quaterniond(xyzxyzw_ + 3)).angle());
  if (tl < threshold_ && Rad2Deg(ang) < threshold_ang_) return true;
  for (auto tf : tfs) {
    auto dtf = TransNormRotDegAbsFromAffine3d(tf.inverse() * cur_tf_);
    if (dtf(0) < threshold_ && dtf(1) < threshold_ang_) return true;
  }
  return false;
}

Optimizer2D::Optimizer2D(Options type)
    : type_(type),
      gproblem_(nullptr),
      cur_tf_(Eigen::Affine2d::Identity()),
      threshold_(0.001),
      threshold_ang_(0.01) {
  memset(xyt_, 0, sizeof(xyt_));
  if (type_ != Options::kLineSearch && type_ != Options::kTrustRegion) {
    std::cerr << __FUNCTION__ << ": invalid type\n";
    std::exit(1);
  }
}

Optimizer2D::~Optimizer2D() {
  if (gproblem_) delete gproblem_;
}

void Optimizer2D::AddResidualBlock(ceres::CostFunction *func) {
  problem_.AddResidualBlock(func, nullptr, xyt_);
}

void Optimizer2D::BuildProblem(ceres::FirstOrderFunction *func) {
  gproblem_ = new ceres::GradientProblem(func);
}

void Optimizer2D::Optimize() {
  if (type_ == Options::kLineSearch) {
    ceres::GradientProblemSolver::Options options;
    ceres::GradientProblemSolver::Summary summary;
    options.max_num_iterations = 50;
    ceres::Solve(options, *gproblem_, xyt_, &summary);
  } else if (type_ == Options::kTrustRegion) {
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.max_num_iterations = 50;
    ceres::Solve(options, &problem_, &summary);
  }
  cur_tf_ = Eigen::Translation2d(xyt_[0], xyt_[1]) *
            Eigen::Rotation2Dd(xyt_[2]) * cur_tf_;
}

bool Optimizer2D::CheckConverge(const std::vector<Eigen::Affine2d> &tfs) {
  double tl = Eigen::Map<Eigen::Vector2d>(xyt_).norm();
  double ang = std::abs(xyt_[2]);
  if (tl < threshold_ && Rad2Deg(ang) < threshold_ang_) return true;
  for (auto tf : tfs) {
    auto dtf = TransNormRotDegAbsFromAffine2d(tf.inverse() * cur_tf_);
    if (dtf(0) < threshold_ && dtf(1) < threshold_ang_) return true;
  }
  return false;
}
