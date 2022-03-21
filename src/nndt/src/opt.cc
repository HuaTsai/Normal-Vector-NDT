#include <nndt/opt.h>

Optimizer::Optimizer()
    : problem_(nullptr),
      cur_tf_(Eigen::Affine3d::Identity()),
      threshold_(0.001),
      iteration_(0),
      max_iterations_(100) {
  memset(xyzxyzw_, 0, sizeof(xyzxyzw_));
  xyzxyzw_[6] = 1;
}

Optimizer::~Optimizer() {
  if (problem_) delete problem_;
}

void Optimizer::BuildProblem(ceres::FirstOrderFunction *func) {
  param_ = new ceres::ProductParameterization(
      new ceres::IdentityParameterization(3),
      new ceres::EigenQuaternionParameterization());
  problem_ = new ceres::GradientProblem(func, param_);
}

void Optimizer::Optimize() {
  ceres::GradientProblemSolver::Options options;
  ceres::GradientProblemSolver::Summary summary;
  options.max_num_iterations = 50;
  ++iteration_;
  ceres::Solve(options, *problem_, xyzxyzw_, &summary);
  Eigen::Affine3d dtf;
  dtf.translation() = Eigen::Map<Eigen::Vector3d>(xyzxyzw_);
  dtf.linear() = Eigen::Map<Eigen::Quaterniond>(xyzxyzw_ + 3).matrix();
  cur_tf_ = cur_tf_ * dtf;
}

bool Optimizer::CheckConverge() {
  double tl = Eigen::Map<Eigen::Vector3d>(xyzxyzw_).norm();
  if (tl < threshold_) {
    return true;
  }
  if (iteration_ > max_iterations_) {
    return true;
  }
  return false;
}
