#include <common/angle_utils.h>
#include <ndt/opt.h>

Optimizer::Optimizer(OptType type)
    : type_(type),
      gproblem_(nullptr),
      cur_tf_(Eigen::Affine3d::Identity()),
      threshold_(0.001),
      threshold_ang_(0.01) {
  memset(xyzxyzw_, 0, sizeof(xyzxyzw_));
  xyzxyzw_[6] = 1;
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
  if (type_ == kLS) {
    ceres::GradientProblemSolver::Options options;
    ceres::GradientProblemSolver::Summary summary;
    options.max_num_iterations = 50;
    ceres::Solve(options, *gproblem_, xyzxyzw_, &summary);
  } else if (type_ == kTR) {
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
  cur_tf_ = cur_tf_ * dtf;
}

bool Optimizer::CheckConverge(const std::vector<Eigen::Affine3d> &tfs) {
  double tl = Eigen::Map<Eigen::Vector3d>(xyzxyzw_).norm();
  double ang = Eigen::AngleAxisd(Eigen::Quaterniond(xyzxyzw_ + 3)).angle();
  if (tl < threshold_ && Rad2Deg(ang) < threshold_ang_) return true;
  for (auto tf : tfs) {
    Eigen::Affine3d d = tf.inverse() * cur_tf_;
    double dt = d.translation().norm();
    double dr = Eigen::AngleAxisd(d.linear()).angle();
    if (dt < threshold_ && Rad2Deg(dr) < threshold_ang_) return true;
  }
  return false;
}
