#include <common/angle_utils.h>
#include <common/eigen_utils.h>
#include <lbfgs/LBFGSB.h>
#include <ndt/opt.h>

class Manifold2D {
 public:
  template <typename T>
  bool operator()(const T *x, const T *dx, T *res) const {
    res[0] = x[0] + dx[0];
    res[1] = x[1] + dx[1];
    res[2] =
        x[2] + dx[2] -
        T(2. * M_PI) * ceres::floor((x[2] + dx[2] + T(M_PI)) / T(2. * M_PI));
    return true;
  }
  static ceres::LocalParameterization *Create() {
    return new ceres::AutoDiffLocalParameterization<Manifold2D, 3, 3>;
  }
};

Optimizer::Optimizer(Options type)
    : type_(type),
      problem_(nullptr),
      param_(nullptr),
      cur_tf3_(Eigen::Affine3d::Identity()),
      cur_tf2_(Eigen::Affine2d::Identity()),
      threshold_(0.001),
      threshold_ang_(0.01),
      tlang_(Eigen::Vector2d::Zero()) {
  memset(xyzrpy_, 0, sizeof(xyzrpy_));
  memset(xyzxyzw_, 0, sizeof(xyzxyzw_));
  memset(xyt_, 0, sizeof(xyt_));
  xyzxyzw_[6] = 1;
  if (type_ == Options::kOptimizer2D) {
    param_ = Manifold2D::Create();
  } else if (type_ == Options::kOptimizer3D) {
    param_ = new ceres::ProductParameterization(
        new ceres::IdentityParameterization(3),
        new ceres::EigenQuaternionParameterization());
  } else if (type_ == Options::kLBFGSPP) {
    // nop
  } else if (type_ == Options::kAnalytic) {
    // nop
  } else {
    std::cerr << __FUNCTION__ << ": invalid type\n";
    std::exit(1);
  }
}

Optimizer::~Optimizer() {
  if (problem_) delete problem_;
  if (costobj_) delete costobj_;
}

void Optimizer::BuildProblem(ceres::FirstOrderFunction *func) {
  if (type_ == Options::kAnalytic)
    problem_ = new ceres::GradientProblem(func);
  else
    problem_ = new ceres::GradientProblem(func, param_);
}

void Optimizer::BuildProblem(CostObj *func) { costobj_ = func; }

void Optimizer::Optimize() {
  ceres::GradientProblemSolver::Options options;
  ceres::GradientProblemSolver::Summary summary;
  options.max_num_iterations = 50;
  if (type_ == Options::kOptimizer2D) {
    ceres::Solve(options, *problem_, xyt_, &summary);
    Eigen::Affine2d dtf =
        Eigen::Translation2d(xyt_[0], xyt_[1]) * Eigen::Rotation2Dd(xyt_[2]);
    cur_tf2_ = dtf * cur_tf2_;
    tlang_ = TransNormRotDegAbsFromAffine2d(dtf);
  } else if (type_ == Options::kOptimizer3D) {
    ceres::Solve(options, *problem_, xyzxyzw_, &summary);
    Eigen::Affine3d dtf;
    dtf.translation() = Eigen::Map<Eigen::Vector3d>(xyzxyzw_);
    dtf.linear() = Eigen::Map<Eigen::Quaterniond>(xyzxyzw_ + 3).matrix();
    cur_tf3_ = dtf * cur_tf3_;
    tlang_ = TransNormRotDegAbsFromAffine3d(dtf);
    std::cerr << tlang_.transpose() << std::endl;
  } else if (type_ == Options::kAnalytic) {
    ceres::Solve(options, *problem_, xyzrpy_, &summary);
    Eigen::Affine3d dtf = Eigen::Translation3d(xyzrpy_[0], xyzrpy_[1], xyzrpy_[2]) *
                          Eigen::AngleAxisd(xyzrpy_[3], Eigen::Vector3d::UnitX()) *
                          Eigen::AngleAxisd(xyzrpy_[4], Eigen::Vector3d::UnitY()) *
                          Eigen::AngleAxisd(xyzrpy_[5], Eigen::Vector3d::UnitZ());
    cur_tf3_ = dtf * cur_tf3_;
    tlang_ = TransNormRotDegAbsFromAffine3d(dtf);
  } else if (type_ == Options::kLBFGSPP) {
    LBFGSpp::LBFGSBParam<double> param;
    LBFGSpp::LBFGSBSolver<double> solver(param);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd lb = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd ub = Eigen::VectorXd::Zero(6);
    // HACK: Bound
    double angle = Deg2Rad(45);
    ub << 5, 5, 5, angle, angle, angle;
    lb = -ub;
    double fx;
    solver.minimize(*costobj_, x, fx, lb, ub);
    Eigen::Affine3d dtf = Eigen::Translation3d(x.head(3)) *
                          Eigen::AngleAxisd(x(3), Eigen::Vector3d::UnitX()) *
                          Eigen::AngleAxisd(x(4), Eigen::Vector3d::UnitY()) *
                          Eigen::AngleAxisd(x(5), Eigen::Vector3d::UnitZ());
    cur_tf3_ = dtf * cur_tf3_;
    tlang_ = TransNormRotDegAbsFromAffine3d(dtf);
  }
}

bool Optimizer::CheckConverge(const std::vector<Eigen::Affine3d> &tfs) {
  std::cerr << "Checking" << std::endl;
  if (tlang_(0) < threshold_ && tlang_(1) < threshold_ang_) return true;
  for (auto tf : tfs) {
    auto dtf = TransNormRotDegAbsFromAffine3d(tf.inverse() * cur_tf3_);
    if (dtf(0) < threshold_ && dtf(1) < threshold_ang_) return true;
  }
  std::cerr << "Not Converge" << std::endl;
  return false;
}

bool Optimizer::CheckConverge(const std::vector<Eigen::Affine2d> &tfs) {
  if (tlang_(0) < threshold_ && tlang_(1) < threshold_ang_) return true;
  for (auto tf : tfs) {
    auto dtf = TransNormRotDegAbsFromAffine2d(tf.inverse() * cur_tf2_);
    if (dtf(0) < threshold_ && dtf(1) < threshold_ang_) return true;
  }
  return false;
}
