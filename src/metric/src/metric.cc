#include <ceres/ceres.h>
#include <common/common.h>
#include <metric/metric.h>

template <typename T>
Eigen::Matrix<T, 2, 2> RotationMatrix2D(T yaw) {
  T cos_yaw = ceres::cos(yaw);
  T sin_yaw = ceres::sin(yaw);
  Eigen::Matrix<T, 2, 2> ret;
  ret << cos_yaw, -sin_yaw, sin_yaw, cos_yaw;
  return ret;
}

struct CostFunctor {
  CostFunctor(const Eigen::Vector2d &p, const Eigen::Vector2d &q)
      : p_(p), q_(q) {}

  template <typename T>
  bool operator()(const T *const xyt, T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(xyt[2]);
    Eigen::Matrix<T, 2, 1> t(xyt[0], xyt[1]);
    Eigen::Matrix<T, 2, 1> p = R * p_.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> q = q_.cast<T>();
    e[0] = (p - q)(0);
    e[1] = (p - q)(1);
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &p,
                                     const Eigen::Vector2d &q) {
    return new ceres::AutoDiffCostFunction<CostFunctor, 2, 3>(
        new CostFunctor(p, q));
  }

  const Eigen::Vector2d p_, q_;
};

nav_msgs::Path GetInitPath() {
  nav_msgs::Path ret;
  ret.header.frame_id = "map";
  try {
    ret.header.stamp = ros::Time::now();
  } catch (const ros::Exception &ex) {
    std::cerr << ex.what() << std::endl;
    std::cerr << "Set stamp to ros::Time(0)" << std::endl;
    ret.header.stamp = ros::Time(0);
  }
  return ret;
}

nav_msgs::Path AlignPoses2D(const nav_msgs::Path &gt,
                            const nav_msgs::Path &est) {
  int n = gt.poses.size();
  double xyt[3] = {0, 0, 0};
  ceres::Problem problem;
  for (int i = 0; i < n; ++i) {
    Eigen::Vector2d p(est.poses[i].pose.position.x,
                      est.poses[i].pose.position.y);
    Eigen::Vector2d q(gt.poses[i].pose.position.x, gt.poses[i].pose.position.y);
    problem.AddResidualBlock(CostFunctor::Create(p, q), nullptr, xyt);
  }
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  nav_msgs::Path ret = GetInitPath();
  for (const auto &p : est.poses) {
    Eigen::Affine3d aff;
    tf2::fromMsg(p.pose, aff);
    Eigen::Affine3d aff2 = Eigen::Translation3d(xyt[0], xyt[1], 0) *
                           Eigen::AngleAxisd(xyt[2], Eigen::Vector3d::UnitZ()) *
                           aff;
    geometry_msgs::PoseStamped pst;
    pst.header = p.header;
    pst.pose = tf2::toMsg(aff2);
    ret.poses.push_back(pst);
  }
  return ret;
}

bool IsValidTime(const nav_msgs::Path &est, const nav_msgs::Path &gt) {
  return est.poses.front().header.stamp >= gt.poses.front().header.stamp &&
         est.poses.back().header.stamp <= gt.poses.back().header.stamp;
}

TrajectoryEvaluation::TrajectoryEvaluation() : evaltype_(kDummy), length_(-1) {}

std::pair<Stat, Stat> TrajectoryEvaluation::ComputeRMSError2D() {
  if (evaltype_ == kDummy) {
    std::cerr << __FUNCTION__ << ": Invalid evaluation type" << std::endl;
    std::exit(1);
  }
  if (!estpath_.poses.size() || !gtpath_.poses.size()) {
    std::cerr << __FUNCTION__ << ": Invalid pose size" << std::endl;
    std::exit(1);
  }
  if (!IsValidTime(estpath_, gtpath_)) {
    std::cerr << __FUNCTION__ << ": Invalid time" << std::endl;
    std::exit(1);
  }
  if (evaltype_ == kRelativeByLength && length_ == -1) {
    std::cerr << __FUNCTION__ << ": No length is set" << std::endl;
    std::exit(1);
  }

  gtsync_ = GetInitPath();
  for (auto p : estpath_.poses) {
    geometry_msgs::PoseStamped pst;
    pst.header = p.header;
    pst.pose = GetPose(gtpath_.poses, p.header.stamp);
    gtsync_.poses.push_back(pst);
  }

  gtlength_ = 0;
  for (size_t i = 0; i < gtsync_.poses.size() - 1; ++i) {
    Eigen::Vector3d p1, p2;
    tf2::fromMsg(estpath_.poses[i].pose.position, p1);
    tf2::fromMsg(estpath_.poses[i + 1].pose.position, p2);
    gtlength_ += (p1 - p2).norm();
  }

  if (evaltype_ == kAbsolute) {
    return AbsoluteTrajectoryError();
  } else if (evaltype_ == kRelativeBySingle) {
    return RelativePoseErrorBySingle();
  } else if (evaltype_ == kRelativeByLength) {
    return RelativePoseErrorByLength();
  }
  return {Stat({}), Stat({})};
}

std::pair<Stat, Stat> TrajectoryEvaluation::AbsoluteTrajectoryError() {
  align_estpath_ = AlignPoses2D(gtsync_, estpath_);
  int n = estpath_.poses.size();
  std::vector<double> tlerr, roterr;
  for (int i = 0; i < n; ++i) {
    Eigen::Affine3d SPi, Qi;
    tf2::fromMsg(align_estpath_.poses[i].pose, SPi);
    tf2::fromMsg(gtsync_.poses[i].pose, Qi);
    Eigen::Affine3d Fi = Qi.inverse() * SPi;
    tlerr.push_back(Fi.translation().norm());
    roterr.push_back(Rad2Deg(Eigen::AngleAxisd(Fi.rotation()).angle()));
  }
  return {Stat(tlerr), Stat(roterr)};
}

std::pair<Stat, Stat>
TrajectoryEvaluation::RelativePoseErrorBySingle() {
  int n = estpath_.poses.size();
  std::vector<double> tlerr, roterr;
  for (int i = 0; i < n - 1; ++i) {
    Eigen::Affine3d Qi, Qj, Pi, Pj;
    tf2::fromMsg(estpath_.poses[i].pose, Pi);
    tf2::fromMsg(estpath_.poses[i + 1].pose, Pj);
    tf2::fromMsg(gtsync_.poses[i].pose, Qi);
    tf2::fromMsg(gtsync_.poses[i + 1].pose, Qj);
    Eigen::Affine3d Ei = (Qi.inverse() * Qj).inverse() * (Pi.inverse() * Pj);
    tlerr.push_back(Ei.translation().norm());
    roterr.push_back(Rad2Deg(Eigen::AngleAxisd(Ei.rotation()).angle()));
  }
  return {Stat(tlerr), Stat(roterr)};
}

std::pair<Stat, Stat>
TrajectoryEvaluation::RelativePoseErrorByLength() {
  int n = estpath_.poses.size();
  std::vector<double> dists;
  for (int i = 0; i < n - 1; ++i) {
    Eigen::Vector3d p1, p2;
    tf2::fromMsg(estpath_.poses[i].pose.position, p1);
    tf2::fromMsg(estpath_.poses[i + 1].pose.position, p2);
    dists.push_back((p1 - p2).norm());
  }

  std::vector<double> tlerr, roterr;
  double distsum = 0;
  for (int sid = 0, eid = 0; sid < n - 1; ++sid) {
    while (eid < n - 1 && distsum < length_) {
      distsum += dists[eid++];
    }
    if (distsum < length_) break;

    Eigen::Affine3d Qi, Qj, Pi, Pj;
    tf2::fromMsg(estpath_.poses[sid].pose, Pi);
    tf2::fromMsg(estpath_.poses[eid - 1].pose, Pj);
    tf2::fromMsg(gtsync_.poses[sid].pose, Qi);
    tf2::fromMsg(gtsync_.poses[eid - 1].pose, Qj);
    Eigen::Affine3d Ei = (Qi.inverse() * Qj).inverse() * (Pi.inverse() * Pj);
    tlerr.push_back(Ei.translation().norm());
    roterr.push_back(Rad2Deg(Eigen::AngleAxisd(Ei.rotation()).angle()));

    distsum -= dists[sid];
  }
  return {Stat(tlerr), Stat(roterr)};
}
