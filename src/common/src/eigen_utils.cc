#include <angles/angles.h>
#include <common/angle_utils.h>
#include <common/eigen_utils.h>
#include <tf2_eigen/tf2_eigen.h>

Eigen::Affine3d Affine3dFromXYZRPY(const Eigen::Matrix<double, 6, 1> &xyzrpy) {
  Eigen::Affine3d ret = Eigen::Translation3d(xyzrpy(0), xyzrpy(1), xyzrpy(2)) *
                        Eigen::AngleAxisd(xyzrpy(5), Eigen::Vector3d::UnitZ()) *
                        Eigen::AngleAxisd(xyzrpy(4), Eigen::Vector3d::UnitY()) *
                        Eigen::AngleAxisd(xyzrpy(3), Eigen::Vector3d::UnitX());
  return ret;
}

Eigen::Affine3d Affine3dFromXYZRPY(const std::vector<double> &xyzrpy) {
  return Affine3dFromXYZRPY(
      Eigen::Map<const Eigen::Matrix<double, 6, 1>>(xyzrpy.data()));
}

Eigen::Matrix<double, 6, 1> XYZRPYFromAffine3d(const Eigen::Affine3d &mtx) {
  Eigen::Matrix<double, 6, 1> ret;
  ret.head(3) = mtx.translation();
  Eigen::Vector3d ypr = mtx.rotation().eulerAngles(2, 1, 0);
  double yaw = angles::normalize_angle(ypr(0));
  double pitch = angles::normalize_angle(ypr(1));
  double roll = angles::normalize_angle(ypr(2));
  if (fabs(pitch) > M_PI / 2) {
    roll = angles::normalize_angle(roll + M_PI);
    pitch = angles::normalize_angle(-pitch + M_PI);
    yaw = angles::normalize_angle(yaw + M_PI);
  }
  ret.tail(3) = Eigen::Vector3d(roll, pitch, yaw);
  return ret;
}

Eigen::Affine3d Affine3dFromAffine2d(const Eigen::Affine2d &aff) {
  double x = aff.translation()(0);
  double y = aff.translation()(1);
  double t = Eigen::Rotation2Dd(aff.rotation()).angle();
  return Eigen::Translation3d(x, y, 0) *
         Eigen::AngleAxisd(t, Eigen::Vector3d::UnitZ());
}

Eigen::Affine3d Conserve2DFromAffine3d(const Eigen::Affine3d &T) {
  auto xyzrpy = XYZRPYFromAffine3d(T);
  xyzrpy(2) = xyzrpy(3) = xyzrpy(4) = 0;
  return Affine3dFromXYZRPY(xyzrpy);
}

Eigen::Vector2d TransNormRotDegAbsFromAffine2d(const Eigen::Affine2d &aff) {
  Eigen::Vector2d ret;
  ret(0) = aff.translation().norm();
  ret(1) = abs(Rad2Deg(Eigen::Rotation2Dd(aff.rotation()).angle()));
  return ret;
}

Eigen::Vector2d TransNormRotDegAbsFromAffine3d(const Eigen::Affine3d &aff) {
  Eigen::Vector2d ret;
  ret(0) = aff.translation().norm();
  ret(1) = abs(Rad2Deg(Eigen::AngleAxisd(aff.rotation()).angle()));
  return ret;
}

geometry_msgs::PoseStamped MakePoseStampedMsg(const ros::Time &time,
                                              const Eigen::Affine3d &aff) {
  geometry_msgs::PoseStamped ret;
  ret.header.frame_id = "map";
  ret.header.stamp = time;
  ret.pose = tf2::toMsg(aff);
  return ret;
}

geometry_msgs::PoseStamped MakePoseStampedMsg(const ros::Time &time,
                                              const Eigen::Matrix4f &mtx) {
  return MakePoseStampedMsg(time, Eigen::Affine3d(mtx.cast<double>()));
}

Mvn::Mvn(const Eigen::Vector2d &mean, const Eigen::Matrix2d &covariance)
    : mean_(mean), covariance_(covariance) {}

double Mvn::pdf(const Eigen::Vector2d &x) const {
  double factor = 1 / (2 * M_PI * std::sqrt(covariance_.determinant()));
  double sqrmd = (x - mean_).dot(covariance_.inverse() * (x - mean_));
  return factor * std::exp(-0.5 * sqrmd);
}
