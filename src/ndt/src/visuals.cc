#include <common/eigen_utils.h>
#include <ndt/visuals.h>
#include <ros/ros.h>
#include <tf2_eigen/tf2_eigen.h>

using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

namespace {
ros::Time GetROSTime() {
  ros::Time ret;
  try {
    ret = ros::Time::now();
  } catch (const ros::Exception &ex) {
    ret = ros::Time(0);
  }
  return ret;
}

MarkerArray Join(const std::vector<Marker> &markers) {
  MarkerArray ret;
  int id = 0;
  auto now = GetROSTime();
  for (const auto &m : markers) {
    ret.markers.push_back(m);
    ret.markers.back().header.stamp = now;
    ret.markers.back().id = id++;
  }
  return ret;
}

MarkerArray Join(const std::vector<MarkerArray> &markerarrays) {
  MarkerArray ret;
  int id = 0;
  auto now = GetROSTime();
  for (const auto &ma : markerarrays) {
    for (const auto &m : ma.markers) {
      ret.markers.push_back(m);
      ret.markers.back().header.stamp = now;
      ret.markers.back().id = id++;
    }
  }
  return ret;
}

MarkerArray Join(const std::vector<MarkerArray> &markerarrays,
                 const std::vector<Marker> &markers) {
  auto mas = markerarrays;
  mas.push_back(Join(markers));
  return Join(mas);
}

Marker InitMarker() {
  Marker ret;
  ret.header.frame_id = "map";
  ret.pose.orientation.w = 1;
  return ret;
}
}  // namespace

MarkerArray MarkerOfNDT(const NMap &map,
                        const std::unordered_set<MarkerOptions> &options,
                        const Eigen::Affine3d &T) {
  bool kred = options.count(MarkerOptions::kRed);
  bool kgreen = options.count(MarkerOptions::kGreen);
  bool kcell = options.count(MarkerOptions::kCell);
  bool kcov = options.count(MarkerOptions::kCov);

  MarkerArray ret;
  if (kcell) {
    Marker cells = InitMarker();
    cells.type = Marker::CUBE_LIST;
    cells.scale.x = cells.scale.y = cells.scale.z = map.GetCellSize();
    for (const auto &elem : map) {
      const auto &cell = elem.second;
      if (!cell.GetHasGaussian()) continue;
      auto cen = cell.GetCenter();
      geometry_msgs::Point c;
      c.x = cen(0), c.y = cen(1), c.z = cen(2);
      cells.points.push_back(c);
      std_msgs::ColorRGBA cl;
      cl.a = 0.2, cl.r = kred ? 1 : 0, cl.g = kgreen ? 1 : 0;
      cells.colors.push_back(cl);
    }
    ret = Join({ret}, {cells});
  }

  if (kcov) {
    std::vector<Marker> covs;
    for (const auto &cell : map.TransformCells(T)) {
      if (!cell.GetHasGaussian()) continue;
      Marker cov = InitMarker();
      cov.type = Marker::SPHERE;
      cov.color.a = 0.5;
      cov.color.r = kred ? 1 : 0, cov.color.g = kgreen ? 1 : 0;
      Eigen::Vector3d evals;
      Eigen::Matrix3d evecs;
      ComputeEvalEvec(cell.GetCov(), evals, evecs);
      Eigen::Quaterniond q(evecs);
      cov.scale.x = 2 * sqrt(evals(0));  // +- 1σ
      cov.scale.y = 2 * sqrt(evals(1));  // +- 1σ
      cov.scale.z = 2 * sqrt(evals(2));  // +- 1σ
      cov.pose.position.x = cell.GetMean()(0);
      cov.pose.position.y = cell.GetMean()(1);
      cov.pose.position.z = cell.GetMean()(2);
      cov.pose.orientation = tf2::toMsg(q);
      covs.push_back(cov);
    }
    ret = Join({ret}, covs);
  }

  return ret;
}

MarkerArray MarkerOfNDT(const std::shared_ptr<NMap> &map,
                        const std::unordered_set<MarkerOptions> &options,
                        const Eigen::Affine3d &T) {
  return MarkerOfNDT(*map, options, T);
}

MarkerArray MarkerOfCell(const Cell &cell,
                         const std::unordered_set<MarkerOptions> &options,
                         const Eigen::Affine3d &T) {
  bool kred = options.count(MarkerOptions::kRed);
  bool kgreen = options.count(MarkerOptions::kGreen);

  if (!cell.GetHasGaussian()) return MarkerArray();

  Marker bd = InitMarker();
  bd.type = Marker::CUBE;
  bd.scale.x = bd.scale.y = bd.scale.z = cell.GetSize();
  bd.pose.position.x = cell.GetCenter()(0);
  bd.pose.position.y = cell.GetCenter()(1);
  bd.pose.position.z = cell.GetCenter()(2);
  bd.color.a = 0.2, bd.color.r = kred ? 1 : 0, bd.color.g = kgreen ? 1 : 0;

  Marker cov = InitMarker();
  cov.type = Marker::SPHERE;
  cov.color.a = 0.5;
  cov.color.r = kred ? 1 : 0, cov.color.g = kgreen ? 1 : 0;
  Eigen::Vector3d evals;
  Eigen::Matrix3d evecs;
  ComputeEvalEvec(cell.GetCov(), evals, evecs);
  Eigen::Quaterniond q(evecs);
  cov.scale.x = 2 * sqrt(evals(0));  // +- 1σ
  cov.scale.y = 2 * sqrt(evals(1));  // +- 1σ
  cov.scale.z = 2 * sqrt(evals(2));  // +- 1σ
  cov.pose.position.x = cell.GetMean()(0);
  cov.pose.position.y = cell.GetMean()(1);
  cov.pose.position.z = cell.GetMean()(2);
  cov.pose.orientation = tf2::toMsg(q);

  return Join({bd, cov});
}

Marker MarkerOfPoints(const std::vector<Eigen::Vector3d> &points, bool red) {
  Marker ret = InitMarker();
  ret.header.stamp = GetROSTime();
  ret.type = Marker::SPHERE_LIST;
  ret.scale.x = ret.scale.y = ret.scale.z = 0.1;
  ret.color.a = 1, ret.color.r = red ? 1 : 0, ret.color.g = red ? 0 : 1;
  for (const auto &point : points) {
    geometry_msgs::Point pt;
    pt.x = point(0), pt.y = point(1), pt.z = point(2);
    ret.points.push_back(pt);
  }
  return ret;
}