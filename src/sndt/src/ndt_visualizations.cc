#include "sndt/ndt_visualizations.h"

#include <bits/stdc++.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <Eigen/Eigen>
#include <gsl/gsl>

using namespace std;
using namespace Eigen;
using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

std_msgs::ColorRGBA MakeColorRGBA(const Color &color, double alpha) {
  std_msgs::ColorRGBA ret;
  ret.a = alpha;
  ret.r = ret.g = ret.b = 0.;
  if (color == Color::kRed) {
    ret.r = 1.;
  } else if (color == Color::kLime) {
    ret.g = 1.;
  } else if (color == Color::kBlue) {
    ret.b = 1.;
  } else if (color == Color::kWhite) {
    ret.a = 0.;
  } else if (color == Color::kBlack) {
    // nop
  } else if (color == Color::kGray) {
    ret.r = ret.g = ret.b = 0.5;
  } else if (color == Color::kYellow) {
    ret.r = ret.g = 1.;
  } else if (color == Color::kAqua) {
    ret.g = ret.b = 1.;
  } else if (color == Color::kFuchsia) {
    ret.r = ret.b = 1.;
  }
  return ret;
}

vector<Vector2d> FindTangentPoints(const Marker &eclipse,
                                   const Vector2d &point) {
  Affine3d aff3;
  tf2::fromMsg(eclipse.pose, aff3);
  Matrix3d mtx = Matrix3d::Identity();
  mtx.block<2, 2>(0, 0) = aff3.rotation().block<2, 2>(0, 0);
  mtx.block<2, 1>(0, 2) = aff3.translation().block<2, 1>(0, 0);
  Affine2d aff2(mtx);
  auto point2 = aff2.inverse() * point;
  auto rx2 = (eclipse.scale.x / 2) * (eclipse.scale.x / 2);
  auto ry2 = (eclipse.scale.y / 2) * (eclipse.scale.y / 2);
  auto x0 = point2(0), x02 = x0 * x0;
  auto y0 = point2(1), y02 = y0 * y0;
  vector<Vector2d> sols(2);
  if (x02 == rx2) {
    auto msol =
        (-rx2 * ry2 * ry2 + rx2 * ry2 * y02) / (2 * rx2 * ry2 * x0 * y0);
    sols[0](0) = (msol * rx2 * (msol * x0 - y0)) / (msol * msol * rx2 + ry2);
    sols[0](1) = y0 + msol * (sols[0](0) - x0);
    sols[1](0) = x0;
    sols[1](1) = 0;
  } else {
    auto msol1 =
        (-x0 * y0 + sqrt(-rx2 * ry2 + rx2 * y02 + ry2 * x02)) / (rx2 - x02);
    sols[0](0) =
        (msol1 * rx2 * (msol1 * x0 - y0)) / (msol1 * msol1 * rx2 + ry2);
    sols[0](1) = y0 + msol1 * (sols[0](0) - x0);
    auto msol2 =
        (-x0 * y0 - sqrt(-rx2 * ry2 + rx2 * y02 + ry2 * x02)) / (rx2 - x02);
    sols[1](0) =
        (msol2 * rx2 * (msol2 * x0 - y0)) / (msol2 * msol2 * rx2 + ry2);
    sols[1](1) = y0 + msol2 * (sols[1](0) - x0);
  }
  for (auto &sol : sols) sol = aff2 * sol;
  return sols;
}

void UpdateMarkerArray(MarkerArray &markerarray, Marker marker) {
  if (!markerarray.markers.size()) {
    markerarray.markers.push_back(marker);
    return;
  }
  marker.header.frame_id = "map";
  marker.header.stamp = markerarray.markers.back().header.stamp;
  marker.id = markerarray.markers.back().id + 1;
  markerarray.markers.push_back(marker);
}

MarkerArray JoinMarkers(const vector<Marker> &ms) {
  MarkerArray ret;
  int id = 0;
  auto now = ros::Time::now();
  for (const auto &m : ms) {
    ret.markers.push_back(m);
    ret.markers.back().header.stamp = now;
    ret.markers.back().id = id++;
  }
  return ret;
}

MarkerArray JoinMarkerArraysAndMarkers(const vector<MarkerArray> &mas,
                                       const vector<Marker> &ms) {
  MarkerArray ret;
  int id = 0;
  auto now = ros::Time::now();
  for (const auto &ma : mas) {
    for (const auto &m : ma.markers) {
      ret.markers.push_back(m);
      ret.markers.back().header.stamp = now;
      ret.markers.back().id = id++;
    }
  }
  for (const auto &m : ms) {
    ret.markers.push_back(m);
    ret.markers.back().header.stamp = now;
    ret.markers.back().id = id++;
  }
  return ret;
}

Marker MarkerOfBoundary(const Vector2d &center, double size,
                        double skew_rad,
                        const Color &color) {
  Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = ros::Time::now();
  ret.id = 0;
  ret.type = Marker::LINE_LIST;
  ret.action = Marker::ADD;
  ret.pose = tf2::toMsg(Affine3d::Identity());
  ret.scale.x = 0.02;
  ret.color = MakeColorRGBA(color);
  double r = size / 2, t = skew_rad;
  auto R = Rotation2Dd(t);
  vector<Vector2d> dxy{Vector2d(r, r), Vector2d(r, -r), Vector2d(-r, -r),
                       Vector2d(-r, r), Vector2d(r, r)};
  transform(dxy.begin(), dxy.end(), dxy.begin(),
            [&R](auto a) { return R * a; });
  for (int i = 0; i < 4; ++i) {
    geometry_msgs::Point pt;
    pt.x = center(0) + dxy[i](0);
    pt.y = center(1) + dxy[i](1);
    pt.z = 0;
    ret.points.push_back(pt);
    pt.x = center(0) + dxy[i + 1](0);
    pt.y = center(1) + dxy[i + 1](1);
    ret.points.push_back(pt);
  }
  return ret;
}

Marker MarkerOfEclipse(const Vector2d &mean, const Matrix2d &covariance,
                       const Color &color,
                       double alpha) {
  Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = ros::Time::now();
  ret.id = 0;
  ret.type = Marker::SPHERE;
  ret.action = Marker::ADD;
  ret.color = MakeColorRGBA(color, alpha);
  EigenSolver<Matrix2d> es(covariance);
  /** Note: "pseudo" computes complex eigen value if no real solution
   * However, covariance is always a real symmetric matrix, which means
   *   1) it is Hermitia, all its eigenvalues are real
   *   2) the decomposed matrix is an orthogonal matrix, i.e., rotaion matrix
   */
  Matrix2d eval = es.pseudoEigenvalueMatrix().cwiseSqrt();
  Matrix2d evec = es.pseudoEigenvectors();
  Matrix3d R = Matrix3d::Identity();
  R.block<2, 2>(0, 0) = evec;
  Quaterniond q(R);
  ret.scale.x = 3 * eval(0, 0);
  ret.scale.y = 3 * eval(1, 1);
  ret.scale.z = 0.1;
  ret.pose.position.x = mean(0);
  ret.pose.position.y = mean(1);
  ret.pose.position.z = 0;
  ret.pose.orientation = tf2::toMsg(q);
  if (ret.scale.x == 0 || ret.scale.y == 0) {
    cout << "cov: " << covariance << endl
         << "evec: " << evec << endl
         << "eval: " << eval << endl
         << "eval w/o sqrt: " << es.pseudoEigenvalueMatrix() << endl;
  }
  return ret;
}

Marker MarkerOfCircle(const Vector2d &mean, double radius,
                      const Color &color,
                      double alpha) {
  Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = ros::Time::now();
  ret.id = 0;
  ret.type = Marker::SPHERE;
  ret.action = Marker::ADD;
  ret.color = MakeColorRGBA(color, alpha);
  ret.scale.x = 2 * radius;
  ret.scale.y = 2 * radius;
  ret.scale.z = 0.1;
  ret.pose.position.x = mean(0);
  ret.pose.position.y = mean(1);
  ret.pose.position.z = 0;
  ret.pose.orientation.w = 1;
  return ret;
}

Marker MarkerOfLines(const vector<Vector2d> &points,
                     const Color &color,
                     double alpha) {
  Expects(points.size() % 2 == 0);
  Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = ros::Time::now();
  ret.id = 0;
  ret.type = Marker::LINE_LIST;
  ret.action = Marker::ADD;
  ret.scale.x = 0.05;
  ret.pose = tf2::toMsg(Affine3d::Identity());
  ret.color = MakeColorRGBA(color, alpha);
  for (const auto &p : points) {
    geometry_msgs::Point pt;
    pt.x = p(0), pt.y = p(1), pt.z = 0;
    ret.points.push_back(pt);
  }
  return ret;
}

Marker MarkerOfLinesByEndPoints(
    const vector<Vector2d> &points,
    const Color &color, double alpha) {
  Expects(points.size() >= 2);
  vector<Vector2d> pts;
  pts.push_back(points.front());
  for (size_t i = 1; i < points.size() - 1; ++i) {
    pts.push_back(points[i]);
    pts.push_back(points[i]);
  }
  pts.push_back(points.back());
  return MarkerOfLines(pts, color, alpha);
}

vector<Vector2d> PointsOfNDTMap(const NDTMap &map) {
  vector<Vector2d> ret;
  for (auto cell : map)
    for (auto pt : cell->GetPoints())
      if (pt.allFinite()) ret.push_back(pt);
  return ret;
}

vector<Vector2d> PointsOfNDTMap(const vector<shared_ptr<NDTCell>> &map) {
  vector<Vector2d> ret;
  for (auto cell : map)
    for (auto pt : cell->GetPoints())
      if (pt.allFinite()) ret.push_back(pt);
  return ret;
}

Marker MarkerOfPoints(const vector<Vector2d> &points, double size,
                      const Color &color,
                      double alpha) {
  auto now = ros::Time::now();
  Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = now;
  ret.id = 0;
  ret.type = Marker::SPHERE_LIST;
  ret.action = Marker::ADD;
  ret.scale.x = ret.scale.y = ret.scale.z = size;
  ret.pose = tf2::toMsg(Affine3d::Identity());
  ret.color = MakeColorRGBA(color);
  for (const auto &point : points) {
    geometry_msgs::Point pt;
    pt.x = point(0), pt.y = point(1), pt.z = 0;
    ret.points.push_back(pt);
  }
  return ret;
}

Marker MarkerOfPoints(const MatrixXd &points, double size,
                      const Color &color,
                      double alpha) {
  vector<Vector2d> pts(points.cols());
  for (int i = 0; i < points.cols(); ++i) pts[i] = points.col(i);
  return MarkerOfPoints(pts, size, color, alpha);
}

Marker MarkerOfArrow(const Vector2d &start, const Vector2d &end,
                     const Color &color,
                     double alpha) {
  Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = ros::Time::now();
  ret.id = 0;
  ret.type = Marker::ARROW;
  ret.action = Marker::ADD;
  ret.scale.x = 0.05;
  ret.scale.y = 0.2;
  ret.pose = tf2::toMsg(Affine3d::Identity());
  ret.color = MakeColorRGBA(color, alpha);
  geometry_msgs::Point pt;
  pt.x = start(0), pt.y = start(1), pt.z = 0;
  ret.points.push_back(pt);
  pt.x = end(0), pt.y = end(1);
  ret.points.push_back(pt);
  return ret;
}

MarkerArray MarkerArrayOfArrow(const MatrixXd &start, const MatrixXd &end,
                               const Color &color,
                               double alpha) {
  Expects(start.cols() == end.cols() && start.rows() == end.rows() &&
          start.rows() == 2);
  MarkerArray ret;
  Marker arrow;
  arrow.header.frame_id = "map";
  arrow.header.stamp = ros::Time::now();
  arrow.id = -1;
  arrow.type = Marker::ARROW;
  arrow.action = Marker::ADD;
  arrow.scale.x = 0.05;
  arrow.scale.y = 0.2;
  arrow.pose = tf2::toMsg(Affine3d::Identity());
  arrow.color = MakeColorRGBA(color, alpha);
  arrow.points.resize(2);
  for (int i = 0; i < start.cols(); ++i) {
    if (!start.col(i).allFinite() || !end.col(i).allFinite()) continue;
    ++arrow.id;
    geometry_msgs::Point pt;
    pt.x = start(0, i), pt.y = start(1, i), pt.z = 0;
    arrow.points[0] = pt;
    pt.x = end(0, i), pt.y = end(1, i);
    arrow.points[1] = pt;
    ret.markers.push_back(arrow);
  }
  return ret;
}

MarkerArray MarkerArrayOfArrow(const vector<Vector2d> &starts, const vector<Vector2d> &ends,
                               const Color &color,
                               double alpha) {
  Expects(starts.size() == ends.size());
  MarkerArray ret;
  Marker arrow;
  arrow.header.frame_id = "map";
  arrow.header.stamp = ros::Time::now();
  arrow.id = -1;
  arrow.type = Marker::ARROW;
  arrow.action = Marker::ADD;
  arrow.scale.x = 0.05;
  arrow.scale.y = 0.2;
  arrow.pose = tf2::toMsg(Affine3d::Identity());
  arrow.color = MakeColorRGBA(color, alpha);
  arrow.points.resize(2);
  for (size_t i = 0; i < starts.size(); ++i) {
    if (!starts[i].allFinite() || !ends[i].allFinite()) continue;
    ++arrow.id;
    geometry_msgs::Point pt;
    pt.x = starts[i](0), pt.y = starts[i](1), pt.z = 0;
    arrow.points[0] = pt;
    pt.x = ends[i](0), pt.y = ends[i](1);
    arrow.points[1] = pt;
    ret.markers.push_back(arrow);
  }
  return ret;
}

// Color set for general usage
MarkerArray MarkerArrayOfNDTCell(const NDTCell *cell) {
  MarkerArray ret;
  auto boundary =
      MarkerOfBoundary(cell->GetCenter(), cell->GetSize(), cell->GetSkewRad());
  auto p_eclipse = MarkerOfEclipse(cell->GetPointMean(), cell->GetPointCov());
  // Old
  if (cell->GetNHasGaussian()) {
    auto n_eclipse =
        MarkerOfEclipse(cell->GetPointMean() + cell->GetNormalMean(),
                        cell->GetNormalCov(), Color::kGray);
    auto points = FindTangentPoints(n_eclipse, cell->GetPointMean());
    auto lines = MarkerOfLines({points[0], cell->GetPointMean(), cell->GetPointMean(), points[1]});
    ret = JoinMarkerArraysAndMarkers({}, {boundary, p_eclipse, n_eclipse, lines});
  } else {
    ret = JoinMarkerArraysAndMarkers({}, {boundary, p_eclipse});
  }

  // New
  // auto nm = (cell->GetPointEvals()(0) > cell->GetPointEvals()(1))
  //               ? cell->GetPointEvecs().col(0)
  //               : cell->GetPointEvecs().col(1);
  // auto normal =
  //     MarkerOfArrow(cell->GetPointMean(), cell->GetPointMean() + nm);
  // ret = JoinMarkerArraysAndMarkers({}, {boundary, p_eclipse, normal});
  return ret;
}

// Color set for target point cloud
MarkerArray MarkerArrayOfNDTCell2(const NDTCell *cell) {
  MarkerArray ret;
  auto boundary = MarkerOfBoundary(cell->GetCenter(), cell->GetSize(),
                                   cell->GetSkewRad(), Color::kRed);
  auto p_eclipse = MarkerOfEclipse(cell->GetPointMean(), cell->GetPointCov(),
                                   Color::kRed);
  // Old
  if (cell->GetNHasGaussian()) {
    auto n_eclipse =
        MarkerOfEclipse(cell->GetPointMean() + cell->GetNormalMean(),
                        cell->GetNormalCov(), Color::kGray);
    auto points = FindTangentPoints(n_eclipse, cell->GetPointMean());
    auto lines = MarkerOfLines({points[0], cell->GetPointMean(), cell->GetPointMean(), points[1]}, Color::kGray);
    ret = JoinMarkerArraysAndMarkers({}, {boundary, p_eclipse, n_eclipse, lines});
  } else {
    ret = JoinMarkerArraysAndMarkers({}, {boundary, p_eclipse});
  }

  // New
  // auto normal =
  //     MarkerOfArrow(cell->GetPointMean(),
  //                   cell->GetPointMean() + cell->GetPointEvecs().col(1));
  // ret = JoinMarkerArraysAndMarkers({}, {boundary, p_eclipse, normal});
  return ret;
}

MarkerArray MarkerArrayOfNDTMap(const NDTMap &map,
                                bool is_target_color) {
  vector<MarkerArray> vma;
  for (auto cell : map) {
    // FIXME: omit when no gaussian
    if (!cell->BothHasGaussian()) { continue; }
    if (is_target_color)
      vma.push_back(MarkerArrayOfNDTCell2(cell));
    else
      vma.push_back(MarkerArrayOfNDTCell(cell));
  }
  return JoinMarkerArraysAndMarkers(vma);
}

MarkerArray MarkerArrayOfNDTMap(const vector<shared_ptr<NDTCell>> &map,
                                bool is_target_color) {
  vector<MarkerArray> vma;
  for (auto cell : map) {
    // FIXME: omit when no gaussian
    if (!cell->BothHasGaussian()) continue;
    if (is_target_color)
      vma.push_back(MarkerArrayOfNDTCell2(cell.get()));
    else
      vma.push_back(MarkerArrayOfNDTCell(cell.get()));
  }
  return JoinMarkerArraysAndMarkers(vma);
}

// TODO: add text with cost value at the middle point
MarkerArray MarkerArrayOfCorrespondences(const NDTCell *source_cell, const NDTCell *target_cell) {
  auto mas = MarkerArrayOfNDTCell(source_cell);
  auto mat = MarkerArrayOfNDTCell2(target_cell);
  auto line =
      MarkerOfLines({source_cell->GetPointMean(), target_cell->GetPointMean()},
                    Color::kBlack, 1);
  return JoinMarkerArraysAndMarkers({mas, mat}, {line});
}

MarkerArray MarkerArrayOfSensor(const vector<Affine2d> &affs) {
  MarkerArray ret;
  auto now = ros::Time::now();
  int i = 0;
  for (const auto &aff : affs) {
    Marker m;
    m.header.frame_id = "map";
    m.header.stamp = now;
    m.id = i++;
    m.type = Marker::CUBE;
    m.action = Marker::ADD;
    Affine3d aff3 = Translation3d(aff.translation().x(), aff.translation().y(), 0) *
                    AngleAxisd(Rotation2Dd(aff.rotation()).angle(), Vector3d::UnitZ());
    m.pose = tf2::toMsg(aff3);
    m.color = MakeColorRGBA(Color::kGray);
    m.scale.x = 0.3;
    m.scale.y = 1.0;
    m.scale.z = 0.5;
    ret.markers.push_back(m);
  }
  return ret;
}
