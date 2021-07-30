/**
 * @file visuals.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Definition of Visual Utilities
 * @version 0.1
 * @date 2021-07-30
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include <sndt/visuals.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

/**
 * @brief Get current ros time. If it throws, catch it and set to @c
 * ros::Time(0).
 */
ros::Time GetROSTime() {
  ros::Time ret;
  try {
    ret = ros::Time::now();
  } catch (const ros::Exception &ex) {
    std::cout << ex.what() << std::endl;
    std::cout << "Set stamp to ros::Time(0)" << std::endl;
    ret = ros::Time(0);
  }
  return ret;
}

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

std::vector<Eigen::Vector2d> FindTangentPoints(const Marker &ellipse,
                                               const Eigen::Vector2d &point) {
  Eigen::Affine3d aff3;
  tf2::fromMsg(ellipse.pose, aff3);
  Eigen::Matrix3d mtx = Eigen::Matrix3d::Identity();
  mtx.block<2, 2>(0, 0) = aff3.rotation().block<2, 2>(0, 0);
  mtx.block<2, 1>(0, 2) = aff3.translation().block<2, 1>(0, 0);
  Eigen::Affine2d aff2(mtx);
  auto point2 = aff2.inverse() * point;
  auto rx2 = (ellipse.scale.x / 2) * (ellipse.scale.x / 2);
  auto ry2 = (ellipse.scale.y / 2) * (ellipse.scale.y / 2);
  auto x0 = point2(0), x02 = x0 * x0;
  auto y0 = point2(1), y02 = y0 * y0;
  std::vector<Eigen::Vector2d> sols(2);
  if (x02 == rx2) {
    auto msol = (-rx2 * ry2 * ry2 + rx2 * ry2 * y02) / (2 * rx2 * ry2 * x0 * y0);
    sols[0](0) = (msol * rx2 * (msol * x0 - y0)) / (msol * msol * rx2 + ry2);
    sols[0](1) = y0 + msol * (sols[0](0) - x0);
    sols[1](0) = x0;
    sols[1](1) = 0;
  } else {
    auto msol1 = (-x0 * y0 + sqrt(-rx2 * ry2 + rx2 * y02 + ry2 * x02)) / (rx2 - x02);
    sols[0](0) = (msol1 * rx2 * (msol1 * x0 - y0)) / (msol1 * msol1 * rx2 + ry2);
    sols[0](1) = y0 + msol1 * (sols[0](0) - x0);
    auto msol2 = (-x0 * y0 - sqrt(-rx2 * ry2 + rx2 * y02 + ry2 * x02)) / (rx2 - x02);
    sols[1](0) = (msol2 * rx2 * (msol2 * x0 - y0)) / (msol2 * msol2 * rx2 + ry2);
    sols[1](1) = y0 + msol2 * (sols[1](0) - x0);
  }
  for (auto &sol : sols)
    sol = aff2 * sol;
  return sols;
}

MarkerArray JoinMarkers(const std::vector<Marker> &markers) {
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

MarkerArray JoinMarkerArrays(const std::vector<MarkerArray> &markerarrays) {
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

MarkerArray JoinMarkerArraysAndMarkers(
    const std::vector<MarkerArray> &markerarrays,
    const std::vector<Marker> &markers) {
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
  for (const auto &m : markers) {
    ret.markers.push_back(m);
    ret.markers.back().header.stamp = now;
    ret.markers.back().id = id++;
  }
  return ret;
}

Marker MarkerOfBoundary(const Eigen::Vector2d &center, double size,
                        double skew_rad, const Color &color, double alpha) {
  Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = GetROSTime();
  ret.id = 0;
  ret.type = Marker::LINE_LIST;
  ret.action = Marker::ADD;
  ret.pose = tf2::toMsg(Eigen::Affine3d::Identity());
  ret.scale.x = 0.2;
  ret.color = MakeColorRGBA(color, alpha);
  double r = size / 2, t = skew_rad;
  auto R = Eigen::Rotation2Dd(t);
  std::vector<Eigen::Vector2d> dxy{
      Eigen::Vector2d(r, r), Eigen::Vector2d(r, -r), Eigen::Vector2d(-r, -r),
      Eigen::Vector2d(-r, r), Eigen::Vector2d(r, r)};
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

Marker MarkerOfEllipse(const Eigen::Vector2d &mean, const Eigen::Matrix2d &covariance,
                       const Color &color,
                       double alpha) {
  Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = GetROSTime();
  ret.id = 0;
  ret.type = Marker::SPHERE;
  ret.action = Marker::ADD;
  ret.color = MakeColorRGBA(color, alpha);
  Eigen::EigenSolver<Eigen::Matrix2d> es(covariance);
  /** Note: "pseudo" computes complex eigen value if no real solution
   * However, covariance is always a real symmetric matrix, which means
   *   1) it is Hermitia, all its eigenvalues are real
   *   2) the decomposed matrix is an orthogonal matrix, i.e., rotaion matrix
   */
  Eigen::Matrix2d eval = es.pseudoEigenvalueMatrix().cwiseSqrt();
  Eigen::Matrix2d evec = es.pseudoEigenvectors();
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  R.block<2, 2>(0, 0) = evec;
  Eigen::Quaterniond q(R);
  ret.scale.x = 3 * eval(0, 0);
  ret.scale.y = 3 * eval(1, 1);
  ret.scale.z = 0.1;
  ret.pose.position.x = mean(0);
  ret.pose.position.y = mean(1);
  ret.pose.position.z = 0;
  ret.pose.orientation = tf2::toMsg(q);
  if (ret.scale.x == 0 || ret.scale.y == 0) {
    std::cout << "cov: " << covariance << std::endl
         << "evec: " << evec << std::endl
         << "eval: " << eval << std::endl
         << "eval w/o sqrt: " << es.pseudoEigenvalueMatrix() << std::endl;
  }
  return ret;
}

Marker MarkerOfCircle(const Eigen::Vector2d &center, double radius,
                      const Color &color,
                      double alpha) {
  Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = GetROSTime();
  ret.id = 0;
  ret.type = Marker::SPHERE;
  ret.action = Marker::ADD;
  ret.color = MakeColorRGBA(color, alpha);
  ret.scale.x = 2 * radius;
  ret.scale.y = 2 * radius;
  ret.scale.z = 0.1;
  ret.pose.position.x = center(0);
  ret.pose.position.y = center(1);
  ret.pose.position.z = 0;
  ret.pose.orientation.w = 1;
  return ret;
}

Marker MarkerOfLines(const std::vector<Eigen::Vector2d> &points,
                     const Color &color,
                     double alpha) {
  Expects(points.size() % 2 == 0);
  Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = GetROSTime();
  ret.id = 0;
  ret.type = Marker::LINE_LIST;
  ret.action = Marker::ADD;
  ret.scale.x = 0.05;
  ret.pose = tf2::toMsg(Eigen::Affine3d::Identity());
  ret.color = MakeColorRGBA(color, alpha);
  for (const auto &p : points) {
    geometry_msgs::Point pt;
    pt.x = p(0), pt.y = p(1), pt.z = 0;
    ret.points.push_back(pt);
  }
  return ret;
}

Marker MarkerOfLinesByMiddlePoints(
    const std::vector<Eigen::Vector2d> &points,
    const Color &color, double alpha) {
  Expects(points.size() >= 2);
  std::vector<Eigen::Vector2d> pts;
  pts.push_back(points.front());
  for (size_t i = 1; i < points.size() - 1; ++i) {
    pts.push_back(points[i]);
    pts.push_back(points[i]);
  }
  pts.push_back(points.back());
  return MarkerOfLines(pts, color, alpha);
}

Marker MarkerOfPoints(const std::vector<Eigen::Vector2d> &points, double size,
                      const Color &color,
                      double alpha) {
  auto now = GetROSTime();
  Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = now;
  ret.id = 0;
  ret.type = Marker::SPHERE_LIST;
  ret.action = Marker::ADD;
  ret.scale.x = ret.scale.y = ret.scale.z = size;
  ret.pose = tf2::toMsg(Eigen::Affine3d::Identity());
  ret.color = MakeColorRGBA(color);
  for (const auto &point : points) {
    geometry_msgs::Point pt;
    pt.x = point(0), pt.y = point(1), pt.z = 0;
    ret.points.push_back(pt);
  }
  return ret;
}

Marker MarkerOfArrow(const Eigen::Vector2d &start, const Eigen::Vector2d &end,
                     const Color &color,
                     double alpha) {
  Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = GetROSTime();
  ret.id = 0;
  ret.type = Marker::ARROW;
  ret.action = Marker::ADD;
  ret.scale.x = 0.05;
  ret.scale.y = 0.2;
  ret.pose = tf2::toMsg(Eigen::Affine3d::Identity());
  ret.color = MakeColorRGBA(color, alpha);
  geometry_msgs::Point pt;
  pt.x = start(0), pt.y = start(1), pt.z = 0;
  ret.points.push_back(pt);
  pt.x = end(0), pt.y = end(1);
  ret.points.push_back(pt);
  return ret;
}

Marker MarkerOfText(std::string text, const Eigen::Vector2d &position,
                    const Color &color, double alpha) {
  Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = GetROSTime();
  ret.id = 0;
  ret.type = Marker::TEXT_VIEW_FACING;
  ret.action = Marker::ADD;
  ret.text = text;
  ret.pose.position.x = position(0);
  ret.pose.position.y = position(1);
  ret.pose.position.z = 0;
  ret.pose.orientation.w = 1;
  ret.scale.z = 0.7;
  ret.color = MakeColorRGBA(color);
  return ret;
}

MarkerArray MarkerArrayOfArrows(const std::vector<Eigen::Vector2d> &starts,
                                const std::vector<Eigen::Vector2d> &ends,
                                const Color &color, double alpha) {
  Expects(starts.size() == ends.size());
  MarkerArray ret;
  Marker arrow;
  arrow.header.frame_id = "map";
  arrow.header.stamp = GetROSTime();
  arrow.id = -1;
  arrow.type = Marker::ARROW;
  arrow.action = Marker::ADD;
  arrow.scale.x = 0.05;
  arrow.scale.y = 0.2;
  arrow.pose = tf2::toMsg(Eigen::Affine3d::Identity());
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

MarkerArray MarkerArrayOfSNDTCell(const SNDTCell *cell) {
  MarkerArray ret;
  auto bdy = MarkerOfBoundary(cell->GetCenter(), cell->GetSize(),
                              cell->GetSkewRad(), Color::kLime);
  auto pell = MarkerOfEllipse(cell->GetPointMean(), cell->GetPointCov());
  if (cell->GetNHasGaussian()) {
    auto nell = MarkerOfEllipse(cell->GetPointMean() + cell->GetNormalMean(),
                                cell->GetNormalCov(), Color::kGray);
    auto points = FindTangentPoints(nell, cell->GetPointMean());
    auto lines = MarkerOfLinesByMiddlePoints(
        {points[0], cell->GetPointMean(), points[1]}, Color::kGray);
    ret = JoinMarkers({bdy, pell, nell, lines});
  } else {
    ret = JoinMarkers({bdy, pell});
  }
  return ret;
}

MarkerArray MarkerArrayOfSNDTCell2(const SNDTCell *cell) {
  MarkerArray ret;
  auto bdy = MarkerOfBoundary(cell->GetCenter(), cell->GetSize(),
                              cell->GetSkewRad(), Color::kRed);
  auto pell = MarkerOfEllipse(cell->GetPointMean(), cell->GetPointCov(),
                              Color::kRed);
  if (cell->GetNHasGaussian()) {
    auto nell = MarkerOfEllipse(cell->GetPointMean() + cell->GetNormalMean(),
                                cell->GetNormalCov(), Color::kGray);
    auto points = FindTangentPoints(nell, cell->GetPointMean());
    auto lines = MarkerOfLinesByMiddlePoints(
        {points[0], cell->GetPointMean(), points[1]}, Color::kGray);
    ret = JoinMarkers({bdy, pell, nell, lines});
  } else {
    ret = JoinMarkers({bdy, pell});
  }
  return ret;
}

MarkerArray MarkerArrayOfSNDTMap(const SNDTMap &map,
                                 bool use_target_color) {
  std::vector<MarkerArray> vma;
  for (auto cell : map) {
    if (!cell->HasGaussian()) { continue; }
    if (use_target_color)
      vma.push_back(MarkerArrayOfSNDTCell2(cell));
    else
      vma.push_back(MarkerArrayOfSNDTCell(cell));
  }
  return JoinMarkerArrays(vma);
}

MarkerArray MarkerArrayOfSNDTMap(const std::vector<std::shared_ptr<SNDTCell>> &map,
                                 bool use_target_color) {
  std::vector<MarkerArray> vma;
  for (auto cell : map) {
    if (!cell->HasGaussian()) continue;
    if (use_target_color)
      vma.push_back(MarkerArrayOfSNDTCell2(cell.get()));
    else
      vma.push_back(MarkerArrayOfSNDTCell(cell.get()));
  }
  return JoinMarkerArrays(vma);
}

MarkerArray MarkerArrayOfCorrespondences(const SNDTCell *source_cell,
                                         const SNDTCell *target_cell,
                                         std::string text, const Color &color) {
  auto mas = MarkerArrayOfSNDTCell(source_cell);
  auto mat = MarkerArrayOfSNDTCell2(target_cell);
  auto mline =
      MarkerOfLines({source_cell->GetPointMean(), target_cell->GetPointMean()},
                    Color::kBlack);
  Eigen::Vector2d middle = (source_cell->GetPointMean() + target_cell->GetPointMean()) / 2;
  auto mtext = MarkerOfText(text, middle, color);
  return JoinMarkerArraysAndMarkers({mas, mat}, {mline, mtext});
}

MarkerArray MarkerArrayOfSensor(const std::vector<Eigen::Affine2d> &affs) {
  MarkerArray ret;
  auto now = GetROSTime();
  int i = 0;
  for (const auto &aff : affs) {
    Marker m;
    m.header.frame_id = "map";
    m.header.stamp = now;
    m.id = i++;
    m.type = Marker::CUBE;
    m.action = Marker::ADD;
    Eigen::Affine3d aff3 =
        Eigen::Translation3d(aff.translation().x(), aff.translation().y(), 0) *
        Eigen::AngleAxisd(Eigen::Rotation2Dd(aff.rotation()).angle(),
                          Eigen::Vector3d::UnitZ());
    m.pose = tf2::toMsg(aff3);
    m.color = MakeColorRGBA(Color::kGray);
    m.scale.x = 0.3;
    m.scale.y = 1.0;
    m.scale.z = 0.5;
    ret.markers.push_back(m);
  }
  return ret;
}
