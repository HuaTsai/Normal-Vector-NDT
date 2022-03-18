/**
 * @file eigen_utils.h
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Eigen Utilities
 * @version 0.1
 * @date 2021-07-29
 * @details All functions are defined inline to prevent compile error of
 * multiple definition in different compilation units. It is the same reason
 * that we implement function in the class definition.
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <bits/stdc++.h>
#include <pcl/search/kdtree.h>
#include <common/angle_utils.h>
#include <common/eigen_utils.h>

#include <Eigen/Dense>

/**
 * @brief Remove points that contain NaN or Inf
 *
 * @param points Input points
 * @return Result points
 */
inline std::vector<Eigen::Vector2d> ExcludeNaNInf(
    const std::vector<Eigen::Vector2d> &points) {
  std::vector<Eigen::Vector2d> ret;
  std::copy_if(points.begin(), points.end(), back_inserter(ret),
               [](const Eigen::Vector2d &v) { return v.allFinite(); });
  return ret;
}

// TODO: document
// inline void ExcludeInfinite(const std::vector<Eigen::Vector2d> &points,
//                             std::vector<Eigen::Vector2d> &valid_points) {
//   valid_points.clear();
//   for (const auto &pt : points)
//     if (pt.allFinite()) valid_points.push_back(pt);
// }

// inline void ExcludeInfinite(const std::vector<Eigen::Vector2d> &points,
//                             const std::vector<Eigen::Matrix2d> &covariances,
//                             std::vector<Eigen::Vector2d> &valid_points,
//                             std::vector<Eigen::Matrix2d> &valid_covariances) {
//   valid_points.clear();
//   valid_covariances.clear();
//   for (size_t i = 0; i < points.size(); ++i) {
//     if (points[i].allFinite() && covariances[i].allFinite()) {
//       valid_points.push_back(points[i]);
//       valid_covariances.push_back(covariances[i]);
//     }
//   }
// }

// inline void ExcludeInfinite(const std::vector<Eigen::Vector2d> &points,
//                             const std::vector<Eigen::Vector2d> &normals,
//                             std::vector<Eigen::Vector2d> &valid_points,
//                             std::vector<Eigen::Vector2d> &valid_normals) {
//   valid_points.clear();
//   valid_normals.clear();
//   for (size_t i = 0; i < points.size(); ++i) {
//     if (points[i].allFinite() && normals[i].allFinite()) {
//       valid_points.push_back(points[i]);
//       valid_normals.push_back(normals[i]);
//     }
//   }
// }

inline pcl::KdTreeFLANN<pcl::PointXY> MakeKDTree(
    const std::vector<Eigen::Vector2d> &points) {
  pcl::PointCloud<pcl::PointXY>::Ptr pc(new pcl::PointCloud<pcl::PointXY>);
  for (const auto &pt : points) {
    pcl::PointXY p;
    p.x = pt(0), p.y = pt(1);
    pc->push_back(p);
  }
  pcl::KdTreeFLANN<pcl::PointXY> ret;
  ret.setInputCloud(pc);
  return ret;
}

inline int FindNearestNeighborIndex(const Eigen::Vector2d &query,
                                    const pcl::KdTreeFLANN<pcl::PointXY> &kd) {
  pcl::PointXY pt;
  pt.x = query(0), pt.y = query(1);
  std::vector<int> idx{0};
  std::vector<float> dist2{0};
  int found = kd.nearestKSearch(pt, 1, idx, dist2);
  if (!found) return -1;
  return idx[0];
}

class RandomTransformGenerator2D {
 public:
  RandomTransformGenerator2D(double radius)
      : dre_(new std::default_random_engine()), radius_(radius) {}
  std::vector<Eigen::Affine2d> Generate(int sizes) {
    std::vector<Eigen::Affine2d> ret;
    std::uniform_real_distribution<> urd(-M_PI, M_PI);
    // std::uniform_real_distribution<> urd2(-M_PI / 4, M_PI / 4);
    double rad = Deg2Rad(10);
    std::uniform_real_distribution<> urd2(-rad, rad);
    for (int i = 0; i < sizes; ++i) {
      double angle = urd(*dre_);
      double x = radius_ * cos(angle);
      double y = radius_ * sin(angle);
      double t = urd2(*dre_);
      // double t = 0;
      ret.push_back(Eigen::Translation2d(x, y) * Eigen::Rotation2Dd(t));
    }
    return ret;
  }

 private:
  std::shared_ptr<std::default_random_engine> dre_;
  double radius_;
};

class RandomTranslationGenerator {
 public:
  explicit RandomTranslationGenerator(double radius)
      : dre_(new std::default_random_engine()), radius_(radius) {}
  std::vector<Eigen::Affine2d> Generate(int sizes) {
    std::vector<Eigen::Affine2d> ret;
    std::uniform_real_distribution<> urd(-M_PI, M_PI);
    for (int i = 0; i < sizes; ++i) {
      double angle = urd(*dre_);
      double x = radius_ * cos(angle);
      double y = radius_ * sin(angle);
      ret.push_back(Eigen::Affine2d(Eigen::Translation2d(x, y)));
    }
    return ret;
  }

 private:
  std::shared_ptr<std::default_random_engine> dre_;
  double radius_;
};

class RandomRotationGenerator {
 public:
  explicit RandomRotationGenerator(double angle_deg)
      : dre_(new std::default_random_engine()),
        angle_rad_(Deg2Rad(angle_deg)) {}
  std::vector<Eigen::Affine2d> Generate(int sizes) {
    std::vector<Eigen::Affine2d> ret;
    std::uniform_real_distribution<> urd(-angle_rad_, angle_rad_);
    for (int i = 0; i < sizes; ++i)
      ret.push_back(Eigen::Affine2d(Eigen::Rotation2Dd(urd(*dre_))));
    return ret;
  }

 private:
  std::shared_ptr<std::default_random_engine> dre_;
  double angle_rad_;
};

class RandomTransformGenerator {
 public:
  RandomTransformGenerator(double radius, double angle_deg)
      : dre_(new std::default_random_engine()),
        radius_(radius),
        angle_rad_(angle_deg * M_PI / 180.) {}
  std::vector<Eigen::Affine2d> Generate(int sizes) {
    std::vector<Eigen::Affine2d> ret;
    std::uniform_real_distribution<> urd(-M_PI, M_PI);
    std::bernoulli_distribution ber;
    for (int i = 0; i < sizes; ++i) {
      double angle = urd(*dre_);
      double x = radius_ * cos(angle);
      double y = radius_ * sin(angle);
      double t = ber(*dre_) ? angle_rad_ : -angle_rad_;
      ret.push_back(Eigen::Translation2d(x, y) * Eigen::Rotation2Dd(t));
    }
    return ret;
  }

 private:
  std::shared_ptr<std::default_random_engine> dre_;
  double radius_;
  double angle_rad_;
};

class UniformTransformGenerator {
 public:
  explicit UniformTransformGenerator(double radius, double angle_deg)
      : radius_(radius), angle_rad_(angle_deg * M_PI / 180.) {}
  std::vector<Eigen::Affine2d> Generate(int sizes) {
    std::vector<Eigen::Affine2d> ret;
    for (int i = 0; i < sizes; ++i) {
      double angle = 2 * M_PI * i / sizes;
      double x = radius_ * cos(angle);
      double y = radius_ * sin(angle);
      double t = (i % 2) ? angle_rad_ : -angle_rad_;
      ret.push_back(Eigen::Translation2d(x, y) * Eigen::Rotation2Dd(t));
    }
    return ret;
  }

 private:
  double radius_;
  double angle_rad_;
};
