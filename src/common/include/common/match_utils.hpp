#pragma once

#include <bits/stdc++.h>

#include <Eigen/Dense>
#include <gsl/gsl>

namespace common {
class RandomTransformGenerator2D {
 public:
  RandomTransformGenerator2D()
      : is_radius_set_(false),
        is_angle_set_(false),
        radius_min_(0),
        radius_max_(0),
        angle_rad_min_(0),
        angle_rad_max_(0),
        center_x_(0),
        center_y_(0),
        center_th_(0) {
    std::random_device rd;
    dre_ = std::make_shared<std::default_random_engine>(rd());
  }

  void SetTranslationRadiusBound(double min, double max) {
    Expects(min >= 0 && max > min);
    radius_min_ = min;
    radius_max_ = max;
    is_radius_set_ = true;
  }

  std::vector<double> GetTranslationRadiusBound() const {
    return {radius_min_, radius_max_};
  }

  void SetRotationDegreeBound(double min, double max) {
    SetRotationRadianBound(min * M_PI / 180., max * M_PI / 180.);
  }

  void SetRotationRadianBound(double min, double max) {
    Expects(min >= 0 && max > min);
    angle_rad_min_ = min;
    angle_rad_max_ = max;
    is_angle_set_ = true;
  }

  std::vector<double> GetRotationDegreeBound() const {
    return {angle_rad_min_ * 180. / M_PI, angle_rad_max_ * 180. / M_PI};
  }

  void SetCenterXYDegree(double x, double y, double degree) {
    SetCenterXYRadian(x, y, degree * M_PI / 180.);
  }

  void SetCenterXYRadian(double x, double y, double radian) {
    center_x_ = x;
    center_y_ = y;
    center_th_ = radian;
  }

  void Clear() {
    is_radius_set_ = false;
    is_angle_set_ = false;
    radius_min_ = 0;
    radius_max_ = 0;
    angle_rad_min_ = 0;
    angle_rad_max_ = 0;
    center_x_ = 0;
    center_y_ = 0;
    center_th_ = 0;
  }

  std::vector<Eigen::Matrix3d> Generate(size_t sizes) {
    Expects(is_radius_set_ && is_angle_set_);
    std::vector<Eigen::Matrix3d> ret(sizes, Eigen::Matrix3d::Identity());
    std::uniform_real_distribution<> radius_urd(radius_min_ * radius_min_,
                                                radius_max_ * radius_max_);
    std::uniform_real_distribution<> theta_urd(0, 2 * M_PI);
    std::uniform_real_distribution<> angle_urd(angle_rad_min_, angle_rad_max_);
    for (size_t i = 0; i < sizes; ++i) {
      double radius = sqrt(radius_urd(*dre_));
      double theta = theta_urd(*dre_);
      double angle = angle_urd(*dre_);
      ret.at(i)(0, 2) = center_x_ + radius * cos(theta);
      ret.at(i)(1, 2) = center_y_ + radius * sin(theta);
      ret.at(i)(0, 0) = cos(center_th_ + angle);
      ret.at(i)(0, 1) = -sin(center_th_ + angle);
      ret.at(i)(1, 0) = sin(center_th_ + angle);
      ret.at(i)(1, 1) = cos(center_th_ + angle);
    }
    return ret;
  }

 private:
  std::shared_ptr<std::default_random_engine> dre_;
  bool is_radius_set_;
  bool is_angle_set_;
  double radius_min_;
  double radius_max_;
  double angle_rad_min_;
  double angle_rad_max_;
  double center_x_;
  double center_y_;
  double center_th_;
};

std::vector<std::vector<double>> Combinations(std::vector<std::vector<double>> x) {
  /* input: {{0, 1}, {a, b}, {x, y}}
   * output: {{0, a, x}, {0, a, y}, ..., {1, b, y}}
   */
  std::vector<std::vector<double>> ret;
  std::vector<std::vector<double>::iterator> its;
  for (size_t i = 0; i < x.size(); ++i) {
    its.push_back(x.at(i).begin());
  }
  while (its.at(0) != x.at(0).end()) {
    std::vector<double> tmp;
    for (size_t i = 0; i < x.size(); ++i) {
      tmp.push_back(*its.at(i));
    }
    ret.push_back(tmp);
    ++its.back();
    for (size_t i = x.size() - 1; (i > 0) && (its.at(i) == x.at(i).end()); --i) {
      its.at(i) = x.at(i).begin();
      ++its.at(i - 1);
    }
  }
  return ret;
}

struct MatchConfig {
  std::string reg;
  int max_iters;
  double max_corr_dists;
  bool set_reci_corr;
  double l2_rel_eps;
  double tf_eps;
  // SICP
  double neighbors;
  // NDT
  double cell_size;
};

class MatchConfigsGenerator {
 public:
  MatchConfigsGenerator() : ctr(0) {}
  void SetMaxIters(std::vector<int>) {
    ++ctr;
  }
 private:
  int ctr;
  std::vector<MatchConfig> configs_;
};
}  // namespace common
