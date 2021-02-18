#pragma once

#include <bits/stdc++.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Eigen>
#include "common/common.h"

using namespace std;

class NDTCell {
 public:
  enum mode { PROVIDE_NORMAL, COMPUTE_NORMAL };
  bool phasGaussian_;
  bool nhasGaussian_;
  vector<pcl::PointXY, Eigen::aligned_allocator<pcl::PointXY>> points_;
  vector<pcl::PointXY, Eigen::aligned_allocator<pcl::PointXY>> normals_;

  void InitializeVariables() {
    phasGaussian_ = nhasGaussian_ = false;
    N_ = 0;
    center_.x = center_.y = 0;
    xsize_ = ysize_ = 0;
    pcov_ = Eigen::Matrix2d::Zero();
    picov_ = Eigen::Matrix2d::Zero();
    pevecs_ = Eigen::Matrix2d::Zero();
    pmean_ = Eigen::Vector2d::Zero();
    pevals_ = Eigen::Vector2d::Zero();
    ncov_ = Eigen::Matrix2d::Zero();
    nicov_ = Eigen::Matrix2d::Zero();
    nevecs_ = Eigen::Matrix2d::Zero();
    nmean_ = Eigen::Vector2d::Zero();
    nevals_ = Eigen::Vector2d::Zero();
  }

  NDTCell() { InitializeVariables(); }

  NDTCell(pcl::PointXY &center, double &xsize, double &ysize) {
    InitializeVariables();
    center_ = center;
    xsize_ = xsize;
    ysize_ = ysize;
  }

  // Get and Set Center
  inline pcl::PointXY getCenter() const { return center_; }
  inline void getCenter(double &cx, double &cy) const {
    cx = center_.x;
    cy = center_.y;
  }
  inline void setCenter(const pcl::PointXY &cn) { center_ = cn; }
  inline void setCenter(double cx, double cy) {
    center_.x = cx;
    center_.y = cy;
  }

  // Get and Set Dimensions
  inline void getDimensions(double &xs, double &ys) const {
    xs = xsize_;
    ys = ysize_;
  }
  inline void setDimensions(double xs, double ys) {
    xsize_ = xs;
    ysize_ = ys;
  }

  inline bool isInside(const pcl::PointXY pt) const {
    if (pt.x < center_.x - xsize_ / 2 || pt.x > center_.x + xsize_ / 2) {
      return false;
    }
    if (pt.y < center_.y - ysize_ / 2 || pt.y > center_.y + ysize_ / 2) {
      return false;
    }
    return true;
  }

  inline Eigen::Vector2d computeMean(
      const vector<pcl::PointXY, Eigen::aligned_allocator<pcl::PointXY>>
          &points) {
    Eigen::Vector2d ret = Eigen::Vector2d::Zero();
    for (size_t i = 0; i < points.size(); ++i) {
      ret += Eigen::Vector2d(points.at(i).x, points.at(i).y);
    }
    ret /= points.size();
    return ret;
  }

  inline Eigen::Matrix2d computeCov(
      const vector<pcl::PointXY, Eigen::aligned_allocator<pcl::PointXY>>
          &points) {
    // TODO: intrinsic covariance
    Eigen::Matrix2d ret;

    int n = points.size();
    if (n == 1) {
      ret << 0.1, 0, 0, 0.1;
      return ret;
    }

    Eigen::Vector2d mean = computeMean(points);
    Eigen::MatrixXd mp(2, n);
    Eigen::Matrix2d covp;
    covp << 0.1, 0, 0, 0.1;
    for (int i = 0; i < n; ++i) {
      mp.block<2, 1>(0, i) = Eigen::Vector2d(points.at(i).x, points.at(i).y) - mean;
    }

    // ret = mp * mp.transpose() / (n - 1);
    ret = mp * mp.transpose() / (n - 1) + covp;
    return ret;
  }

  inline vector<pcl::PointXY, Eigen::aligned_allocator<pcl::PointXY>>
  excludeNaN(const vector<pcl::PointXY, Eigen::aligned_allocator<pcl::PointXY>>
                 &points) {
    vector<pcl::PointXY, Eigen::aligned_allocator<pcl::PointXY>> ret;
    for (auto &pt : points) {
      if (!isnan(pt.x) && !isnan(pt.y)) {
        ret.push_back(pt);
      }
    }
    return ret;
  }

  void computePGaussian() {
    if (!phasGaussian_) {
      auto new_points = excludeNaN(points_);
      if (points_.size() != new_points.size()) {
        dprintf("cell @ (%.2f, %.2f): points_ exclude %ld NaNs\n", center_.x,
                center_.y, points_.size() - new_points.size());
      }
      if (new_points.size() == 0) {
        dprintf("cell @ (%.2f, %.2f): no points!\n", center_.x, center_.y);
        pmean_.setZero();
        pcov_.setZero();
        return;
      }
      pmean_ = computeMean(new_points);
      pcov_ = computeCov(new_points);
      bool invertable = false;
      pcov_.computeInverseWithCheck(picov_, invertable);
      if (!invertable) {
        dprintf("cell @ (%.2f, %.2f): pcov_ not invertable\n", center_.x,
                center_.y);
        picov_.setZero();
      }
      phasGaussian_ = true;
    }
  }

  void computeNGaussian() {
    if (!nhasGaussian_) {
      auto new_normals = excludeNaN(normals_);
      if (normals_.size() != new_normals.size()) {
        dprintf("cell @ (%.2f, %.2f): normals_ exclude %ld NaNs\n", center_.x,
                center_.y, normals_.size() - new_normals.size());
      }
      if (new_normals.size() == 0) {
        dprintf("cell @ (%.2f, %.2f): no normals!\n", center_.x, center_.y);
        nmean_.setZero();
        ncov_.setZero();
        return;
      }
      nmean_ = computeMean(new_normals);
      ncov_ = computeCov(new_normals);
      bool invertable = false;
      ncov_.computeInverseWithCheck(nicov_, invertable);
      if (!invertable) {
        dprintf("cell @ (%.2f, %.2f): ncov_ not invertable\n", center_.x,
                center_.y);
        nicov_.setZero();
      }
      nhasGaussian_ = true;
    }
  }

  void computeGaussian() {
    assert(points_.size() == normals_.size());
    N_ = points_.size();
    computePGaussian();
    computeNGaussian();
  }

  // Get and Set Points
  inline Eigen::Matrix2d getPointCov() const { return pcov_; }
  inline Eigen::Matrix2d getPointInverseCov() const { return picov_; }
  inline Eigen::Vector2d getPointMean() const { return pmean_; }
  inline Eigen::Matrix2d getPointEvecs() const { return pevecs_; }
  inline Eigen::Vector2d getPointEvals() const { return pevals_; }
  inline void setPointMean(const Eigen::Vector2d &mean) { pmean_ = mean; }
  inline void setPointCov(const Eigen::Matrix2d &cov) { pcov_ = cov; }
  inline void setPointEvals(const Eigen::Vector2d &ev) { pevals_ = ev; }
  inline void setPointEvecs(const Eigen::Matrix2d &ev) { pevecs_ = ev; }

  // Get and Set Normals
  inline Eigen::Matrix2d getNormalCov() const { return ncov_; }
  inline Eigen::Matrix2d getNormalInverseCov() const { return nicov_; }
  inline Eigen::Vector2d getNormalMean() const { return nmean_; }
  inline Eigen::Matrix2d getNormalEvecs() const { return nevecs_; }
  inline Eigen::Vector2d getNormalEvals() const { return nevals_; }
  inline void setNormalMean(const Eigen::Vector2d &mean) { nmean_ = mean; }
  inline void setNormalCov(const Eigen::Matrix2d &cov) { ncov_ = cov; }
  inline void setNormalEvals(const Eigen::Vector2d &ev) { nevals_ = ev; }
  inline void setNormalEvecs(const Eigen::Matrix2d &ev) { nevecs_ = ev; }

  void addPoint(const pcl::PointXY &pt) { points_.push_back(pt); }
  void addPoints(pcl::PointCloud<pcl::PointXY> &pt) {
    points_.insert(points_.begin(), pt.points.begin(), pt.points.end());
  }

  void addNormal(const pcl::PointXY &normal) { normals_.push_back(normal); }
  void addNormals(pcl::PointCloud<pcl::PointXY> &normals) {
    normals_.insert(normals_.begin(), normals.points.begin(), normals.points.end());
  }

  void setN(int N) { N_ = N; }
  int getN() { return N_; }

  void ToString() {
    dprintf("cell @ (%.2f, %.2f):\n", center_.x, center_.y);
    dprintf("  N: %ld\n", N_);
    dprintf("  phas: %s\n", (phasGaussian_ ? "true" : "false"));
    dprintf("  nhas: %s\n", (nhasGaussian_ ? "true" : "false"));
    dprintf("  dimen: (%.2f, %.2f)\n", xsize_, ysize_);
    dprintf("  pmean: (%.2f, %.2f)\n", pmean_(0), pmean_(1));
    dprintf("  pcova: (%.2f, %.2f, %.2f, %.2f)\n", pcov_(0, 0), pcov_(0, 1), pcov_(1, 0), pcov_(1, 1));
    dprintf("  nmean: (%.2f, %.2f)\n", nmean_(0), nmean_(1));
    dprintf("  ncova: (%.2f, %.2f, %.2f, %.2f)\n", ncov_(0, 0), ncov_(0, 1), ncov_(1, 0), ncov_(1, 1));
  }

 private:
  pcl::PointXY center_;
  double xsize_, ysize_;
  Eigen::Matrix2d pcov_;
  Eigen::Matrix2d picov_;
  Eigen::Matrix2d pevecs_;
  Eigen::Vector2d pmean_;
  Eigen::Vector2d pevals_;
  Eigen::Matrix2d ncov_;
  Eigen::Matrix2d nicov_;
  Eigen::Matrix2d nevecs_;
  Eigen::Vector2d nmean_;
  Eigen::Vector2d nevals_;
  unsigned int N_;
};
