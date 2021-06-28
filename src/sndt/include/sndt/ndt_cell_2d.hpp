#pragma once

#include <bits/stdc++.h>
#include <Eigen/Eigen>
#include <gsl/gsl>

using namespace std;
using namespace Eigen;

Vector2d ComputeMean(const vector<Vector2d> &points) {
  Vector2d ret = Vector2d::Zero();
  for (const auto &vec : points)
    ret += vec;
  ret /= points.size();
  return ret;
}

Matrix2d ComputeMean(const vector<Matrix2d> &points) {
  Matrix2d ret = Matrix2d::Zero();
  for (const auto &mtx : points)
    ret += mtx;
  ret /= points.size();
  return ret;
}

Matrix2d ComputeCov(const vector<Vector2d> &points, const Vector2d &mean,
                    const Matrix2d &intrinsic = Matrix2d::Zero()) {
  Matrix2d ret;
  int n = points.size();
  if (n == 1) { return intrinsic; }
  MatrixXd mp(2, n);
  for (int i = 0; i < n; ++i)
    mp.col(i) = points.at(i) - mean;
  ret = mp * mp.transpose() / (n - 1) + intrinsic;
  return ret;
}

Matrix2d ComputeCov(const vector<Vector2d> &points, const Vector2d &mean,
                    const vector<Matrix2d> &intrinsics) {
  Expects(points.size() == intrinsics.size());
  return ComputeCov(points, mean, ComputeMean(intrinsics));
}

vector<Vector2d> ExcludeNaNInf(const vector<Vector2d> &points) {
  vector<Vector2d> ret;
  copy_if(points.begin(), points.end(), back_inserter(ret),
          [](const Vector2d &v) { return v.allFinite(); });
  return ret;
}

pair<vector<Vector2d>, vector<Matrix2d>> ExcludeNaNInf2(
    const vector<Vector2d> &points, const vector<Matrix2d> &covariances) {
  Expects(points.size() == covariances.size());
  pair<vector<Vector2d>, vector<Matrix2d>> ret;
  for (size_t i = 0; i < points.size(); ++i) {
    if (points[i].allFinite() && covariances[i].allFinite()) {
      ret.first.push_back(points[i]);
      ret.second.push_back(covariances[i]);
    }
  }
  return ret;
}

class NDTCell {
 public:
  NDTCell() { InitializeVariables(); }

  void InitializeVariables() {
    phasgaussian_ = nhasgaussian_ = false;
    N_ = 0;
    center_.setZero(), size_.setZero();
    pcov_.setZero(), pevecs_.setZero();
    pmean_.setZero(), pevals_.setZero();
    ncov_.setZero(), nevecs_.setZero();
    nmean_.setZero(), nevals_.setZero();
  }

  // Check if the given point is in the cell
  bool IsInside(const Vector2d &point) {
    return point(0) >= center_(0) - size_(0) / 2 &&
           point(0) <= center_(0) + size_(0) / 2 &&
           point(1) >= center_(1) - size_(1) / 2 &&
           point(1) <= center_(1) + size_(1) / 2;
  }

  // Compute gaussians
  void ComputeGaussian() {
    Expects(points_.size() == normals_.size());
    N_ = points_.size();
    ComputePGaussian();
    ComputeNGaussian();
  }

  // TODO: update eval and evec
  void ComputePGaussian() {
    if (!phasgaussian_) {
      auto valids = ExcludeNaNInf(points_);
      if (valids.size() == 0) {
        pmean_.setZero(), pcov_.setZero();
        return;
      }
      pmean_ = ComputeMean(valids);
      pcov_ = ComputeCov(valids, pmean_);
      if (!pcov_.isZero())
        phasgaussian_ = true;
    }
  }

  // TODO: update eval and evec
  void ComputeNGaussian() {
    if (!nhasgaussian_) {
      auto valids = ExcludeNaNInf(normals_);
      if (!valids.size()) {
        nmean_.setZero(), ncov_.setZero();
        return;
      }
      nmean_ = ComputeMean(valids);
      ncov_ = ComputeCov(valids, nmean_);
      if (!ncov_.isZero())
        nhasgaussian_ = true;
    }
  }

  // Compute gaussians with covariances
  void ComputeGaussianWithCovariances() {
    Expects(points_.size() == normals_.size());
    Expects(points_.size() == point_covs_.size());
    // Expects(normals_.size() == normal_covs_.size());
    N_ = points_.size();
    ComputePGaussianWithCovariances();
    ComputeNGaussianWithCovariances();
  }

  void ComputePGaussianWithCovariances() {
    if (!phasgaussian_) {
      auto valids = ExcludeNaNInf2(points_, point_covs_);
      if (valids.first.size() == 0) {
        pmean_.setZero(), pcov_.setZero();
        return;
      }
      pmean_ = ComputeMean(valids.first);
      pcov_ = ComputeCov(valids.first, pmean_, valids.second);
      if (!pcov_.isZero())
        phasgaussian_ = true;
    }
  }

  void ComputeNGaussianWithCovariances() {
    if (!nhasgaussian_) {
      auto valids = ExcludeNaNInf(normals_);
      if (!valids.size()) {
        nmean_.setZero(), ncov_.setZero();
        return;
      }
      nmean_ = ComputeMean(valids);
      ncov_ = ComputeCov(valids, nmean_, Matrix2d::Identity() * 0.1);
      if (!ncov_.isZero())
        nhasgaussian_ = true;
    }
  }

  // AddPoint(s) and AddNormal(s)
  void AddPoint(const Vector2d &point) {
    points_.push_back(point);
  }

  void AddPointWithCovariance(const Vector2d &point, const Matrix2d &covariance) {
    points_.push_back(point);
    point_covs_.push_back(covariance);
  }

  void AddPoints(const vector<Vector2d> &points) {
    points_.insert(points_.end(), points.begin(), points.end());
  }

  void AddNormal(const Vector2d &normal) {
    normals_.push_back(normal);
  }

  // FIXME
  // void AddNormalWithCovariance(const Vector2d &normal, const Matrix2d &covariance) {
  //   normals_.push_back(normal);
  //   normal_covs_.push_back(covariance);
  // }

  void AddNormals(const vector<Vector2d> &normals) {
    normals_.insert(normals_.end(), normals.begin(), normals.end());
  }

  // Debug string messages
  string ToString() {
    stringstream ss;
    ss.setf(ios::fixed | ios::boolalpha);
    ss.precision(2);
    ss << "cell @ (" << center_(0) << ", " << center_(1) << "):" << endl
       << "  N: " << N_ << endl
       << "  𝓝p: " << phasgaussian_ << ", 𝓝n: " << nhasgaussian_ << endl
       << "  sz: (" << size_(0) << ", " << size_(1) << ")" << endl
       << "  μp: (" << pmean_(0) << ", " << pmean_(1) << ")" << endl
       << "  Σp: (" << pcov_(0, 0) << ", " << pcov_(0, 1) << ", "
                    << pcov_(1, 0) << ", " << pcov_(1, 1) << ")" << endl
       << "  μn: (" << nmean_(0) << ", " << nmean_(1) << ")" << endl
       << "  Σn: (" << ncov_(0, 0) << ", " << ncov_(0, 1) << ", "
                    << ncov_(1, 0) << ", " << ncov_(1, 1) << ")" << endl;
    for (int i = 0; i < N_; ++i) {
      ss << "  p[" << i << "]: (" << points_[i](0) << ", " << points_[i](1) << ")" << endl
         << "  n[" << i << "]: (" << normals_[i](0) << ", " << normals_[i](1) << ")" << endl;
    }
    return ss.str();
  }

  // Get and Set methods of variables
  int GetN() const { return N_; }
  bool GetPHasGaussian() const { return phasgaussian_; }
  bool GetNHasGaussian() const { return nhasgaussian_; }
  bool BothHasGaussian() const { return phasgaussian_ && nhasgaussian_; }
  Vector2d GetCenter() const { return center_; }
  Vector2d GetSize() const { return size_; }
  Vector2d GetPointMean() const { return pmean_; }
  Matrix2d GetPointCov() const { return pcov_; }
  Vector2d GetPointEvals() const { return pevals_; }
  Matrix2d GetPointEvecs() const { return pevecs_; }
  Vector2d GetNormalMean() const { return nmean_; }
  Matrix2d GetNormalCov() const { return ncov_; }
  Vector2d GetNormalEvals() const { return nevals_; }
  Matrix2d GetNormalEvecs() const { return nevecs_; }

  void SetN(int N) { N_ = N; }
  void SetPHasGaussian(bool phasgaussian) { phasgaussian_ = phasgaussian; }
  void SetNHasGaussian(bool nhasgaussian) { nhasgaussian_ = nhasgaussian; }
  void SetCenter(const Vector2d &center) { center_ = center; }
  void SetSize(const Vector2d &size) { size_ = size; }
  void SetPointMean(const Vector2d &mean) { pmean_ = mean; }
  void SetPointCov(const Matrix2d &cov) { pcov_ = cov; }
  void SetPointEvals(const Vector2d &evals) { pevals_ = evals; }
  void SetPointEvecs(const Matrix2d &evecs) { pevecs_ = evecs; }
  void SetNormalMean(const Vector2d &mean) { nmean_ = mean; }
  void SetNormalCov(const Matrix2d &cov) { ncov_ = cov; }
  void SetNormalEvals(const Vector2d &evals) { nevals_ = evals; }
  void SetNormalEvecs(const Matrix2d &evecs) { nevecs_ = evecs; }

 private:
  int N_;
  bool phasgaussian_, nhasgaussian_; 
  Vector2d center_, size_;
  Matrix2d pcov_, pevecs_;
  Vector2d pmean_, pevals_;
  Matrix2d ncov_, nevecs_;
  Vector2d nmean_, nevals_;
  vector<Vector2d> points_, normals_;
  vector<Matrix2d> point_covs_;
  // FIXME
  // vector<Matrix2d> normal_covs_; 
};
