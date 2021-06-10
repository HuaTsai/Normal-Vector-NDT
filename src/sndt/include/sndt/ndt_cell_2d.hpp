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
                    const Matrix2d &intrinsic) {
  Matrix2d ret;
  int n = points.size();
  if (n == 1) { return intrinsic; }
  MatrixXd mp(2, n);
  for (int i = 0; i < n; ++i)
    mp.block<2, 1>(0, i) = points.at(i) - mean;
  ret = mp * mp.transpose() / (n - 1) + intrinsic;
  return ret;
}

Matrix2d ComputeCov(const vector<Vector2d> &points, const Vector2d &mean) {
  Matrix2d intrinsic;
  intrinsic.setZero();
  return ComputeCov(points, mean, intrinsic);
}

Matrix2d ComputeCov(const vector<Vector2d> &points, const Vector2d &mean,
                    const vector<Matrix2d> &intrinsics) {
  Expects(points.size() == intrinsics.size());
  Matrix2d ret;
  int n = points.size();
  if (n == 1) { return intrinsics.at(0); }
  MatrixXd mp(2, n);
  for (int i = 0; i < n; ++i)
    mp.block<2, 1>(0, i) = points.at(i) - mean;
  ret = mp * mp.transpose() / (n - 1) + ComputeMean(intrinsics);
  return ret;
}

vector<Vector2d> ExcludeNaN(const vector<Vector2d> &points) {
  vector<Vector2d> ret;
  copy_if(points.begin(), points.end(), back_inserter(ret),
          [](const Vector2d &v) { return !isnan(v(0)) && !isnan(v(1)); });
  return ret;
}

class NDTCell {
 public:
  NDTCell() { InitializeVariables(); }

  void InitializeVariables() {
    phasgaussian_ = nhasgaussian_ = false;
    N_ = 0;
    center_.setZero(), size_.setZero();
    pcov_.setZero(), picov_.setZero(), pevecs_.setZero();
    pmean_.setZero(), pevals_.setZero();
    ncov_.setZero(), nicov_.setZero(), nevecs_.setZero();
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

  void ComputePGaussian() {
    if (!phasgaussian_) {
      auto valid_points = ExcludeNaN(points_);
      if (valid_points.size() == 0) {
        pmean_.setZero(), pcov_.setZero();
        return;
      }
      pmean_ = ComputeMean(valid_points);
      pcov_ = ComputeCov(valid_points, pmean_);
      bool invertible = false;
      pcov_.computeInverseWithCheck(picov_, invertible);
      if (!invertible)
        picov_.setZero();
      phasgaussian_ = true;
    }
  }

  void ComputeNGaussian() {
    if (!nhasgaussian_) {
      auto valid_normals = ExcludeNaN(normals_);
      if (!valid_normals.size()) {
        nmean_.setZero(), ncov_.setZero();
        return;
      }
      nmean_ = ComputeMean(valid_normals);
      ncov_ = ComputeCov(valid_normals, nmean_);
      bool invertible = false;
      ncov_.computeInverseWithCheck(nicov_, invertible);
      if (!invertible)
        nicov_.setZero();
      nhasgaussian_ = true;
    }
  }

  // AddPoint(s) and AddNormal(s)
  void AddPoint(const Vector2d &point) {
    points_.push_back(point);
  }

  void AddPoints(const vector<Vector2d> &points) {
    points_.insert(points_.end(), points.begin(), points.end());
  }

  void AddNormal(const Vector2d &normal) {
    normals_.push_back(normal);
  }

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
  Matrix2d GetPointInverseCov() const { return picov_; }
  Vector2d GetPointEvals() const { return pevals_; }
  Matrix2d GetPointEvecs() const { return pevecs_; }
  Vector2d GetNormalMean() const { return nmean_; }
  Matrix2d GetNormalCov() const { return ncov_; }
  Matrix2d GetNormalInverseCov() const { return nicov_; }
  Vector2d GetNormalEvals() const { return nevals_; }
  Matrix2d GetNormalEvecs() const { return nevecs_; }

  void SetN(int N) { N_ = N; }
  void SetPHasGaussian(bool phasgaussian) { phasgaussian_ = phasgaussian; }
  void SetNHasGaussian(bool nhasgaussian) { nhasgaussian_ = nhasgaussian; }
  void SetCenter(const Vector2d &center) { center_ = center; }
  void SetSize(const Vector2d &size) { size_ = size; }
  void SetPointMean(const Vector2d &mean) { pmean_ = mean; }
  void SetPointCov(const Matrix2d &cov) {
    pcov_ = cov;
    bool invertible = false;
    pcov_.computeInverseWithCheck(picov_, invertible);
    if (!invertible) { picov_.setZero(); }
  }
  void SetPointEvals(const Vector2d &evals) { pevals_ = evals; }
  void SetPointEvecs(const Matrix2d &evecs) { pevecs_ = evecs; }
  void SetNormalMean(const Vector2d &mean) { nmean_ = mean; }
  void SetNormalCov(const Matrix2d &cov) {
    ncov_ = cov;
    bool invertible = false;
    ncov_.computeInverseWithCheck(nicov_, invertible);
    if (!invertible) { nicov_.setZero(); }
  }
  void SetNormalEvals(const Vector2d &evals) { nevals_ = evals; }
  void SetNormalEvecs(const Matrix2d &evecs) { nevecs_ = evecs; }

 private:
  int N_;
  bool phasgaussian_, nhasgaussian_; 
  Vector2d center_, size_;
  Matrix2d pcov_, picov_, pevecs_;
  Vector2d pmean_, pevals_;
  Matrix2d ncov_, nicov_, nevecs_;
  Vector2d nmean_, nevals_;
  vector<Vector2d> points_, normals_;
};
