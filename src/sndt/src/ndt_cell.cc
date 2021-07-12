#include "sndt/ndt_cell.h"

#include <bits/stdc++.h>

#include <Eigen/Eigen>
#include <gsl/gsl>

using namespace std;
using namespace Eigen;

Vector2d ComputeMean(const vector<Vector2d> &points) {
  Vector2d ret = Vector2d::Zero();
  for (const auto &vec : points) ret += vec;
  ret /= points.size();
  return ret;
}

Matrix2d ComputeMean(const vector<Matrix2d> &points) {
  Matrix2d ret = Matrix2d::Zero();
  for (const auto &mtx : points) ret += mtx;
  ret /= points.size();
  return ret;
}

// Compute covaraince, the intrinsic provides an offset
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

// Compute covariance, the size of intrinsics should equal to that of points
Matrix2d ComputeCov(const vector<Vector2d> &points, const Vector2d &mean,
                    const vector<Matrix2d> &intrinsics) {
  Expects(points.size() == intrinsics.size());
  return ComputeCov(points, mean, ComputeMean(intrinsics));
}

void ComputeEvalEvec(const Matrix2d &covariance, Vector2d &evals,
                     Matrix2d &evecs) {
  EigenSolver<Matrix2d> es(covariance);
  Matrix2d evals_ = es.pseudoEigenvalueMatrix();
  Matrix2d evecs_ = es.pseudoEigenvectors();
  if (evals_(0, 0) < evals_(1, 1)) {
    evals(0) = evals_(1, 1);
    evals(1) = evals_(0, 0);
    evecs.col(0) = evecs_.col(1);
    evecs.col(1) = evecs_.col(0);
  } else {
    evals = evals_.diagonal();
    evecs = evecs_;
  }
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

/********* NDTCell Definitions Start Here *********/

void NDTCell::InitializeVariables() {
  phasgaussian_ = nhasgaussian_ = false;
  N_ = 0;
  skew_rad_ = size_ = 0;
  center_.setZero();
  pcov_.setZero(), pevecs_.setZero();
  pmean_.setZero(), pevals_.setZero();
  ncov_.setZero(), nevecs_.setZero();
  nmean_.setZero(), nevals_.setZero();
}

void NDTCell::ComputeGaussian() {
  Expects(points_.size() == normals_.size());
  N_ = points_.size();
  ComputePGaussian();
  ComputeNGaussian();
}

void NDTCell::ComputePGaussian() {
  if (!phasgaussian_) {
    pmean_.setZero(), pcov_.setZero();
    auto valids = ExcludeNaNInf(points_);
    if (!valids.size()) { return; }
    pmean_ = ComputeMean(valids);
    pcov_ = ComputeCov(valids, pmean_);
    if (!pcov_.isZero())
      phasgaussian_ = true;
  }
}

void NDTCell::ComputeNGaussian() {
  if (!nhasgaussian_) {
    nmean_.setZero(), ncov_.setZero();
    auto valids = ExcludeNaNInf(normals_);
    if (!valids.size()) { return; }
    nmean_ = ComputeMean(valids);
    ncov_ = ComputeCov(valids, nmean_);
    if (!ncov_.isZero())
      nhasgaussian_ = true;
  }
}

void NDTCell::ComputeGaussianWithCovariances() {
  Expects(points_.size() == normals_.size());
  Expects(points_.size() == point_covs_.size());
  N_ = points_.size();
  ComputePGaussianWithCovariances();
  ComputeNGaussianWithCovariances();
}

void NDTCell::ComputePGaussianWithCovariances() {
  if (!phasgaussian_) {
    pmean_.setZero(), pcov_.setZero();
    auto valids = ExcludeNaNInf2(points_, point_covs_);
    if (valids.first.size() == 0) { return; }
    pmean_ = ComputeMean(valids.first);
    pcov_ = ComputeCov(valids.first, pmean_, valids.second);
    if (!pcov_.isZero()) {
      ComputeEvalEvec(pcov_, pevals_, pevecs_);
      phasgaussian_ = true;
    }
  }
}

void NDTCell::ComputeNGaussianWithCovariances() {
  if (!nhasgaussian_) {
    nmean_.setZero(), ncov_.setZero();
    auto valids = ExcludeNaNInf(normals_);
    if (!valids.size()) { return; }
    nmean_ = ComputeMean(valids);
    ncov_ = ComputeCov(valids, nmean_, Matrix2d::Identity() * 0.1);
    if (!ncov_.isZero()) nhasgaussian_ = true;
  }
}

void NDTCell::AddPoint(const Vector2d &point) { points_.push_back(point); }

void NDTCell::AddPointWithCovariance(const Vector2d &point,
                                     const Matrix2d &covariance) {
  points_.push_back(point);
  point_covs_.push_back(covariance);
}

void NDTCell::AddNormal(const Vector2d &normal) { normals_.push_back(normal); }

// void NDTCell::AddNormalWithCovariance(const Vector2d &normal,
//                                       const Matrix2d &covariance) {
//   normals_.push_back(normal);
//   normal_covs_.push_back(covariance);
// }

MatrixXd NDTCell::GetPointsMatrix() const {
  MatrixXd ret(2, points_.size());
  for (int i = 0; i < ret.cols(); ++i)
    ret.col(i) = points_[i];
  return ret;
}

MatrixXd NDTCell::GetNormalsMatrix() const {
  MatrixXd ret(2, normals_.size());
  for (int i = 0; i < ret.cols(); ++i)
    ret.col(i) = normals_[i];
  return ret;
}

string NDTCell::ToString() {
  stringstream ss;
  ss.setf(ios::fixed | ios::boolalpha);
  ss.precision(2);
  ss << "cell @ (" << center_(0) << ", " << center_(1) << "):" << endl
     << "  N: " << N_ << endl
     << "  𝓝p: " << phasgaussian_ << ", 𝓝n: " << nhasgaussian_ << endl
     << "  sz: " << size_ << endl
     << "  μp: (" << pmean_(0) << ", " << pmean_(1) << ")" << endl
     << "  Σp: (" << pcov_(0, 0) << ", " << pcov_(0, 1) << ", " << pcov_(1, 0)
     << ", " << pcov_(1, 1) << ")" << endl
     << "  μn: (" << nmean_(0) << ", " << nmean_(1) << ")" << endl
     << "  Σn: (" << ncov_(0, 0) << ", " << ncov_(0, 1) << ", " << ncov_(1, 0)
     << ", " << ncov_(1, 1) << ")" << endl;
  for (int i = 0; i < N_; ++i) {
    ss << "  p[" << i << "]: (" << points_[i](0) << ", " << points_[i](1) << ")"
       << endl
       << "  n[" << i << "]: (" << normals_[i](0) << ", " << normals_[i](1)
       << ")" << endl;
  }
  if (skew_rad_ != 0)
    ss << "  skewed cell with " << skew_rad_ << " rad" << endl;
  return ss.str();
}
