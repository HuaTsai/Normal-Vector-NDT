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
  if (n <= 2) { return intrinsic; }
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
  evals = es.pseudoEigenvalueMatrix().diagonal();
  evecs = es.pseudoEigenvectors();
}

vector<Vector2d> ExcludeNaNInf(const vector<Vector2d> &points) {
  vector<Vector2d> ret;
  copy_if(points.begin(), points.end(), back_inserter(ret),
          [](const Vector2d &v) { return v.allFinite(); });
  return ret;
}

// FIXME: not use now, we should not remove duplicates
vector<Vector2d> RemoveDuplicates(const vector<Vector2d> &points) {
  vector<Vector2d> ret;
  for (const auto &pt : points) {
    bool exist = false;
    for (const auto &retpt : ret)
      if (retpt == pt) {
        exist = true;
        break;
      }
    if (!exist)
      ret.push_back(pt);
  }
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

NDTCell::NDTCell() {
  phasgaussian_ = nhasgaussian_ = false;
  n_ = 0;
  skew_rad_ = size_ = 0;
  center_.setZero();
  pcov_.setZero(), pevecs_.setZero();
  pmean_.setZero(), pevals_.setZero();
  ncov_.setZero(), nevecs_.setZero();
  nmean_.setZero(), nevals_.setZero();
  mark = false;
}

void NDTCell::ComputeGaussian() {
  Expects(points_.size() == normals_.size());
  Expects(points_.size() == point_covs_.size());
  n_ = points_.size();
  ComputePGaussian();
  ComputeNGaussian();
}

void NDTCell::ComputePGaussian() {
  if (!phasgaussian_) {
    pmean_.setZero(), pcov_.setZero();
    auto valids = ExcludeNaNInf2(points_, point_covs_);
    if (valids.first.size() == 0) { return; }
    pmean_ = ComputeMean(valids.first);
    pcov_ = ComputeCov(valids.first, pmean_, valids.second);
    if (!pcov_.isZero()) {
      ComputeEvalEvec(pcov_, pevals_, pevecs_);
      if (pevals_(0) != 0 && pevals_(1) != 0)
        phasgaussian_ = true;
    }
  }
}

// TODO: set minimum eigen value
void NDTCell::ComputeNGaussian() {
  if (!nhasgaussian_) {
    nmean_.setZero(), ncov_.setZero();
    auto valids = ExcludeNaNInf(normals_);
    if (!valids.size()) { return; }
    nmean_ = ComputeMean(valids);
    // FIXME: Here we try to use cell that >= 3 points
    // ncov_ = ComputeCov(valids, nmean_, Matrix2d::Identity() * 0.01);
    ncov_ = ComputeCov(valids, nmean_);
    if (!ncov_.isZero()) {
      ComputeEvalEvec(ncov_, nevals_, nevecs_);
      if (nevals_(0) <= 0 || nevals_(1) <= 0)
        return;
      int maxidx, minidx;
      double maxval = nevals_.maxCoeff(&maxidx);
      double minval = nevals_.minCoeff(&minidx);
      if (maxval > 1000 * minval) {
        nevals_(minidx) = maxval / 1000.;
        ncov_ = nevecs_ * nevals_.asDiagonal() * nevecs_.transpose();
      }
      nhasgaussian_ = true;
    }
    // BUG
    if (ncov_.isZero() && valids.size() >= 3) {
      ncov_ = Matrix2d::Identity() * 0.01;
      nhasgaussian_ = true;
      mark = true;
    }
  }
}

void NDTCell::AddPoint(const Vector2d &point) { points_.push_back(point); }

void NDTCell::AddPointWithCovariance(const Vector2d &point,
                                     const Matrix2d &covariance) {
  points_.push_back(point);
  point_covs_.push_back(covariance);
}

void NDTCell::AddNormal(const Vector2d &normal) { normals_.push_back(normal); }

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
  char c[600];
  sprintf(c, "cell @ (%.2f, %.2f):\n"
             "   N: %d\n"
             "  𝓝p: %s, 𝓝n: %s\n"
             "  sz: %.2f\n"
             "  μp: (%.2f, %.2f)\n"
             "  Σp: (%.4f, %.4f, %.4f, %.4f)\n"
             " evp: (%.2f, %.2f)\n"
             " ecp: (%.2f, %.2f), (%.2f, %.2f)\n"
             "  μn: (%.2f, %.2f)\n"
             "  Σn: (%.4f, %.4f, %.4f, %.4f)\n"
             " evn: (%.2f, %.2f)\n"
             " ecn: (%.2f, %.2f), (%.2f, %.2f)\n"
             "skew: %.2f\n",
             center_(0), center_(1), n_,
             phasgaussian_ ? "true" : "false", nhasgaussian_ ? "true" : "false", size_,
             pmean_(0), pmean_(1), pcov_(0, 0), pcov_(0, 1), pcov_(1, 0), pcov_(1, 1),
             pevals_(0), pevals_(1), pevecs_(0, 0), pevecs_(1, 0), pevecs_(1, 0), pevecs_(1, 1),
             nmean_(0), nmean_(1), ncov_(0, 0), ncov_(0, 1), ncov_(1, 0), ncov_(1, 1),
             nevals_(0), nevals_(1), nevecs_(0, 0), nevecs_(1, 0), nevecs_(1, 0), nevecs_(1, 1),
             skew_rad_);
  stringstream ss;
  ss.setf(ios::fixed | ios::boolalpha);
  ss.precision(2);
  for (int i = 0; i < n_; ++i) {
    ss << "p[" << i << "]: (" << points_[i](0) << ", " << points_[i](1) << "), "
       << "n[" << i << "]: (" << normals_[i](0) << ", " << normals_[i](1) << ")" << endl;
  }
  return string(c) + ss.str();
}
