#pragma once

#include <bits/stdc++.h>
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

class NDTCell {
 public:
  NDTCell() { InitializeVariables(); }
  void InitializeVariables();

  // Compute Gaussians w/o Covariances
  void ComputeGaussian();
  void ComputePGaussian();
  void ComputeNGaussian();

  // Compute Gaussians w/ Covariances
  void ComputeGaussianWithCovariances();
  void ComputePGaussianWithCovariances();
  void ComputeNGaussianWithCovariances();

  // Add Point and Normal
  void AddPoint(const Vector2d &point);
  void AddPointWithCovariance(const Vector2d &point, const Matrix2d &covariance);
  void AddNormal(const Vector2d &normal);
  // FIXME: How does individual normal have covariance?
  // void AddNormalWithCovariance(const Vector2d &normal, const Matrix2d &covariance);

  // MatrixXd Getter of Points and Normals
  MatrixXd GetPointsMatrix() const;
  MatrixXd GetNormalsMatrix() const;

  // Debug Message
  string ToString();

  // Defined Get Methods of Variables
  int GetN() const { return N_; }
  bool GetPHasGaussian() const { return phasgaussian_; }
  bool GetNHasGaussian() const { return nhasgaussian_; }
  bool BothHasGaussian() const { return phasgaussian_ && nhasgaussian_; }
  double GetSkewRad() const { return skew_rad_; }
  double GetSize() const { return size_; }
  Vector2d GetCenter() const { return center_; }
  Vector2d GetPointMean() const { return pmean_; }
  Matrix2d GetPointCov() const { return pcov_; }
  Vector2d GetPointEvals() const { return pevals_; }
  Matrix2d GetPointEvecs() const { return pevecs_; }
  Vector2d GetNormalMean() const { return nmean_; }
  Matrix2d GetNormalCov() const { return ncov_; }
  Vector2d GetNormalEvals() const { return nevals_; }
  Matrix2d GetNormalEvecs() const { return nevecs_; }
  vector<Vector2d> GetPoints() const { return points_; }
  vector<Vector2d> GetNormals() const { return normals_; }

	// Defined Set Methods of Variables
  void SetN(int N) { N_ = N; }
  void SetPHasGaussian(bool phasgaussian) { phasgaussian_ = phasgaussian; }
  void SetNHasGaussian(bool nhasgaussian) { nhasgaussian_ = nhasgaussian; }
  void SetSkewRad(double skew_rad) { skew_rad_ = skew_rad; }
  void SetSize(double size) { size_ = size; }
  void SetCenter(const Vector2d &center) { center_ = center; }
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
  double skew_rad_, size_;
  Vector2d center_;
  Vector2d pmean_, pevals_, nmean_, nevals_;
  Matrix2d pcov_, pevecs_, ncov_, nevecs_;
  vector<Vector2d> points_, normals_;
  vector<Matrix2d> point_covs_;
  // FIXME: How does individual normal have covariance?
  // vector<Matrix2d> normal_covs_; 
};
