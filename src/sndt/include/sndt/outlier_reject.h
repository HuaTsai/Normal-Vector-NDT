#pragma once
#include <bits/stdc++.h>

#include <Eigen/Dense>

class RangeOutlierRejection {
 public:
  RangeOutlierRejection();

  void AddCorrespondence(const Eigen::Vector2d &p, const Eigen::Vector2d &q);

  std::vector<int> GetIndices();

  void set_reject(bool reject) { reject_ = reject; }
  void set_multiplier(double multiplier) { multiplier_ = multiplier; }

 private:
  bool reject_;
  double multiplier_;
  std::vector<double> distances_;
};

class AngleOutlierRejection {
 public:
  AngleOutlierRejection();

  void AddCorrespondence(const Eigen::Vector2d &np, const Eigen::Vector2d &nq);

  std::vector<int> GetIndices();

  void set_reject(bool reject) { reject_ = reject; }
  void set_multiplier(double multiplier) { multiplier_ = multiplier; }

 private:
  bool reject_;
  double multiplier_;
  std::vector<double> angles_;
};

class OutlierRejection {
 public:
  OutlierRejection();

  void AddCorrespondence(const Eigen::Vector2d &p,
                         const Eigen::Vector2d &q,
                         const Eigen::Vector2d &np,
                         const Eigen::Vector2d &nq);

  std::vector<int> GetIndices();

  void set_reject(bool reject) { reject_ = reject; }
  void set_multiplier(double multiplier) { multiplier_ = multiplier; }

 private:
  bool reject_;
  double multiplier_;
  std::vector<double> distances_;
};

enum class Rejection { kThreshold, kStatistic, kBoth };

class OutlierRejectionMaker {
 public:
  explicit OutlierRejectionMaker(int n) {
    for (int i = 0; i < n; ++i) indices_.push_back(i);
  }

  void RangeRejection(const std::vector<Eigen::Vector2d> &ps,
                      const std::vector<Eigen::Vector2d> &qs,
                      Rejection method,
                      const std::vector<double> &params);

  void AngleRejection(const std::vector<Eigen::Vector2d> &nps,
                      const std::vector<Eigen::Vector2d> &nqs,
                      Rejection method,
                      const std::vector<double> &params);

  std::vector<int> indices() { return indices_; }

 private:
  std::vector<Eigen::Vector2d> ps_, qs_, nps_, nqs_;
  std::vector<int> indices_;
};
