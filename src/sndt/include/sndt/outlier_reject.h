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

  void AddCorrespondence(const Eigen::Vector2d &p, const Eigen::Vector2d &q,
                         const Eigen::Vector2d &np, const Eigen::Vector2d &nq);

  std::vector<int> GetIndices();

  void set_reject(bool reject) { reject_ = reject; }
  void set_multiplier(double multiplier) { multiplier_ = multiplier; }

 private:
  bool reject_;
  double multiplier_;
  std::vector<double> distances_;
};
