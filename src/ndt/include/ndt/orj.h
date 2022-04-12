#pragma once
#include <bits/stdc++.h>

#include <Eigen/Dense>

enum class Rejection { kThreshold, kStatistic, kBoth };

class Orj {
 public:
  Orj() = delete;

  explicit Orj(int n) {
    for (int i = 0; i < n; ++i) indices_.push_back(i);
  }

  void RangeRejection(const std::vector<Eigen::Vector3d> &ps,
                      const std::vector<Eigen::Vector3d> &qs,
                      Rejection method,
                      const std::vector<double> &params);

  void AngleRejection(const std::vector<Eigen::Vector3d> &nps,
                      const std::vector<Eigen::Vector3d> &nqs,
                      Rejection method,
                      const std::vector<double> &params);

  template <typename... Args>
  void RetainIndices(Args &... args) {
    // *this is captured by reference, no matter [&] or [=]
    auto retain = [&](auto &data) {
      int i = 0;
      for (auto id : indices_) data[i++] = data[id];
      data.resize(indices_.size());
    };
    (retain(args), ...);  // C++17: Fold expressions
  }

  std::vector<int> indices() { return indices_; }

 private:
  std::vector<Eigen::Vector3d> ps_, qs_, nps_, nqs_;
  std::vector<int> indices_;
};

class Orj2D {
 public:
  Orj2D() = delete;

  explicit Orj2D(int n) {
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

  template <typename... Args>
  void RetainIndices(Args &... args) {
    // *this is captured by reference, no matter [&] or [=]
    auto retain = [&](auto &data) {
      int i = 0;
      for (auto id : indices_) data[i++] = data[id];
      data.resize(indices_.size());
    };
    (retain(args), ...);  // C++17: Fold expressions
  }

  std::vector<int> indices() { return indices_; }

 private:
  std::vector<Eigen::Vector2d> ps_, qs_, nps_, nqs_;
  std::vector<int> indices_;
};
