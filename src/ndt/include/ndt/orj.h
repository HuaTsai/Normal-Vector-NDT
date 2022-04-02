#pragma once
#include <bits/stdc++.h>

#include <Eigen/Dense>

class Orj {
 public:
  enum Rejection { kThreshold, kStatistic, kBoth };
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
