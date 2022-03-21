#pragma once
#include <nndt/nmap.h>

class NDTMatcher {
 public:
  enum MatchType { kNDT, kNNDT };

  explicit NDTMatcher(MatchType type, double cell_size);

  void SetTarget(const std::vector<Eigen::Vector3d> &points);

  void SetSource(const std::vector<Eigen::Vector3d> &points);

  Eigen::Affine3d Align(const Eigen::Affine3d &guess = Eigen::Affine3d::Identity());

  int iteration() const { return iteration_; }

 private:
  std::shared_ptr<NMap> smap;
  std::shared_ptr<NMap> tmap;
  MatchType type_;
  double cell_size_;
  double d2_;
  int iteration_;
};
