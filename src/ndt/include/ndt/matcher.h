#pragma once
#include <ndt/nmap.h>
#include <ndt/timer.h>

class NDTMatcher {
 public:
  enum MatchType { kNDTLS, kNNDTLS, kNDTTR, kNNDTTR };

  explicit NDTMatcher(MatchType type, double cell_size, double d2 = 0.05);

  void SetTarget(const std::vector<Eigen::Vector3d> &points);

  void SetSource(const std::vector<Eigen::Vector3d> &points);

  Eigen::Affine3d Align(
      const Eigen::Affine3d &guess = Eigen::Affine3d::Identity());

  int corres() const { return corres_; }

  int iteration() const { return iteration_; }

  Timer timer() const { return timer_; }

  std::shared_ptr<NMap> smap() const { return smap_; }

  std::shared_ptr<NMap> tmap() const { return tmap_; }

 private:
  std::shared_ptr<NMap> smap_;
  std::shared_ptr<NMap> tmap_;
  MatchType type_;
  Timer timer_;
  double cell_size_;
  double d2_;
  int iteration_;
  int corres_;
};
