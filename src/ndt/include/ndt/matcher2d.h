#pragma once
#include <ndt/nmap2d.h>
#include <ndt/options.h>
#include <ndt/timer.h>

class NDTMatcher2D {
 public:
  NDTMatcher2D() = delete;

  // TODO: private constructor and make factory methods
  explicit NDTMatcher2D(std::unordered_set<Options> options,
                        double cell_size,
                        double d2 = 0.05);

  explicit NDTMatcher2D(std::unordered_set<Options> options,
                        std::vector<double> cell_sizes,
                        double d2 = 0.05);

  bool HasOption(Options option);

  void SetTarget(const std::vector<Eigen::Vector2d> &points);

  void SetSource(const std::vector<Eigen::Vector2d> &points);

  Eigen::Affine2d Align(
      const Eigen::Affine2d &guess = Eigen::Affine2d::Identity());

  int corres() const { return corres_; }

  int iteration() const { return iteration_; }

  Timer timer() const { return timer_; }

  std::shared_ptr<NMap2D> smap() const { return smap_; }

  std::shared_ptr<NMap2D> tmap() const { return tmap_; }

  void set_intrinsic(double intrinsic) { intrinsic_ = intrinsic; }

  std::vector<Eigen::Affine2d> tfs() const { return tfs_; }

 private:
  Eigen::Affine2d AlignImpl(
      const Eigen::Affine2d &guess = Eigen::Affine2d::Identity());

  std::unordered_set<Options> options_;
  std::vector<Eigen::Vector2d> spts_;
  std::vector<Eigen::Vector2d> tpts_;
  std::vector<Eigen::Affine2d> tfs_;
  std::shared_ptr<NMap2D> smap_;
  std::shared_ptr<NMap2D> tmap_;
  std::vector<double> cell_sizes_;
  Timer timer_;
  double cell_size_;
  double d2_;
  double intrinsic_;
  int iteration_;
  int corres_;
};
