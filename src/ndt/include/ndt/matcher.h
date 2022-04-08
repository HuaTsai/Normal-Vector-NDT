#pragma once
#include <ndt/nmap.h>
#include <ndt/options.h>
#include <ndt/timer.h>

class NDTMatcher {
 public:
  NDTMatcher() = delete;

  static NDTMatcher GetIter(std::unordered_set<Options> options,
                            std::vector<double> cell_sizes,
                            double d2 = 0.05);

  static NDTMatcher GetBasic(std::unordered_set<Options> options,
                             double cell_sizes,
                             double d2 = 0.05);

  bool HasOption(Options option);

  void SetTarget(const std::vector<Eigen::Vector3d> &points);

  void SetSource(const std::vector<Eigen::Vector3d> &points);

  Eigen::Affine3d Align(
      const Eigen::Affine3d &guess = Eigen::Affine3d::Identity());

  int corres() const { return corres_; }

  int iteration() const { return iteration_; }

  Timer timer() const { return timer_; }

  std::shared_ptr<NMap> smap() const { return smap_; }

  std::shared_ptr<NMap> tmap() const { return tmap_; }

  void set_intrinsic(double intrinsic) { intrinsic_ = intrinsic; }

  std::vector<Eigen::Affine3d> tfs() const { return tfs_; }

 private:
  explicit NDTMatcher(std::unordered_set<Options> options,
                      std::vector<double> cell_sizes,
                      double cell_size,
                      double d2,
                      double intrinsic);

  Eigen::Affine3d AlignImpl(
      const Eigen::Affine3d &guess = Eigen::Affine3d::Identity());

  std::unordered_set<Options> options_;
  std::vector<Eigen::Vector3d> spts_;
  std::vector<Eigen::Vector3d> tpts_;
  std::vector<Eigen::Affine3d> tfs_;
  std::shared_ptr<NMap> smap_;
  std::shared_ptr<NMap> tmap_;
  std::vector<double> cell_sizes_;
  Timer timer_;
  double cell_size_;
  double d2_;
  double intrinsic_;
  int iteration_;
  int corres_;
};
