#pragma once
#include <ndt/nmap.h>
#include <ndt/options.h>
#include <ndt/timer.h>

class Matcher {
 public:
  explicit Matcher(std::unordered_set<Options> options)
      : options_(options), iteration_(0), corres_(0) {}

  void SetSource(const std::vector<Eigen::Vector3d> &points) { spts_ = points; }

  void SetTarget(const std::vector<Eigen::Vector3d> &points) { tpts_ = points; }

  virtual Eigen::Affine3d Align(
      const Eigen::Affine3d &guess = Eigen::Affine3d::Identity()) = 0;

  std::vector<Eigen::Affine3d> tfs() const { return tfs_; }

  Timer timer() const { return timer_; }

  int iteration() const { return iteration_; }

  int corres() const { return corres_; }

 protected:
  bool HasOption(Options option) { return options_.count(option); }

  std::unordered_set<Options> options_;
  std::vector<Eigen::Vector3d> spts_;
  std::vector<Eigen::Vector3d> tpts_;
  std::vector<Eigen::Affine3d> tfs_;
  Timer timer_;
  int iteration_;
  int corres_;
};

class NDTMatcher : public Matcher {
 public:
  NDTMatcher() = delete;

  static NDTMatcher GetIter(std::unordered_set<Options> options,
                            std::vector<double> cell_sizes,
                            double d2 = 0.05);

  static NDTMatcher GetBasic(std::unordered_set<Options> options,
                             double cell_size,
                             double d2 = 0.05);

  Eigen::Affine3d Align(
      const Eigen::Affine3d &guess = Eigen::Affine3d::Identity()) override;

  std::shared_ptr<NMap> smap() const { return smap_; }

  std::shared_ptr<NMap> tmap() const { return tmap_; }

  void set_intrinsic(double intrinsic) { intrinsic_ = intrinsic; }

 private:
  explicit NDTMatcher(std::unordered_set<Options> options,
                      std::vector<double> cell_sizes,
                      double cell_size,
                      double d2,
                      double intrinsic);

  Eigen::Affine3d AlignImpl(
      const Eigen::Affine3d &guess = Eigen::Affine3d::Identity());

  std::shared_ptr<NMap> smap_;
  std::shared_ptr<NMap> tmap_;
  std::vector<double> cell_sizes_;
  double cell_size_;
  double d2_;
  double intrinsic_;
};

class ICPMatcher : public Matcher {
 public:
  ICPMatcher() = delete;

  static ICPMatcher GetBasic(std::unordered_set<Options> options);

  Eigen::Affine3d Align(
      const Eigen::Affine3d &guess = Eigen::Affine3d::Identity()) override;

 private:
  explicit ICPMatcher(std::unordered_set<Options> options);
};

class SICPMatcher : public Matcher {
 public:
  SICPMatcher() = delete;

  static SICPMatcher GetBasic(std::unordered_set<Options> options,
                              double radius);

  Eigen::Affine3d Align(
      const Eigen::Affine3d &guess = Eigen::Affine3d::Identity()) override;

 private:
  explicit SICPMatcher(std::unordered_set<Options> options, double radius);
  std::vector<Eigen::Vector3d> snms_;
  std::vector<Eigen::Vector3d> tnms_;
  double radius_;
};
