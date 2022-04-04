#pragma once
#include <ndt/nmap.h>
#include <ndt/timer.h>

class NDTMatcher {
 public:
  enum class Options {
    kLineSearch,
    kTrustRegion,
    kNDT,
    kNormalNDT,
    k1to1,
    k1ton,
    kIterative,
    kPointCov
  };

  NDTMatcher() = delete;

  // TODO: private constructor and make factory methods
  explicit NDTMatcher(std::unordered_set<Options> options,
                      double cell_size,
                      double d2 = 0.05);

  explicit NDTMatcher(std::unordered_set<Options> options,
                      std::vector<double> cell_sizes,
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

 private:
  Eigen::Affine3d AlignImpl(
      const Eigen::Affine3d &guess = Eigen::Affine3d::Identity());

  std::unordered_set<Options> options_;
  std::vector<Eigen::Vector3d> spts_;
  std::vector<Eigen::Vector3d> tpts_;
  std::shared_ptr<NMap> smap_;
  std::shared_ptr<NMap> tmap_;
  std::vector<double> cell_sizes_;
  Timer timer_;
  double cell_size_;
  double d2_;
  int iteration_;
  int corres_;
  bool orj_;
};

// XXX: Global variables in order for easy usage in applications
constexpr NDTMatcher::Options kLS = NDTMatcher::Options::kLineSearch;
constexpr NDTMatcher::Options kTR = NDTMatcher::Options::kTrustRegion;
constexpr NDTMatcher::Options kNDT = NDTMatcher::Options::kNDT;
constexpr NDTMatcher::Options kNNDT = NDTMatcher::Options::kNormalNDT;
constexpr NDTMatcher::Options k1to1 = NDTMatcher::Options::k1to1;
constexpr NDTMatcher::Options k1ton = NDTMatcher::Options::k1ton;
constexpr NDTMatcher::Options kIterative = NDTMatcher::Options::kIterative;
constexpr NDTMatcher::Options kPointCov = NDTMatcher::Options::kPointCov;
