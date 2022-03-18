#pragma once
#include <bits/stdc++.h>
#include <nav_msgs/Path.h>

struct Stat {
  explicit Stat(const std::vector<double> &vals) : data(vals) {
    if (data.empty()) {
      min = max = median = mean = stdev = rms = 0;
      return;
    }
    int n = data.size();
    std::sort(data.begin(), data.end());
    min = data.front();
    max = data.back();
    median = data[n / 2];
    mean = std::accumulate(data.begin(), data.end(), 0.) / n;
    double sum = 0;
    for (auto d : data) sum += (d - mean) * (d - mean);
    stdev = std::sqrt(sum / (n - 1));
    sum = 0;
    for (auto d : data) sum += d * d;
    rms = std::sqrt(sum / n);
  }
  void PrintResult() {
    std::printf(
        "min: %.2f, max: %.2f, med: %.2f, mean: %.2f, std: %.2f, rms: %.2f\n",
        min, max, median, mean, stdev, rms);
  }
  std::vector<double> data;
  double min;
  double max;
  double median;
  double mean;
  double stdev;
  double rms;
};

class TrajectoryEvaluation {
 public:
  enum EvalType { kDummy, kAbsolute, kRelativeBySingle, kRelativeByLength };

  TrajectoryEvaluation();

  std::pair<Stat, Stat> ComputeRMSError2D();

  void set_evaltype(const EvalType &evaltype) { evaltype_ = evaltype; }
  void set_estpath(const nav_msgs::Path &estpath) { estpath_ = estpath; }
  void set_gtpath(const nav_msgs::Path &gtpath) { gtpath_ = gtpath; }
  void set_length(double length) { length_ = length; }
  nav_msgs::Path estpath() const { return estpath_; }
  nav_msgs::Path gtpath() const { return gtpath_; }
  nav_msgs::Path align_estpath() const { return align_estpath_; }
  double gtlength() const { return gtlength_; }

 private:
  /**
   * @brief Absolute Trajectory Error
   * @details Implement from Sturm, Jürgen, et al. "A Benchmark for the
   * Evaluation of RGB-D SLAM Systems", 2012.
   */
  std::pair<Stat, Stat> AbsoluteTrajectoryError();

  /**
   * @brief Relative Pose Error by a consecutive matching
   * @details Implement from Sturm, Jürgen, et al. "A Benchmark for the
   * Evaluation of RGB-D SLAM Systems", 2012.
   */
  std::pair<Stat, Stat> RelativePoseErrorBySingle();

  /**
   * @brief Relative Pose Error by a fixed length
   * @details Implement from Sturm, Jürgen, et al. "A Benchmark for the
   * Evaluation of RGB-D SLAM Systems", 2012.
   */
  std::pair<Stat, Stat> RelativePoseErrorByLength();

  EvalType evaltype_;
  nav_msgs::Path estpath_;
  nav_msgs::Path gtpath_;
  nav_msgs::Path align_estpath_;
  nav_msgs::Path gtsync_;
  double length_;
  double gtlength_;
};

class TimeEvaluation {
 public:
 private:
};
