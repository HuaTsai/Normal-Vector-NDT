#pragma once
#include <bits/stdc++.h>
#include <ceres/ceres.h>
#include <sndt/used_time.h>

#include <Eigen/Dense>

enum class Converge {
  kNotConverge,
  kThreshold,
  kMaxIterations,
  kConsecutiveMinimum,
  kBounce,
  kNoCorrespondence
};

// Variables started with "_" are output parameters
struct CommonParameters {
  CommonParameters() {
    InitializeInput();
    InitializeOutput();
  }

  void InitializeInput() {
    // solver = ceres::DENSE_QR;
    solver = ceres::SPARSE_NORMAL_CHOLESKY;
    minimize = ceres::TRUST_REGION;
    max_iterations = 100;
    ceres_max_iterations = 50;
    threshold = 0.0001;
    threshold_t = 0.5;
    verbose = false;
    inspect = true;
    save_costs = false;
    reject = false;
  }

  void InitializeOutput() {
    _iteration = 0;
    _ceres_iteration = 0;
    _search_iteration = 0;
    _initial_cost = 0;
    _final_cost = 0;
    _opt_time = 0;
    _converge = Converge::kNotConverge;
  }

  ceres::LinearSolverType solver;
  ceres::MinimizerType minimize;
  int max_iterations;
  int ceres_max_iterations;
  double threshold;
  double threshold_t; /**< unit: degree */
  bool verbose;
  bool inspect;
  bool save_costs;
  bool reject;

  int _iteration;
  int _ceres_iteration;
  int _search_iteration;
  double _initial_cost;
  double _final_cost;
  double _opt_time; /**< optimize time in ms */
  Converge _converge;
  UsedTime _usedtime;

  /**< Iteration, (Before, After), works when save_costs is true */
  std::vector<std::pair<double, double>> _all_costs;
  /**< Iteration, Correspondence, (Before, After), works when save_costs is true
   */
  std::vector<std::vector<std::pair<double, double>>> _costs;
  /**< Iteration, Inneriteration, Solution, works when inspect is true */
  std::vector<std::vector<Eigen::Affine2d>> _sols;
  /**< Iteration, Correspondence, (Source, Target) */
  std::vector<std::vector<std::pair<int, int>>> _corres;
};

struct ICPParameters : CommonParameters {
  ICPParameters() {}
};

struct Pt2plICPParameters : CommonParameters {
  Pt2plICPParameters() : radius(1.5) {}
  double radius;
};

// struct GICPParameters : CommonParameters { };

struct SICPParameters : CommonParameters {
  SICPParameters() : radius(1.5) {}
  double radius;
};

struct NDTParameters : CommonParameters {
  NDTParameters() : cell_size(1.5), r_variance(0.0625), t_variance(0.0001) {}
  double cell_size;
  double r_variance; /**< radius variance, i.e., +-r -> (r / 3)^2 */
  double t_variance; /**< angle variance, i.e., +-t -> (t / 3)^2 */
  double d2;
};

struct P2DNDTParameters : NDTParameters {
  P2DNDTParameters() {}
};

struct D2DNDTParameters : NDTParameters {
  D2DNDTParameters() {}
};

struct SNDTParameters : NDTParameters {
  SNDTParameters() : radius(1.5) {}
  double radius;
};
