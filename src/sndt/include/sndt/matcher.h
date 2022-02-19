/**
 * @file matcher.h
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Declaration of Matching Algorithms
 * @version 0.1
 * @date 2021-07-29
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <ceres/types.h>
#include <sndt/ndt_map.h>
#include <sndt/sndt_map.h>
#include <sndt/used_time.h>

enum class Method { kDUMMY, kICP, kPt2PlICP, kSICP, kP2DNDT, kD2DNDT, kSNDT };

enum class FixStrategy { kNoFix, kFixX, kFixY, kFixTheta };

enum class Converge {
  kNotConverge,
  kThreshold,
  kMaxIterations,
  kConsecutiveMinimum,
  kBounce
};

// Variables started with "_" are output parameters
struct CommonParameters {
  CommonParameters() {
    InitializeInput();
    InitializeOutput();
  }

  void InitializeInput() {
    method = Method::kDUMMY;
    fixstrategy = FixStrategy::kNoFix;
    solver = ceres::DENSE_QR;
    max_iterations = 100;
    ceres_max_iterations = 50;
    threshold = 0.0001;
    threshold_t = 0.2;
    verbose = false;
    inspect = true;
    save_costs = false;
    reject = false;
  }

  void InitializeOutput() {
    _iteration = 0;
    _ceres_iteration = 0;
    _initial_cost = 0;
    _final_cost = 0;
    _converge = Converge::kNotConverge;
  }

  Method method;
  FixStrategy fixstrategy;
  ceres::LinearSolverType solver;
  int max_iterations;
  int ceres_max_iterations;
  double threshold;
  double threshold_t;
  bool verbose;
  bool inspect;
  bool save_costs;
  bool reject;

  int _iteration;
  int _ceres_iteration;
  double _initial_cost;
  double _final_cost;
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
  ICPParameters() { method = Method::kICP; }
};

struct Pt2plICPParameters : CommonParameters {
  Pt2plICPParameters() : radius(1.5) { method = Method::kPt2PlICP; }
  double radius;
};

// struct GICPParameters : CommonParameters { };

struct SICPParameters : CommonParameters {
  SICPParameters() : radius(1.5) { method = Method::kSICP; }
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
  P2DNDTParameters() { method = Method::kP2DNDT; }
};

struct D2DNDTParameters : NDTParameters {
  D2DNDTParameters() { method = Method::kP2DNDT; }
};

struct SNDTParameters : NDTParameters {
  SNDTParameters() : radius(1.5) { method = Method::kSNDT; }
  double radius;
};

Eigen::Affine2d ICPMatch(
    const std::vector<Eigen::Vector2d> &target_points,
    const std::vector<Eigen::Vector2d> &source_points,
    ICPParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d Pt2plICPMatch(
    const std::vector<Eigen::Vector2d> &target_points,
    const std::vector<Eigen::Vector2d> &source_points,
    Pt2plICPParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

// TODO: GICPMatch
// Eigen::Affine2d GICPMatch(
//     const std::vector<Eigen::Vector2d> &target_points,
//     const std::vector<Eigen::Vector2d> &source_points,
//     GICPParameters &params,
//     const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d SICPMatch(
    const std::vector<Eigen::Vector2d> &target_points,
    const std::vector<Eigen::Vector2d> &source_points,
    SICPParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

// BUG: This match does not work and it is awful.
Eigen::Affine2d P2DNDTMDMatch(
    const NDTMap &target_map,
    const std::vector<Eigen::Vector2d> &source_points,
    P2DNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d P2DNDTMatch(
    const NDTMap &target_map,
    const std::vector<Eigen::Vector2d> &source_points,
    P2DNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d D2DNDTMDMatch(
    const NDTMap &target_map,
    const NDTMap &source_map,
    D2DNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d D2DNDTMatch(
    const NDTMap &target_map,
    const NDTMap &source_map,
    D2DNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d SNDTMDMatch(
    const SNDTMap &target_map,
    const SNDTMap &source_map,
    SNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d SNDTMatch(
    const SNDTMap &target_map,
    const SNDTMap &source_map,
    SNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d SNDTMDMatch2(
    const NDTMap &target_map,
    const NDTMap &source_map,
    D2DNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d SNDTMatch2(
    const NDTMap &target_map,
    const NDTMap &source_map,
    D2DNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d SNDTCellMatch(
    const SNDTCell *target_cell,
    const SNDTCell *source_cell,
    SNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity(),
    int method = 0);
