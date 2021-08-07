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
#include <sndt/ndt_map.h>
#include <sndt/sndt_map.h>
#include <ceres/types.h>

enum class Converge {
  kNotConverge,
  kThreshold,
  kMaxIterations,
  kConsecutiveMinimum
};

struct UsedTime {
  UsedTime() {
    Initialize();
  }
  void Initialize() { total = optimize = others = 0; }
  int total;
  int optimize;
  int others;
};

// Variables started with "_" are output parameters
struct CommonParameters {
  CommonParameters() {
    InitializeInput();
    InitializeOutput();
  }
  void InitializeInput() {
    max_iterations = 100;
    max_min_ctr = 5;
    threshold = 0.01;
    huber = 1;
    verbose = false;
    solver = ceres::DENSE_QR;
  }
  void InitializeOutput() {
    _iteration = 0;
    _min_ctr = 0;
    _min_cost = std::numeric_limits<double>::max();
    _converge = Converge::kNotConverge;
    _costs.clear();
    _corres.clear();
    _usedtime.Initialize();
  }
  int max_iterations;
  int max_min_ctr;
  double threshold;
  double huber;
  bool verbose;
  ceres::LinearSolverType solver;

  int _iteration;
  int _min_ctr;
  double _min_cost;
  Converge _converge;
  std::vector<double> _costs;
  std::vector<int> _corres;
  UsedTime _usedtime;
};

// struct ICPParameters : CommonParameters { };

// struct GICPParameters : CommonParameters { };

struct SICPParameters : CommonParameters {
  SICPParameters() {
    radius = 1.5;
  }
  double radius;
};

// struct NDTP2DParameters { };

struct NDTD2DParameters : CommonParameters {
  NDTD2DParameters() {
    cell_size = 1.5;
    r_variance = 0.0625;
    t_variance = 0.0001;
  }
  double cell_size;
  double r_variance;  /**< radius variance, i.e., +-r -> (r / 3)^2 */
  double t_variance;  /**< angle variance, i.e., +-t -> (t / 3)^2 */
};

struct SNDTParameters : CommonParameters {
  SNDTParameters() {
    cell_size = 1.5;
    radius = 1.5;
    r_variance = 0.0625;
    t_variance = 0.0001;
  }
  double cell_size;
  double radius;
  double r_variance;  /**< radius variance, i.e., +-r -> (r / 3)^2 */
  double t_variance;  /**< angle variance, i.e., +-t -> (t / 3)^2 */
};

// TODO: ICPMatch
// Eigen::Affine2d ICPMatch(
//     const std::vector<Eigen::Vector2d> &target_points,
//     const std::vector<Eigen::Vector2d> &source_points,
//     ICPParameters &params,
//     const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

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

// TODO: NDTP2DMatch
// Eigen::Affine2d NDTP2DMatch(
//     const NDTMap &target_map, const std::vector<Eigen::Vector2d> &source_points,
//     NDTD2DParameters &params,
//     const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d NDTD2DMatch(
    const NDTMap &target_map, const NDTMap &source_map,
    NDTD2DParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d SNDTMatch(
    const SNDTMap &target_map, const SNDTMap &source_map,
    SNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());
