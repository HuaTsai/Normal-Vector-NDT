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

enum Converge {
  kNotConverge,
  kThreshold,
  kMaxIterations,
  kConsecutiveMinimum
};

struct UsedTime {
  UsedTime() {
    others = search = optimize = 0;
  }
  int others;
  int search;
  int optimize;
};

struct CommonParameters {
  CommonParameters() {
    max_iterations = 100;
    threshold = 0.01;
    huber = 0;
    verbose = false;
  }
  int max_iterations;
  double threshold;
  double huber;
  bool verbose;
  UsedTime usedtime;
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
  NDTD2DParameters() {}
};

struct SNDTParameters : CommonParameters {
  SNDTParameters() {}
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
