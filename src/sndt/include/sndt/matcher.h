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

struct ICPParameters { };

struct GICPParameters { };

struct SICPParameters { };

struct NDTP2DParameters { };

struct NDTD2DParameters {
  NDTD2DParameters() {
    max_iterations = 100;
    threshold = 0.01;
    huber = 0;
    verbose = false;
  }
  int max_iterations;
  double threshold;
  double huber;
  bool verbose;
};

struct SNDTParameters {
  SNDTParameters() {
    max_iterations = 100;
    threshold = 0.01;
    huber = 0;
    verbose = false;
  }
  int max_iterations;
  double threshold;
  double huber;
  bool verbose;
};


// TODO: ICPMatch
// Eigen::Affine2d ICPMatch(
//     const std::vector<Eigen::Vector2d> &target_points,
//     const std::vector<Eigen::Vector2d> &source_points,
//     const ICPParameters &params,
//     const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

// TODO: GICPMatch
// Eigen::Affine2d GICPMatch(
//     const std::vector<Eigen::Vector2d> &target_points,
//     const std::vector<Eigen::Vector2d> &source_points,
//     const GICPParameters &params,
//     const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

// TODO: SICPMatch
// Eigen::Affine2d SICPMatch(
//     const std::vector<Eigen::Vector2d> &target_points,
//     const std::vector<Eigen::Vector2d> &source_points,
//     const SICPParameters &params,
//     const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

// TODO: NDTP2DMatch
// Eigen::Affine2d NDTP2DMatch(
//     const NDTMap &target_map, const std::vector<Eigen::Vector2d> &source_points,
//     const NDTD2DParameters &params,
//     const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d NDTD2DMatch(
    const NDTMap &target_map, const NDTMap &source_map,
    const NDTD2DParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d SNDTMatch(
    const SNDTMap &target_map, const SNDTMap &source_map,
    const SNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());
