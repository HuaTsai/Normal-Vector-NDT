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
  // double inliner_ratio
};

struct SICPParameters {
  // TODO: SICPParameters
};

struct NDTD2DParameters {
  // TODO: NDTD2DParameters
};

Eigen::Affine2d SNDTMatch(
    const SNDTMap &target_map, const SNDTMap &source_map,
    const SNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());
