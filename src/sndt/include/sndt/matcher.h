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
#include <sndt/optimizer.h>
#include <sndt/sndt_map.h>

// Least Squares
Eigen::Affine2d ICPMatch(
    const std::vector<Eigen::Vector2d> &target_points,
    const std::vector<Eigen::Vector2d> &source_points,
    ICPParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d SICPMatch(
    const std::vector<Eigen::Vector2d> &target_points,
    const std::vector<Eigen::Vector2d> &source_points,
    SICPParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d DMatch(
    const NDTMap &target_map,
    const NDTMap &source_map,
    D2DNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d SMatch(
    const NDTMap &target_map,
    const NDTMap &source_map,
    D2DNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

// Generalized
Eigen::Affine2d IMatch(
    const std::vector<Eigen::Vector2d> &target_points,
    const std::vector<Eigen::Vector2d> &source_points,
    ICPParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d PMatch(
    const std::vector<Eigen::Vector2d> &target_points,
    const std::vector<Eigen::Vector2d> &source_points,
    SICPParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d D2DNDTMatch(
    const NDTMap &target_map,
    const NDTMap &source_map,
    D2DNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d SNDTMatch2(
    const NDTMap &target_map,
    const NDTMap &source_map,
    D2DNDTParameters &params,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

#include <sndt/matcher_arc.h>
