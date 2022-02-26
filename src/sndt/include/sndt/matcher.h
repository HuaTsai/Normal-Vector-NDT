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
