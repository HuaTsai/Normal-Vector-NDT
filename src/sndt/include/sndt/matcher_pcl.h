#pragma once
#include <Eigen/Dense>

Eigen::Affine2d PCLICP(
    const std::vector<Eigen::Vector2d> &target_points,
    const std::vector<Eigen::Vector2d> &source_points,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());

Eigen::Affine2d PCLNDT(
    const std::vector<Eigen::Vector2d> &target_points,
    const std::vector<Eigen::Vector2d> &source_points,
    const Eigen::Affine2d &guess_tf = Eigen::Affine2d::Identity());
